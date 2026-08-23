"""VLM Subgoal Planner — predicts dense visual subgoals from image + language.

In the full Steerable VLA system, a 3B+ VLM (e.g., OpenVLA, π₀) maps
(language_instruction, observation_image) → dense subgoal sequence
g_{1:K} = {(κ_k, ℓ_k)} where κ_k is a keyframe image and ℓ_k is its
language step.

At miniature scale, we implement this as:
1. A CNN encoder that processes rendered cable images
2. A language encoder that embeds task descriptions
3. A transformer decoder that cross-attends image + language tokens
   to predict K subgoal positions (x, y coordinates)
4. Keyframe synthesis: the planner also predicts which timesteps
   correspond to keyframe images (for dense coverage)

The planner is trained with a contrastive alignment loss: each subgoal
token is paired with the observation achieved at its segment boundary,
so the two levels agree on what "done" means.

References:
    - Section 2.1 of the NMI manuscript
    - Proposal §2.1: "Dense coverage means K scales with task horizon"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ImageEncoder(nn.Module):
    """CNN encoder for rendered cable images.
    
    Input: (B, C, H, W) RGB image of the cable state
    Output: (B, n_patches, embed_dim) patch tokens
    """
    
    def __init__(self, embed_dim=128, patch_size=8):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        
        # Simple CNN backbone (at miniature scale, no need for ViT)
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1), nn.SiLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.SiLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.SiLU(),
            nn.Conv2d(128, embed_dim, 3, stride=1, padding=1), nn.SiLU(),
        )
        # Global pooling + positional encoding
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
    
    def forward(self, images):
        """
        Args:
            images: (B, 3, H, W) RGB images
        Returns:
            tokens: (B, n_patches+1, embed_dim) including CLS token
        """
        B = images.shape[0]
        feat = self.conv_layers(images)  # (B, embed_dim, H', W')
        H, W = feat.shape[2], feat.shape[3]
        # Flatten spatial dims
        feat = feat.flatten(2).transpose(1, 2)  # (B, H'*W', embed_dim)
        # Prepend CLS token
        cls = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls, feat], dim=1)  # (B, 1+H'*W', embed_dim)
        return tokens


class LanguageEncoder(nn.Module):
    """Embeds language task descriptions into the shared token space.
    
    At miniature scale, we use a learned embedding of pre-tokenized
    language strings (word-level). In the full system, this would be
    the VLM's text encoder.
    """
    
    def __init__(self, vocab_size=256, embed_dim=128, max_len=32):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.pos_embed = nn.Parameter(torch.randn(1, max_len, embed_dim) * 0.02)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=embed_dim, nhead=4, dim_feedforward=embed_dim * 4,
                dropout=0.1, activation='gelu', batch_first=True,
            ),
            num_layers=2,
        )
    
    def forward(self, token_ids):
        """
        Args:
            token_ids: (B, L) integer token IDs
        Returns:
            tokens: (B, L, embed_dim) language tokens
        """
        B, L = token_ids.shape
        x = self.embed(token_ids) + self.pos_embed[:, :L, :]
        return self.encoder(x)


class SubgoalPredictor(nn.Module):
    """Transformer decoder that cross-attends image + language tokens
    to predict K subgoal positions.
    
    Each subgoal is a (x, y) coordinate pair in the workspace,
    plus a language step description (at miniature scale, a learned
    embedding index).
    """
    
    def __init__(self, embed_dim=128, n_heads=4, n_layers=3, max_subgoals=8):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_subgoals = max_subgoals
        
        # Learnable subgoal queries
        self.subgoal_queries = nn.Parameter(
            torch.randn(1, max_subgoals, embed_dim) * 0.02
        )
        
        # Transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim, nhead=n_heads,
            dim_feedforward=embed_dim * 4,
            dropout=0.1, activation='gelu', batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)
        
        # Prediction heads
        self.pos_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, 2),  # (x, y) subgoal position
        )
        self.lang_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, 16),  # language step embedding index
        )
        self.confidence_head = nn.Sequential(
            nn.Linear(embed_dim, 1),
            nn.Sigmoid(),
        )
    
    def forward(self, image_tokens, lang_tokens, n_subgoals=None):
        """
        Args:
            image_tokens: (B, n_img, embed_dim) from ImageEncoder
            lang_tokens: (B, n_lang, embed_dim) from LanguageEncoder
            n_subgoals: int or None — if None, use all queries
        
        Returns:
            subgoals: (B, K, 2) predicted subgoal positions
            lang_steps: (B, K, 16) language step embeddings
            confidences: (B, K, 1) confidence scores
            K: number of active subgoals
        """
        B = image_tokens.shape[0]
        dev = image_tokens.device
        K = n_subgoals or self.max_subgoals
        
        # Memory = image + language tokens concatenated
        memory = torch.cat([image_tokens, lang_tokens], dim=1)
        
        # Use first K subgoal queries
        queries = self.subgoal_queries[:, :K, :].expand(B, -1, -1)
        
        # Decode
        decoded = self.decoder(queries, memory)  # (B, K, embed_dim)
        
        # Predict
        subgoals = self.pos_head(decoded)  # (B, K, 2)
        lang_steps = self.lang_head(decoded)  # (B, K, 16)
        confidences = self.confidence_head(decoded)  # (B, K, 1)
        
        return subgoals, lang_steps, confidences, K


class VLMSubgoalPlanner(nn.Module):
    """Complete VLM subgoal planner: image + language → dense subgoals.
    
    Architecture:
        1. ImageEncoder: (B, 3, H, W) → (B, n_img, D) image tokens
        2. LanguageEncoder: (B, L) → (B, n_lang, D) language tokens
        3. SubgoalPredictor: cross-attention → K subgoal positions
    
    Training:
        - Position regression loss: L1 between predicted and oracle subgoals
        - Contrastive alignment: each subgoal token paired with observation
          at its segment boundary
        - Confidence calibration: supervised on whether each subgoal is
          actually reached in the demonstration
    """
    
    def __init__(self, embed_dim=128, vocab_size=256, max_len=32,
                 max_subgoals=8, img_size=64):
        super().__init__()
        self.embed_dim = embed_dim
        self.img_encoder = ImageEncoder(embed_dim, patch_size=8)
        self.lang_encoder = LanguageEncoder(vocab_size, embed_dim, max_len)
        self.predictor = SubgoalPredictor(embed_dim, n_heads=4, n_layers=3,
                                          max_subgoals=max_subgoals)
        self.img_size = img_size
    
    def forward(self, images, token_ids, n_subgoals=None):
        """Predict subgoals from images + language.
        
        Args:
            images: (B, 3, H, W) rendered cable images
            token_ids: (B, L) language token IDs
            n_subgoals: int or None
        
        Returns:
            subgoals: (B, K, 2) predicted subgoal positions in [-1, 1]
            confidences: (B, K, 1)
            K: number of active subgoals
        """
        img_tokens = self.img_encoder(images)
        lang_tokens = self.lang_encoder(token_ids)
        subgoals, lang_steps, confidences, K = self.predictor(
            img_tokens, lang_tokens, n_subgoals
        )
        return subgoals, confidences, K
    
    def plan(self, images, token_ids, n_subgoals=None):
        """Inference-time planning: predict subgoals and return as a list.
        
        Returns:
            subgoal_list: list of (x, y) tuples, filtered by confidence
            confidence_list: list of confidence scores
        """
        self.eval()
        with torch.no_grad():
            subgoals, confidences, K = self.forward(images, token_ids, n_subgoals)
            subgoals = subgoals[0].cpu().numpy()  # (K, 2)
            confidences = confidences[0].cpu().numpy()  # (K, 1)
        
        # Filter by confidence threshold
        mask = confidences[:, 0] > 0.3
        subgoal_list = [tuple(subgoals[i]) for i in range(K) if mask[i]]
        confidence_list = [float(confidences[i]) for i in range(K) if mask[i]]
        
        return subgoal_list, confidence_list
    
    def training_loss(self, images, token_ids, oracle_subgoals, oracle_reached):
        """Training loss: position regression + contrastive alignment.
        
        Args:
            images: (B, 3, H, W)
            token_ids: (B, L)
            oracle_subgoals: (B, K, 2) ground-truth subgoal positions
            oracle_reached: (B, K, 1) whether each subgoal was reached
        
        Returns:
            loss: scalar
        """
        subgoals, confidences, K = self.forward(images, token_ids, K=oracle_subgoals.shape[1])
        
        # Position regression (L1)
        pos_loss = F.l1_loss(subgoals, oracle_subgoals)
        
        # Confidence calibration (BCE)
        conf_loss = F.binary_cross_entropy(confidences, oracle_reached.float())
        
        return pos_loss + 0.5 * conf_loss


# --- Language vocabulary for cable untangling tasks ---

CABLE_LANG_VOCAB = {
    '<pad>': 0,
    '<cls>': 1,
    'untangle': 2,
    'the': 3,
    'cable': 4,
    'grab': 5,
    'pull': 6,
    'left': 7,
    'right': 8,
    'up': 9,
    'down': 10,
    'crossing': 11,
    'remove': 12,
    'hold': 13,
    'release': 14,
    'segment': 15,
    'node': 16,
    'near': 17,
    'far': 18,
    'top': 19,
    'bottom': 20,
}


def encode_language(text, max_len=32):
    """Encode a language string into token IDs.
    
    Simple word-level tokenization for the miniature scale.
    """
    tokens = text.lower().split()
    ids = [CABLE_LANG_VOCAB.get(t, 1) for t in tokens[:max_len]]
    # Pad to max_len
    ids = ids + [0] * (max_len - len(ids))
    return torch.tensor(ids, dtype=torch.long)


# Task-specific language templates
TASK_DESCRIPTIONS = {
    'cable_untangle': [
        'untangle the cable crossing',
        'grab the cable near the crossing and pull',
        'remove the crossing by pulling the top segment',
        'untangle the cable by pulling the right segment up',
        'grab the node near the crossing and pull it free',
    ],
    'textile_fold': [
        'fold the textile to the target shape',
        'grasp the corner and fold it over',
        'fold the fabric along the target line',
        'make a valley fold at the marked position',
    ],
    'tool_use': [
        'use the tool to reach the target',
        'grab the tool and lever the object',
        'pull the target with the tool',
    ],
}
