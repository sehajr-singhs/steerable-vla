"""The low-level action expert: conditional flow matching over action chunks.

Faithful to the proposal at miniature scale:

  * CFM loss on linear interpolation paths x_s = (1-s) x0 + s x1 over the
    CONTINUOUS part of the action chunk (gripper deltas)
  * a discrete Bernoulli head for the grab sequence -- separate from the
    continuous flow, exactly as real VLA systems separate grasp decisions
    from motion (a 0/1 flag fit by regression collapses to sub-threshold
    values, which is why the two are decoupled here)
  * conditioning c = [obs tokens, subgoal tokens] fused in a shared encoder
  * SMC adapter: v^S = v^N + lambda(x_s, s, c) * v^J(x_s, s, c, u)
    - steering branch anchored: v^J(., u=0) ~= 0 via an auxiliary loss
    - gate lambda = sigmoid MLP, so corrections engage softly
  * inference: Euler integration of dx/ds = v^S from s=0 to 1, then
    grabs sampled from the Bernoulli head

Baselines reuse the same class: flow_flat (no subgoal, no steering),
ours_nofilter (no CBF at inference), ours_full (everything).
"""

import torch
import torch.nn as nn

from ..config import TrainConfig, DataConfig


def _d(x, device):
    return torch.as_tensor(x, dtype=torch.float32, device=device)


class FlowExpert(nn.Module):
    def __init__(self, dim_obs, dim_subgoal, dim_action, dim_steer=3,
                 cfg: TrainConfig = None):
        super().__init__()
        cfg = cfg or TrainConfig()
        h, L = cfg.hidden, cfg.latent
        # dim_action = H*3 (full chunk: H*2 deltas + H grab bits). The flow
        # runs over the CONTINUOUS deltas only; grabs are a discrete head.
        self.n_grabs = dim_action // 3
        self.dim_action = dim_action - self.n_grabs
        self.cond_obs = nn.Sequential(nn.Linear(dim_obs, h), nn.SiLU())
        self.cond_sub = nn.Sequential(nn.Linear(dim_subgoal, h), nn.SiLU())
        self.cond_fuse = nn.Sequential(nn.Linear(2 * h, L), nn.SiLU())
        self.vel_head = nn.Sequential(
            nn.Linear(self.dim_action + 1 + L, h), nn.SiLU(),
            nn.Linear(h, h), nn.SiLU(),
            nn.Linear(h, self.dim_action))
        self.steer_head = nn.Sequential(
            nn.Linear(self.dim_action + 1 + L + dim_steer, h), nn.SiLU(),
            nn.Linear(h, h), nn.SiLU(),
            nn.Linear(h, self.dim_action))
        self.gate = nn.Sequential(
            nn.Linear(self.dim_action + 1 + L, h), nn.SiLU(),
            nn.Linear(h, 1))
        # The discrete grab decision gets a fully INDEPENDENT conditioning
        # path: gradients from the continuous flow (which dominates the joint
        # loss) must not be able to corrupt the features the grasp head needs.
        # With a shared encoder the flow loss drifts `c` away from
        # grasp-relevant structure and the head collapses to "never grab"
        # (the 80/20 class prior) no matter the BCE weighting.
        self.grab_obs = nn.Sequential(nn.Linear(dim_obs, h), nn.SiLU())
        self.grab_sub = nn.Sequential(nn.Linear(dim_subgoal, h), nn.SiLU())
        self.grab_head = nn.Sequential(
            nn.Linear(2 * h, h), nn.SiLU(),
            nn.Linear(h, h), nn.SiLU(),
            nn.Linear(h, self.n_grabs))
        self.use_subgoal = True
        self.use_steering = True
        self.cfg = cfg

    def condition(self, obs, subgoal):
        o = self.cond_obs(obs)
        if subgoal is None or not self.use_subgoal:
            s = torch.zeros_like(o)
        else:
            s = self.cond_sub(subgoal)
        return self.cond_fuse(torch.cat([o, s], dim=-1))

    def velocity(self, x_s, s, c, u=None):
        """v^S(x_s, s, c, u) = v^N + lambda * v^J with anchoring at u=0."""
        if isinstance(s, torch.Tensor):
            s_t = s
            if s_t.dim() == 1:
                s_t = s_t.unsqueeze(1)
            if s_t.shape[0] == 1 and x_s.shape[0] > 1:
                s_t = s_t.expand(x_s.shape[0], -1)
        else:
            s_t = torch.full((x_s.shape[0], 1), s, dtype=torch.float32,
                             device=x_s.device)
        base = torch.cat([x_s, s_t, c], dim=-1)
        v_n = self.vel_head(base)
        if not self.use_steering or u is None:
            return v_n
        u_t = _d(u, x_s.device)
        if u_t.dim() == 1:
            u_t = u_t.unsqueeze(0).expand(x_s.shape[0], -1)
        lam = torch.sigmoid(self.gate(base))
        v_j = self.steer_head(torch.cat([base, u_t], dim=-1))
        return v_n + lam * v_j

    # ------------------------------------------------------------------
    # training
    # ------------------------------------------------------------------

    @staticmethod
    def _focal_bce(logit, target, gamma=2.0, alpha=0.75):
        """Focal BCE: upweights hard positives (alpha) and downweights easy
        negatives (gamma). Prevents the 80/20 class prior from collapsing
        the grab head to 'never grab'."""
        bce = torch.nn.functional.binary_cross_entropy_with_logits(
            logit, target, reduction='none')
        pt = torch.where(target == 1, torch.sigmoid(logit),
                         1 - torch.sigmoid(logit))
        focal = alpha * (1 - pt) ** gamma
        return (focal * bce).mean()

    def cfm_loss(self, obs, subgoal, chunk, nudge, rng, steer_prob):
        """CFM over continuous deltas + focal BCE over the grab sequence.

        chunk layout: [deltas (H*2) | grabs (H)] per row.
        """
        B, D = chunk.shape
        H = self.n_grabs
        x1 = chunk[:, : 2 * H]
        g1 = chunk[:, 2 * H:]                      # grab flags in {0, 1}
        x0 = torch.randn_like(x1)
        s = torch.rand(B, 1, device=x1.device)
        x_s = (1 - s) * x0 + s * x1
        target = x1 - x0
        c = self.condition(obs, subgoal)
        use_u = (rng.rand() < steer_prob) if self.use_steering else False
        if use_u:
            pred = self.velocity(x_s, s, c, u=nudge)
        else:
            pred = self.velocity(x_s, s, c, u=None)
        loss = torch.mean((pred - target) ** 2)
        # discrete head: focal BCE over the grab sequence, on its own path
        go = self.grab_obs(obs)
        if subgoal is not None and self.use_subgoal:
            gs = self.grab_sub(subgoal)
        else:
            gs = torch.zeros_like(go)
        g_logit = self.grab_head(torch.cat([go, gs], dim=-1))
        loss = loss + self._focal_bce(g_logit, g1, gamma=2.0, alpha=0.75)
        # anchoring auxiliary loss: steering branch must vanish at u=0
        if self.use_steering:
            u0 = torch.zeros_like(nudge)
            s_t = torch.full((B, 1), 0.5, device=x1.device)
            base = torch.cat([x_s.detach(), s_t, c.detach()], dim=-1)
            lam = torch.sigmoid(self.gate(base))
            v_j0 = self.steer_head(torch.cat([base, u0], dim=-1))
            loss = loss + 0.25 * torch.mean((lam * v_j0) ** 2)
        return loss

    # ------------------------------------------------------------------
    # inference
    # ------------------------------------------------------------------

    def _sample(self, obs, subgoal, nudge, flow_steps):
        """One Euler integration of the flow from s=0 to 1 (deltas only)."""
        dev = next(self.parameters()).device
        D = self.dim_action
        x = torch.randn(1, D, device=dev)
        c = self.condition(_d(obs, dev).unsqueeze(0),
                           _d(subgoal, dev).unsqueeze(0))
        ds = 1.0 / flow_steps
        for k in range(flow_steps):
            s = k * ds
            v = self.velocity(x, s, c, u=nudge)
            x = x + ds * v
        return x[0]

    @torch.no_grad()
    def act(self, obs, subgoal, nudge=None, flow_steps=None, n_samples=None):
        """Integrate the flow (n_samples averaged), sample grabs, return chunk.

        Averaging several independent flow samples is the standard variance
        reduction for sampling-based action generation: each sample is a
        Monte-Carlo draw from the conditional action distribution, and the
        mean is far less jerky than any single draw.
        """
        self.eval()
        dev = next(self.parameters()).device
        flow_steps = flow_steps or self.cfg.flow_steps
        if n_samples is None:
            n_samples = getattr(self.cfg, "n_samples", 1)
        x = torch.stack([self._sample(obs, subgoal, nudge, flow_steps)
                         for _ in range(n_samples)]).mean(0)
        o = _d(obs, dev).unsqueeze(0)
        s_ = _d(subgoal, dev).unsqueeze(0)
        go = self.grab_obs(o)
        gs = self.grab_sub(s_)
        grabs = (self.grab_head(torch.cat([go, gs], dim=-1)) > 0).float()
        return torch.cat([x, grabs[0]], dim=-1).cpu().numpy()


class BCPolicy(nn.Module):
    """Behavioral-cloning baseline: flat MLP over chunks (no flow).

    Same discrete/continuous split as the flow expert: deltas by regression,
    grab sequence by a Bernoulli head.
    """

    def __init__(self, dim_obs, dim_subgoal, dim_action, cfg: TrainConfig = None):
        super().__init__()
        cfg = cfg or TrainConfig()
        h = cfg.hidden
        self.n_grabs = dim_action // 3
        self.net = nn.Sequential(
            nn.Linear(dim_obs + dim_subgoal, h), nn.SiLU(),
            nn.Linear(h, h), nn.SiLU(),
            nn.Linear(h, dim_action))
        self.use_subgoal = True

    def forward(self, obs, subgoal, g1=None):
        if subgoal is None or not self.use_subgoal:
            subgoal = torch.zeros_like(obs)
        out = self.net(torch.cat([obs, subgoal], dim=-1))
        H = self.n_grabs
        deltas = out[:, : 2 * H]
        g_logit = out[:, 2 * H:]
        if g1 is None:
            return deltas
        loss = torch.mean((deltas - g1[:, : 2 * H]) ** 2)
        loss = loss + torch.nn.functional.binary_cross_entropy_with_logits(
            g_logit, g1[:, 2 * H:], pos_weight=torch.tensor(4.0))
        return loss

    def act(self, obs, subgoal, **kw):
        self.eval()
        with torch.no_grad():
            dev = next(self.parameters()).device
            o = _d(obs, dev).unsqueeze(0)
            s = _d(subgoal, dev).unsqueeze(0)
            out = self.net(torch.cat([o, s], dim=-1))[0]
            H = self.n_grabs
            deltas = out[: 2 * H]
            grabs = (out[2 * H:] > 0).float()
            return torch.cat([deltas, grabs], dim=-1).cpu().numpy()


def make_policy(kind, dims, tcfg: TrainConfig, dcfg: DataConfig):
    """Factory for all study variants.
    
    Variants:
      bc: Behavioral cloning (flat MLP, no flow, no subgoals)
      flow_flat: Flow matching without subgoals or steering
      ours_nofilter: Flow + SMC + subgoals, no CBF filter at inference
      ours_full: Flow + SMC + subgoals + CBF filter (full system)
      rt2: RT-2 style VLM → discretized actions (no hierarchy)
      diffusion: Diffusion Policy (flat denoising, no hierarchy)
      act: Action Chunking Transformer with CVAE
      transformer: Self-attention over nodes (previous baseline)
    """
    dim_obs, dim_sub, dim_act = dims
    if kind == "bc":
        p = BCPolicy(dim_obs, dim_sub, dim_act, tcfg)
        p.use_subgoal = False
        return p
    if kind == "flow_flat":
        p = FlowExpert(dim_obs, dim_sub, dim_act, cfg=tcfg)
        p.use_subgoal = False
        p.use_steering = False
        return p
    if kind == "ours_nofilter":
        p = FlowExpert(dim_obs, dim_sub, dim_act, cfg=tcfg)
        return p
    if kind == "ours_full":
        p = FlowExpert(dim_obs, dim_sub, dim_act, cfg=tcfg)
        return p
    if kind == "rt2":
        from ..baselines.rt2_policy import RT2Policy
        p = RT2Policy(dim_obs=dim_obs, dim_subgoal=dim_sub, dim_action=dim_act,
                      embed_dim=tcfg.hidden)
        p._ensure_built(dim_obs + dim_sub)  # force-build layers now
        return p
    if kind == "diffusion":
        from ..baselines.diffusion_policy import DiffusionPolicy
        return DiffusionPolicy(dim_obs, dim_sub, dim_act, hidden=tcfg.hidden)
    if kind == "act":
        from ..baselines.act_policy import ACTPolicy
        return ACTPolicy(dim_obs, dim_sub, dim_act, embed_dim=tcfg.hidden)
    if kind == "transformer":
        from .transformer_policy import TransformerPolicy
        n_nodes = (dim_obs - 4) // 2
        p = TransformerPolicy(n_nodes, dim_obs, hidden=tcfg.hidden, lr=tcfg.lr)
        return p
    raise ValueError(kind)


def train_policy(policy, dataset, tcfg: TrainConfig, dcfg: DataConfig,
                 device, epochs=None, seed=0, kind="ours_full",
                 curriculum_fn=None, verbose=False):
    """Train a policy with cosine LR schedule and optional curriculum.

    curriculum_fn(epoch) -> float: returns the crossing_target for this epoch.
    When provided, the caller re-generates data at the new difficulty level.
    When not provided, we just train on the provided data.
    """
    import math
    import numpy as _np
    torch.manual_seed(seed)
    rng = _np.random.RandomState(seed)
    epochs = epochs or tcfg.epochs
    # Some policies (TransformerPolicy) have their own optimizer
    if hasattr(policy, '_opt'):
        opt = policy._opt
    else:
        opt = torch.optim.Adam(policy.parameters(), lr=tcfg.lr)
    # Cosine annealing schedule
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=epochs, eta_min=getattr(tcfg, 'lr_min', 1e-5))
    obs = torch.as_tensor(_np.array([it[0] for it in dataset]), device=device)
    sub = torch.as_tensor(_np.array([it[1] for it in dataset]), device=device)
    chunk = torch.as_tensor(_np.array([it[2] for it in dataset]), device=device)
    nudge = torch.as_tensor(_np.array([it[3] for it in dataset]), device=device)
    n = len(dataset)
    if n == 0:
        return policy
    policy.train()
    best_loss = float('inf')
    for ep in range(epochs):
        perm = rng.permutation(n)
        tot = 0.0
        nbatch = 0
        for b in range(0, n, tcfg.batch):
            idx = perm[b:b + tcfg.batch]
            ob, su, ch, nu = obs[idx], sub[idx], chunk[idx], nudge[idx]
            # TransformerPolicy handles its own optimization internally
            if hasattr(policy, 'train_step') and not isinstance(policy, (FlowExpert, BCPolicy)):
                loss = policy.train_step(ob, ch)
                tot += float(loss) * len(idx)
                nbatch += 1
                continue
            opt.zero_grad()
            if isinstance(policy, BCPolicy):
                loss = policy.forward(ob, su, g1=ch)
            elif hasattr(policy, 'cfm_loss'):
                loss = policy.cfm_loss(ob, su, ch, nu, rng, dcfg.steer_prob)
            else:
                # New baselines (RT2, Diffusion, ACT) have their own forward
                try:
                    loss = policy.forward(ob, su, actions=ch)
                except TypeError:
                    try:
                        loss = policy.forward(ob, su, g1=ch)
                    except TypeError:
                        loss = policy.forward(ob, su)
            loss.backward()
            if tcfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(policy.parameters(), tcfg.grad_clip)
            opt.step()
            tot += float(loss.detach()) * len(idx)
            nbatch += 1
        scheduler.step()
        avg = tot / max(1, n)
        if avg < best_loss:
            best_loss = avg
        if verbose and ep % 50 == 0:
            lr_now = opt.param_groups[0]['lr']
            print(f"  ep {ep:4d}/{epochs}  loss={avg:.5f}  lr={lr_now:.2e}")
    return policy
