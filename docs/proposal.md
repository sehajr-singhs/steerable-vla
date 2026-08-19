# Compositional Generalization in Embodied Foundation Models via Multimodal Subgoal Prompting and Flow-Based Action Execution

**A research proposal and internship application package**

*Prepared for the Research Scientist Internship — Physical Intelligence (π-series / P0)*
*Author: Sehaj Singh · Independent Research · 2026*

> This document is the source of truth for the proposal. The website (`docs/index.html`)
> renders it; the two manuscripts (`docs/papers/nmi_paper.tex`, `docs/papers/ieee_paper.tex`)
> are compiled from this material. The statement of purpose is `docs/sop.md`; the execution
> plan is `docs/roadmap.md`.

---

## 1. TITLE & ABSTRACT

### Title

**Compositional Generalization in Embodied Foundation Models via Multimodal Subgoal Prompting and Flow-Based Action Execution**

### Abstract (Nature Machine Intelligence format)

Generalist robot policies have scaled to thousands of tasks, yet they still fail
the test that defines embodied intelligence: performing a *long-horizon* task on
an *unseen* object, with an *unseen* morphology, in a world that does not hold
still. Cables tangle in ways no training set contains; textiles shift under their
own weight; a hook that was never seen must still be used. Scaling monolithic
vision-language-action (VLA) weights alone does not produce compositional
generalization: the policy either memorizes the data distribution or loses the
spatial-temporal anchors that make execution reliable.

We argue that zero-shot generalization across unseen morphologies and chaotic
manipulation tasks is achieved not by scaling a single network, but by
**structuring the problem into a steerable hierarchy**: a high-level
vision-language model (VLM) that decomposes a task into dense, *visual* subgoals —
predicted keyframes and language steps — and a low-level continuous-time
flow-matching action expert that executes the segment between consecutive
subgoals at control frequency. Between these two levels sits the paper's core
contribution, the **Steerable Multimodal Conditioning (SMC) layer**: a gated
adapter that lets human corrections, world-model replans, and runtime constraint
setpoints inject steering signals into the action flow *mid-execution*, with a
provable Lipschitz bound on how far a steered trajectory may deviate from the
nominal one (no jerks, by construction). Around the whole loop we place a
**runtime formal-verification safety envelope**: a control-barrier-function
filter that certifies each executed action against joint limits, collision
margins, and contact-force bounds before it reaches the actuators.

We instantiate and evaluate this hierarchy on three deliberately chaotic,
high-entropy manipulation tasks — cable untangling, textile folding on shifting
fabric, and adaptive tool use in clutter — under a pre-registered zero-shot
protocol that holds out entire morphologies, stiffness regimes, and tool classes
from training. The proposal defines the architecture, the training and data-mixing
pipeline (teleoperation + procedural simulation + synthetic subgoal generation),
the baselines and ablations, and the safety analysis, so that every number in the
resulting manuscripts traces to a committed experiment artifact.

### Contribution statement

1. **A steerable two-level hierarchy for long-horizon manipulation**, in which a
   VLM produces dense visual subgoals and a flow-matching expert executes
   subgoal-bounded action segments — factorizing policy learning so that
   composition happens at the subgoal level, not inside a single weight matrix.
2. **The Steerable Multimodal Conditioning (SMC) layer**, a gated flow-field
   adapter with a formal smoothness guarantee: steering engages through a
   bounded-Lipschitz gate, and a Gronwall-type bound certifies that injected
   corrections change the executed trajectory by at most a computable envelope —
   steering without jerks, by construction.
3. **A runtime formal-verification safety envelope** (CBF-QP filter) around the
   flow expert, with forward-invariance guarantees and offline falsification,
   so that steerability never trades away actuator safety.
4. **Three chaotic benchmarks and a pre-registered zero-shot protocol** for
   cross-morphology, cross-material, cross-tool generalization, with
   intervention rate and jerk as first-class metrics alongside task success.

---

## 2. SYSTEM ARCHITECTURE & MATHEMATICAL FORMULATION

### 2.1 Notation and problem setting

Let an observation be $o_t = (I_t^{cam_1}, \dots, I_t^{cam_V}, q_t, \dot{q}_t, \tau_t)$ —
multi-view RGB-D images, proprioception (joint positions/velocities, end-effector
pose), and optional tactile readings — and let $\ell$ be a language instruction.
An **action chunk** is a window of future actions
$a_{t:t+H} = (a_t, \dots, a_{t+H-1}) \in \mathbb{R}^{H \times D}$, executed with a
receding horizon (execute $k$ steps, re-plan).

We parameterize actions in a **canonical cross-embodiment frame**: normalized
end-effector position deltas $\Delta p_{ee}$, yaw delta $\Delta \psi$, and a
binary gripper command, in the end-effector frame at chunk start. Joint-space
demonstrations are converted to this frame by forward kinematics, which is what
lets one flow expert serve many morphologies zero-shot.

### 2.2 The two-level hierarchy

**Level 1 — the semantic planner (VLM).** A 3B+ parameter vision-language model
$\Lambda_\phi$ ingests the instruction and current observation and emits a dense
subgoal sequence

$$
g_{1:K} = \Lambda_\phi(\ell, o_t), \qquad g_k = (\kappa_k, \ell_k),
$$

where $\kappa_k$ is a *visual subgoal* (a predicted keyframe image of the scene at
the moment subgoal $k$ is achieved) and $\ell_k$ is its natural-language step.
Keyframes are emitted as discrete VQ tokens in a frozen image-token vocabulary and
decoded to pixels for conditioning. Dense coverage means $K$ scales with task
horizon (an untangling task with three crossings gets $\sim$6–10 subgoals, not
one).

**Level 2 — the action expert (flow matching).** The low-level expert is a
continuous-time flow-matching model $\pi_\theta$ over action chunks, conditioned
on the VLM's subgoal latents, the observation stream, and any active steering
signals. Conditioning is via cross-attention over a token sequence
$c = [\text{VLM subgoal latents};\, \text{observation tokens};\, \text{steering tokens}]$
using the same backbone as the VLM's decoder (π0-style parameterization), with
separate action and time-token embeddings.

**The joint latent space.** Multimodal prompts (text + visual subgoals + runtime
constraints) are fused in a shared token space: text and constraint strings pass
through the language tokenizer; keyframes pass through the frozen vision encoder;
both are projected by the same cross-attention layers onto the action-latent
stream. A contrastive alignment loss over (subgoal token, achieved-observation
chunk) pairs makes the two levels agree on what "done" means at each segment
boundary, which is what makes the decomposition *composable* rather than
arbitrary.

### 2.3 Flow matching over action trajectories

Following the π-series formulation, we train the expert with **conditional flow
matching**. Let $x_1 \sim p_{\text{data}}$ be a demonstrated action chunk and
$x_0 \sim \mathcal{N}(0, I)$ an independent noise sample. Define the linear
interpolation path

$$
x_s = (1 - s)\, x_0 + s\, x_1, \qquad s \in [0, 1],
$$

whose velocity is the constant $x_1 - x_0$. The model learns the velocity field

$$
\mathcal{L}_{\text{CFM}}(\theta) \;=\; \mathbb{E}_{s \sim \mathcal{U}[0,1],\; (x_0, x_1) \sim q,\; c} \left[\, \left\|\, v_\theta(x_s, s, c) - (x_1 - x_0) \,\right\|^2 \,\right],
$$

where the conditioning $c$ includes the VLM subgoal latents, observations, and
steering tokens. At inference the trajectory is the ODE solution

$$
\frac{dx_s}{ds} = v_\theta(x_s, s, c), \qquad x_1 = x_0 + \int_0^1 v_\theta(x_s, s, c)\, ds,
$$

integrated with a few Euler/Heun steps (2–10), which is what makes the head fast
enough for 50 Hz receding-horizon control. Flow matching is preferred over
score-based diffusion because (i) few-step sampling preserves action smoothness at
control frequency, (ii) the marginal likelihood path is simple and stable to train
at scale, and (iii) the ODE formulation gives us a *velocity field we can analyze*
— the substrate for the steering and safety theory below.

### 2.4 The Steerable Multimodal Conditioning (SMC) layer

The core novelty: **the policy can be steered mid-execution without leaving the
flow's geometry.** Steering signals come from three sources:

- **Human corrections** — clicked keypoints, drag vectors on the current view, or
  a single "nudge" in end-effector space;
- **World-model replans** — a predictive model notices the object slipped, or a
  subgoal image becomes infeasible, and proposes a replacement subgoal;
- **Runtime constraint setpoints** — force/torque ceilings, joint limits, keep-out
  zones, updated during execution.

All of them enter through a single gated adapter on the velocity field. Write the
nominal field $v_\theta^N$ and let $u_j \in \mathbb{R}^{d_j}$ be steering signal
$j$ with neutral value $u_j^0$ (the value at which steering is inactive). The
steered field is

$$
v_\theta^{S}(x_s, s, c, u) \;=\; v_\theta^{N}(x_s, s, c)
\;+\; \sum_{j=1}^{J} \lambda_j(x_s, s) \, v_\theta^{j}(x_s, s, c, u_j),
$$

with the **anchoring constraint** $v_\theta^{j}(\cdot, \cdot, \cdot, u_j^0) = 0$ —
removing steering exactly recovers the nominal field — and gates
$\lambda_j = \sigma(w_j^\top \varphi(x_s, s))$ with bounded Lipschitz constant
$L_\lambda$. The gate is what makes steering *smooth*: a steering signal engages
softly over a window rather than as a step in action space.

**The no-jerk theorem.** Assume $v_\theta^S$ is $L_v$-Lipschitz in $x$ and the
steering contribution is bounded by $\| \sum_j \lambda_j v_\theta^j \| \le M$. If a
steering signal is applied over an interval of flow-time length $\Delta s$, then
the steered and nominal trajectories satisfy

$$
\left\| x_S(s) - x_N(s) \right\| \;\le\; \frac{M}{L_v}\left(e^{L_v \Delta s} - 1\right),
$$

by Grönwall's inequality. Because $x_S$ and $x_N$ coincide where steering is off
and the RHS is monotone in $\Delta s$, we get a *computable smoothness envelope*:
an operator can bound how far a correction may move the trajectory before
execution. Jerk, defined as $\max_t \| \dddot{p}_{ee}(t) \|$, is bounded by the
same quantity scaled by the inverse of the gate's rise time — steering is
continuous in flow time by construction, and the bound is verified numerically
during training and reported as a metric (Sec. 3.4).

### 2.5 The runtime formal-verification safety envelope

Around the steered flow sits a **safety filter that is verified, not hoped**.
Let the safe set be

$$
\mathcal{C} = \{\, x : h(x) \ge 0 \,\},
$$

defined by a control barrier function $h$ encoding joint limits, collision
margins, and contact-force ceilings. At each control step, with the flow expert's
candidate action $u_{\text{cmd}}$, the filter solves the convex QP

$$
u^{*} \;=\; \arg\min_{u \in \mathcal{U}} \;\tfrac12 \left\| u - u_{\text{cmd}} \right\|_{W}^2
\quad \text{s.t.} \quad
\dot{h}(x) + \alpha\, h(x) \;\ge\; 0,
$$

where $\dot{h} = L_f h + L_g h\, u$ and $\alpha$ is an extended class-$\mathcal{K}$
gain. Standard CBF theory (Ames et al.) gives **forward invariance**: if
$h(x(0)) \ge 0$, then $h(x(t)) \ge 0$ for all $t$ — the robot provably never
violates the safety set. The composition with SMC is the point of the architecture:
*steerability is admitted only through the filter*, so a human nudge, a replan, or
a constraint change may reshape the trajectory within the certified tube but never
outside it. We verify the envelope offline on simulated rollouts (falsification
search over steering amplitudes and timings) and report violations as a first-class
metric; the filter is a convex QP solvable in microseconds, so it runs inside the
50 Hz control loop.

### 2.6 Compositional generalization mechanism

Why should this structure generalize where a monolithic policy does not?

1. **Subgoal factorization.** The policy factorizes as
   $p(a \mid o, \ell, g_{1:K}) = \prod_k p(a^{[k]} \mid o, g_k, \ell)$: each flow
   segment is conditioned on *one* subgoal. Training data is segmented at subgoal
   boundaries (automatically, via the VLM's keyframe predictions), so the expert
   learns *local* skills — "reach the keyframe" — that compose across tasks.
2. **Synthetic subgoal coverage.** Subgoals are generated procedurally (VLM
   keyframe extraction from demos + language-conditioned image synthesis), so the
   *conditioning distribution* is denser than the demonstration distribution. The
   expert is trained on subgoals it has never seen reached — the compositional
   substrate for zero-shot tasks.
3. **Canonical actions.** The cross-embodiment frame (Sec. 2.1) means a skill
   learned on one arm transfers to another without reparameterization.

---

## 3. EXPERIMENTAL DESIGN & BENCHMARKS

Three tasks, chosen to be maximally *chaotic* — non-rigid, high-entropy, and
open-ended in a way that defeats both hardcoded logic and monolithic imitation.

### Task A — Cable Untangling (non-rigid object manipulation)

- **Setup.** A deformable cable of length 40–80 cm, placed in a crossed
  configuration (2–3 crossings), must be brought to a knot-free target state.
- **Unseen axes (zero-shot).** cable stiffness (bending modulus ×0.4–2.2), length,
  crossing topology (including crossing types absent from training), table
  friction, initial configuration sampled from a held-out procedural family.
- **Metrics.** untangle success (all crossings cleared, no new crossings created),
  crossing-count reduction, completion time, max jerk, safety violations.
- **Why it's chaotic.** The cable's state space is effectively infinite and its
  dynamics are history-dependent; a policy cannot memorize configurations, and
  contact-rich manipulation makes jerky, high-force actions fail visibly.

### Task B — Textile Folding (shifting, deformable materials)

- **Setup.** A fabric piece (towels, shirts, arbitrary polygonal sheets) must be
  folded to a target polyline; the fabric *shifts* during execution — under its
  own drape, from aeration, or from contact — so the subgoal images go stale and
  must be re-anchored.
- **Unseen axes.** fabric stiffness and friction (held-out material classes),
  initial fold geometry, target fold topology, grasp-point visibility.
- **Metrics.** fold success (IoU with target ≥ 0.8), grasp success rate, regrasp
  count, slippage events, completion time.
- **Why it's chaotic.** The state is a high-dimensional surface that changes
  between and during grasps; this task is the strongest test of *steerable*
  conditioning, because the world model's replan (Sec. 2.4) fires constantly.

### Task C — Adaptive Tool Use in Cluttered Environments

- **Setup.** A cluttered scene with multiple unknown tools (hooks, rakes, levers,
  scoops) and a goal (retrieve an object beyond reach, lever a lid, pull a cable).
- **Unseen axes.** tool morphology (held-out shape classes), clutter density,
  tool-object contact geometry, goal affordance composition.
- **Metrics.** task success, tool-selection accuracy, interventions per 100
  episodes, safety violations (contact force above threshold).
- **Why it's chaotic.** Correct action requires *semantic* inference (which tool,
  which affordance) fused with *geometric* execution (where to grasp, how to
  lever) — the exact division of labor the hierarchy is built for.

### Data-collection pipeline

1. **Teleoperation.** 80–120 expert episodes per task on a bimanual
   teleoperation rig (ALOHA-class) with multi-view RGB-D, proprioception, and
   optional tactile; joint-space trajectories converted to the canonical frame.
2. **Procedural simulation.** MuJoCo/Isaac-style simulators with aggressive domain
   randomization (materials, friction, mass, lighting, camera pose) generating an
   order of magnitude more episodes than teleop, with ground-truth subgoal
   boundaries for free.
3. **Synthetic subgoal generation.** Offline, a VLM extracts keyframes from demos;
   a language-conditioned image model synthesizes *novel* subgoal images (same
   scene, alternative configurations), which are rendered and fed back into
   training — the conditioning-coverage engine of Sec. 2.6.
4. **Data flywheel.** Deployment rollouts are scored (success / progress /
   smoothness / coverage) and curated; failures that came close are *relabeled* by
   the teleoperator and re-ingested — the curation loop that the author's prior
   work (DataFly) showed is the variable that decides whether the loop compounds.

### Zero-shot evaluation protocol (pre-registered)

- Held-out **morphologies**, **materials**, and **tool classes** never appear in
  any training set — not in sim, not in teleop, not in synthetic subgoals.
- 3 seeds × fixed evaluation start sets; every reported number regenerates from a
  committed experiment artifact.
- Primary metrics reported with intervention rate and max jerk as *costs*, so a
  policy cannot win by being dangerous.

### Baselines and ablations

| Baseline | Description |
|---|---|
| RT-2-style | VLM → discretized actions, no hierarchy |
| Diffusion Policy | flat diffusion over action chunks |
| ACT | action-chunking transformer, no flow |
| π0-style vanilla | flow-matching VLA, no subgoals, no SMC, no filter |
| Hierarchical LLM | text-only subgoals (no visual keyframes) + flow expert |
| **Ours (full)** | visual subgoal hierarchy + SMC + CBF envelope |

Ablations: −SMC (no steering), −safety filter (no QP gate), −visual subgoals
(text only), −dense subgoals (single keyframe), −canonical frame (joint-space
actions). The ablation set is the mechanism test: each component is claimed to
move a specific metric (SMC → intervention rate and jerk; filter → safety
violations; visual subgoals → zero-shot success; density → horizon length;
canonical frame → cross-morphology transfer).

---

## 4. INTERNSHIP STATEMENT OF PURPOSE

*Full text: [`docs/sop.md`](sop.md).* In brief: the pitch is not "I can write
Python for a robot arm." It is that the author has built — and published — the
data-side thesis of what makes embodied policies scale (curation loops, relabeling
deployment failures, label efficiency), has worked end-to-end with diffusion
policies and VLA architectures, has deployed policies on real hardware and
measured the intervention rate, and wants to spend an internship turning
steerability and safety from afterthoughts into first-class components of the
π-series stack.

## 5. NINETY-DAY RESEARCH ROADMAP

*Full week-by-week plan: [`docs/roadmap.md`](roadmap.md).* Thirteen weeks: baseline
reproduction → subgoal pipeline → SMC → safety envelope → sim ablations → data
flywheel → sim-to-real → three real-hardware task suites → zero-shot ablations →
paper writing (NMI + IEEE) and public release.
