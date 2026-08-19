# 90-Day Research Roadmap

*Thirteen weeks. Three milestones. Two manuscripts. Every number traceable to a
committed artifact.*

**Milestone gate:** M1 (end of W4) — SMC steering demo in sim + safety filter
running in the control loop. M2 (end of W6) — full sim ablation suite on all three
tasks. M3 (end of W13) — real-hardware results, NMI + IEEE manuscripts submitted,
public repo release.

---

## Phase I — Foundation (W1–W3)

### Week 1 — Baseline reproduction and infra
- Reproduce a π0-style flow-matching VLA baseline (open weights) on the
  Open X-Embodiment subset + LIBERO-style sim tasks; verify 50 Hz receding-horizon
  execution.
- Stand up the evaluation harness: fixed start sets, 3 seeds, intervention-rate
  and jerk logging from day one (metrics that are cheap now, impossible to add
  retroactively).
- Inventory hardware: teleop rig calibration (ALOHA-class), camera extrinsic
  registration, proprioception log format. Decide the canonical action frame
  (Sec. 2.1) and write the FK converter.
- **Deliverable:** reproducing baseline; harness green; calibration report.

### Week 2 — Teleop data + subgoal extraction pipeline
- Collect first 50 expert episodes across the three tasks (untangling, folding,
  tool use), logged with raw multi-view video for offline keyframing.
- Build the offline subgoal extractor: VLM keyframe selection on demo video →
  language-step labeling → (optionally) diffusion-based synthesis of *novel*
  subgoal images.
- Segment action chunks at subgoal boundaries; sanity-check that segmentation is
  stable across repeated demos of the same task.
- **Deliverable:** 50 episodes; subgoal extractor producing (keyframe, text) pairs
  with a coverage histogram per task.

### Week 3 — Procedural sim + subgoal-conditioned training
- Procedural simulators for all three tasks with aggressive domain randomization
  (materials, friction, mass, lighting, camera pose); ground-truth subgoal
  boundaries emitted by the simulator.
- First subgoal-conditioned training runs: each flow segment conditioned on one
  visual subgoal; contrastive alignment loss (Sec. 2.2) between subgoal tokens and
  achieved-observation chunks.
- Scale sweep: subgoal density K vs. horizon; does dense subgoal coverage
  monotonically help beyond a floor?
- **Deliverable:** sim suite; training loop green; first density-vs-horizon curve.

## Phase II — Steerability and safety (W4–W6)

### Week 4 — SMC layer v1 + safety filter
- Implement the SMC gated adapter (Sec. 2.4): steering tokens, gate
  parameterization, anchoring constraint, and the Grönwall-bound calculator.
- Train with *synthetic* steering supervision: random trajectory perturbations
  relabeled as steering signals, so the adapter learns "absorb a correction,
  return to nominal" without needing human-in-the-loop during training.
- Implement the CBF-QP filter (joint limits, collision margin, force ceiling);
  measure solve time in the 50 Hz loop; offline falsification search over steering
  amplitudes/timings.
- **M1 milestone.** **Deliverable:** steering demo in sim (human nudge + world
  model replan + constraint setpoint), filter running, falsification report.

### Week 5 — Sim evaluation suite, round 1
- All three tasks × held-out difficulty splits; baselines (RT-2-style, Diffusion
  Policy, ACT, vanilla π0-style, text-only-subgoal hierarchical); our full system.
- First ablation pass: −SMC, −filter, −visual subgoals, −dense subgoals,
  −canonical frame.
- **Deliverable:** leaderboard v1 with intervention rate + jerk as costs.

### Week 6 — Data flywheel on sim deployment
- Deploy the policy in sim, run the curation loop from the DataFly thesis:
  score rollouts, curate near-misses, relabel failures with the teleoperator
  (teleop rig doubles as oracle), retrain, re-measure.
- Vary the curation rule (self vs. curated-relabel vs. blind-relabel) to confirm
  the loop's compounding law transfers from the toy study to the VLA setting.
- **M2 milestone.** **Deliverable:** flywheel curves; ablation suite finalized.

## Phase III — Sim-to-real and hardware (W7–W10)

### Week 7 — Sim-to-real bridge
- Domain-randomization tuning against real teleop episodes; real-data mixing
  sweep (0%, 5%, 10%, 20% real teleop in the batch); subgoal supervision on real
  video.
- **Deliverable:** sim-to-real gap report; mixing curve.

### Week 8 — Real hardware: untangling + tool use
- Deploy on the rig for Task A (cable untangling) and Task C (tool use): 100
  episode eval sets, intervention-rate tracking, safety-filter logging.
- First real steering eval: human nudges mid-task; measure jerk under steering vs.
  without.
- **Deliverable:** real-hardware leaderboard v1 for A and C.

### Week 9 — Real hardware: textile folding
- Task B (textile folding on shifting fabric) with world-model replanning active;
  force-limit steering via the filter (contact forces capped in-task).
- **Deliverable:** real-hardware leaderboard v1 for B; full-task suite green.

### Week 10 — Zero-shot ablations on hardware
- Held-out morphologies/materials/tools on the real rig (per pre-registered
  protocol): the actual zero-shot claim, measured.
- **Deliverable:** zero-shot results for all three tasks; component-vs-metric
  attribution table.

## Phase IV — Analysis and release (W11–W13)

### Week 11 — Mechanism analysis
- Error attribution (which failure class does each component remove?); steering
  amplitude/timing sweep vs. safety violations; composability study (unseen
  combinations of morphology × material × tool).
- **Deliverable:** mechanism section data; video selects.

### Week 12 — Figures, video, code cleanup
- Monochrome figure set (architecture, safety envelope, flywheel curves,
  zero-shot leaderboards); demo video with steering overlays; reproduce-everything
  pass (all numbers from committed artifacts).
- **Deliverable:** figure/video/code freeze.

### Week 13 — Papers and release
- Finalize NMI manuscript (`nmi_paper.tex`) and IEEE manuscript
  (`ieee_paper.tex`); compile clean; submission checklists.
- Public release: repo with README, results, and the site (`docs/`).
- **M3 milestone.** **Deliverable:** both manuscripts submitted; repo public;
  site live.

---

## Budget of effort

| Phase | Weeks | Focus |
|---|---|---|
| I. Foundation | 1–3 | baseline, data, subgoal pipeline |
| II. Steerability & safety | 4–6 | SMC, CBF filter, sim ablations, flywheel |
| III. Sim-to-real & hardware | 7–10 | real untangling/folding/tool use, zero-shot |
| IV. Analysis & release | 11–13 | mechanism study, papers, release |

**Risks and hedges.** (i) *Teleop throughput* — hedge: procedural sim data is
generated continuously from W3, so training never blocks on human hours; (ii)
*real steering interface latency* — hedge: steering is validated in sim first and
the filter guarantees bounded deviation regardless of interface; (iii) *Task B
hardware difficulty* — hedge: folding is the last hardware task, so failures there
cannot block the untangling/tool-use results; (iv) *compute* — the flow expert is
2–3B active parameters; all ablations run at fixed budget with early-stopping, so
the sweep is scheduling-bound, not compute-bound.
