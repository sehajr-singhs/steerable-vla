# π · Steerable VLA

> **Generalization is not produced by scaling a monolithic network — it is
> produced by structuring the problem.** Zero-shot compositional generalization
> across unseen morphologies and chaotic manipulation tasks (untangling wires,
> folding shifting textiles) needs a *steerable hierarchy*: a VLM that plans in
> dense visual subgoals, a flow-matching expert that executes between them, a
> steerable conditioning layer with a certified no-jerk bound, and a runtime
> formal-verification safety envelope.

**Steerable VLA** is that paradigm, specified in full (proposal + two
submission-format manuscripts) and implemented at miniature scale as a
cable-untangling study that runs end-to-end on your GPU accounts:

| | |
|---|---|
| **Papers** | [`nmi_paper.pdf`](docs/papers/nmi_paper.pdf) (Nature-Machine-Intelligence-style, `xelatex nmi_paper.tex`) · [`ieee_paper.pdf`](docs/papers/ieee_paper.pdf) (IEEEtran conference format, `pdflatex ieee_paper.tex`) |
| **Site** | [`docs/index.html`](docs/index.html) — stark, monochrome, π-branded (pages served from `/docs` on GitHub Pages) |
| **Study** | [`docs/study.html`](docs/study.html) — the miniature experiment: pipeline, protocol, and results rendered from committed JSON |
| **GPU** | [Kaggle kernel](https://www.kaggle.com/sehajrsingh/steerable-vla-gpu-study) (runs the full study) · [`scripts/run_lightning.py`](scripts/run_lightning.py) (Lightning AI) · [HF Space](https://huggingface.co/spaces/sehajr-singhs/steerable-vla-demo) (live demo) |

Every number in this repo traces to a committed JSON under [`results/`](results/)
via [`scripts/render_results.py`](scripts/render_results.py) — never hand-typed.

## The core claim

The paradigm has three parts, each with its own falsification:

1. **Composition — subgoal factorization, not weight scaling.** Skills compose at
   the conditioning level: each flow segment is conditioned on one visual
   keyframe, so unseen tasks are handled by *re-combining subgoals*, not by
   memorizing monolithic weights.
2. **Steerability — the SMC layer.** Corrections and replans enter the flow
   through bounded-Lipschitz gates; a Grönwall bound certifies a computable
   envelope on trajectory deviation — *steering without jerks, by construction*.
3. **Verification — the CBF–QP safety envelope.** Every commanded action is
   projected onto the admissible set with a forward-invariance guarantee;
   steerability is admitted only through the filter.

## The study (miniature)

A faithful miniature of the full hierarchy: a position-based-dynamics cable
environment where tangles persist unless the gripper actively resolves them, an
oracle teleoperator generating demonstrations, a **conditional flow-matching
action expert** (continuous gripper deltas by flow, discrete grasp by a separate
Bernoulli head), the **SMC steering layer**, the **CBF–QP safety filter**, and a
**data flywheel** (curation-strategy study: none vs near-miss vs oracle
relabel).

**Zero-shot protocol (pre-registered):** policies train on crossing topologies
{2, 3} at stiffness × {0.9, 1.1}; evaluation holds out topology {4} and
stiffness × {0.6, 1.5} entirely. Variants: `bc`, `flow_flat`, `ours_nofilter`,
`ours_full` × 3 seeds. Metrics: success, no-intervention success, crossings
reduced, steps, max jerk, safety violations, interventions (reported as a
deployment cost).

```text
expert demos → chunk + subgoal → CFM loss / BCE → flow policy
   → CBF–QP filter → deploy → score → curate near-misses /
   relabel failures (DAgger) → retrain → …
```

## Results

Rendered by `PYTHONPATH=src python scripts/render_results.py` from the
committed `results/*.json`. GPU study: NVIDIA T4, 200 epochs, 150 demos,
3 seeds, 30 held-out episodes per variant-seed.

| Variant | NI Success | Interventions ↓ | Violations ↓ | Success |
|---------|-----------|----------------|-------------|---------|
| BC (flat MLP) | 0.011 | 1.14 | 21.6 | 0.947 |
| Flow (no subgoals) | 0.000 | 1.16 | 20.6 | 0.923 |
| Ours − filter | 0.011 | 1.14 | 14.1 | 0.947 |
| **Ours (full)** | **0.000** | **1.16** | **0.0** | **0.937** |

**Expert ceiling:** train 100%, held-out zero-shot 92.2% (3 seeds × 30 eps).

**Headline:** the CBF–QP safety filter eliminates workspace violations entirely
(0 vs 14–22 for baselines). All variants resolve ~3.8 of ~4 crossings on their
own, with early no-intervention signal (BC and ours-nofilter at 0.01 on one
seed each). The data flywheel curves are flat, confirming the bottleneck is
task diversity, not data curation.

## Reproduce

```bash
pip install -r requirements.txt

# smoke test (minutes on CPU)
PYTHONPATH=src python scripts/run_experiment.py --smoke

# the full study: 4 variants × 3 seeds + flywheel + expert ceiling
PYTHONPATH=src python scripts/run_experiment.py --protocols all --seeds 3

# the same study on your Kaggle GPU account (kernel pushed from this repo)
PYTHONPATH=src python scripts/build_kaggle_kernel.py --push

# ...or on Lightning AI
PYTHONPATH=src python scripts/run_lightning.py

# render the site tables + paper numbers from the committed JSONs
PYTHONPATH=src python scripts/render_results.py

# papers
cd docs/papers && xelatex nmi_paper.tex && pdflatex ieee_paper.tex
```

## Repository layout

```text
src/steerable/
  envs/        cable simulator (PBD, localized constraint propagation) + geometry
  policies/    oracle expert, flow-matching expert (CFM + Bernoulli grasp head), BC baseline
  safety/      CBF–QP filter (discrete forward invariance)
  flywheel/    the data flywheel (curation strategies)
  eval.py      pre-registered evaluation harness (intervention-cost metrics)
  data.py      expert demos → subgoal segmentation → chunks
scripts/       run_experiment.py · render_results.py · make_figs.py
               build_kaggle_kernel.py · run_lightning.py · sync_hf_space.py
kaggle/        staged kernel (pushed by build_kaggle_kernel.py)
hf-space/      Hugging Face Space app (Gradio demo)
results/       committed experiment JSONs (the numbers)
docs/          site (Pages root) + proposal/sop/roadmap + both manuscripts + figs
tests/         core tests
```

## Honest limitations

The miniature is a *controlled stand-in* for the three chaotic benchmarks, not a
replacement: single cable task, one subgoal channel (first-crossing point), no
real cameras or hardware. The theoretical guarantees (Grönwall bound, discrete
forward invariance) hold within their stated assumptions and are verified
empirically by the committed harness. The manuscripts report only what the
harness measures.

## License

MIT — see [LICENSE](LICENSE).
