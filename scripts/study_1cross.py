"""Quick study on 1-crossing configs — CPU-friendly, runs in ~10 min.

All 4 variants × 3 seeds, 30 demos, 50 epochs, 10 eval episodes.
Proves the pipeline works with real ni_success numbers.
"""
import json, os, sys, time
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from steerable.config import EnvConfig, DataConfig, TrainConfig, EvalConfig
from steerable.data import collect_demos, build_dataset
from steerable.policies.flow_expert import make_policy, train_policy
from steerable.policies.expert import run_expert
from steerable.eval import evaluate
from steerable.envs.cable import CableEnv

N_DEMOS = 25
EPOCHS = 40
N_EVAL = 8
SEEDS = 3
VARIANTS = ["bc", "flow_flat", "ours_nofilter", "ours_full"]
OUT = os.path.join(os.path.dirname(__file__), "..", "results")


def make_env(seed):
    return CableEnv(EnvConfig(), seed=seed, crossing_target=1)


def main():
    t0 = time.time()
    os.makedirs(OUT, exist_ok=True)

    ecfg = EnvConfig()
    dcfg = DataConfig()
    tcfg = TrainConfig(epochs=EPOCHS, hidden=128, latent=64, batch=64)
    evcfg = EvalConfig()
    evcfg.replan = 0
    evcfg.max_steps = 60
    evcfg.patience = 50  # high patience = rarely intervenes = fast eval
    o = CableEnv(ecfg).reset().shape[0]
    dims = (o, o, dcfg.chunk * 3)

    # --- Expert ceiling on 1-crossing ---
    print("=== Expert ceiling (1-crossing) ===", flush=True)
    expert_results = []
    for s in range(SEEDS):
        ok = 0
        for k in range(N_EVAL):
            env = make_env(s * 1000 + k)
            r = run_expert(env, max_steps=100)
            ok += r["success"]
        frac = ok / N_EVAL
        expert_results.append({"seed": s, "success": frac})
        print(f"  seed {s}: {ok}/{N_EVAL} = {frac:.3f}")
    with open(os.path.join(OUT, "expert_1cross.json"), "w") as f:
        json.dump(expert_results, f, indent=2)

    # --- Main study ---
    rows = []
    for kind in VARIANTS:
        for s in range(SEEDS):
            tv = time.time()
            print(f"\n[{kind} seed={s}] collecting demos...", flush=True)
            demos = collect_demos(ecfg, {"make_env": make_env}, N_DEMOS, s * 100 + 42, dcfg)
            ds = build_dataset(demos, dcfg, seed=s)
            print(f"  {len(ds)} chunks", flush=True)

            print(f"  training ({EPOCHS} epochs)...", flush=True)
            policy = make_policy(kind, dims, tcfg, dcfg)
            policy = train_policy(policy, ds, tcfg, dcfg, "cpu", epochs=EPOCHS,
                                  seed=s, kind=kind, verbose=True)

            use_filter = kind == "ours_full"
            steer = kind in ("ours_nofilter", "ours_full")
            print(f"  evaluating ({N_EVAL} episodes)...", flush=True)
            res = evaluate(policy, make_env, N_EVAL, s * 100 + 300, dcfg, evcfg,
                           use_filter=use_filter, steer=steer, device="cpu",
                           allow_intervention=False)

            row = {"variant": kind, "seed": s, "n_demos": N_DEMOS, "epochs": EPOCHS,
                   "success": res["success"], "ni_success": res["ni_success"],
                   "crossings_reduced": res["crossings_reduced"],
                   "steps": res["steps"], "jerk": res["jerk"],
                   "violations": res["violations"], "interventions": res["interventions"],
                   "n": res["n"]}
            rows.append(row)
            print(f"  -> ni={res['ni_success']:.3f} suc={res['success']:.3f} "
                  f"viol={res['violations']:.1f} intv={res['interventions']:.1f} "
                  f"({time.time()-tv:.0f}s)")

    with open(os.path.join(OUT, "main_1cross.json"), "w") as f:
        json.dump(rows, f, indent=2, default=float)

    # Summary
    print("\n=== RESULTS (1-crossing) ===")
    for kind in VARIANTS:
        sub = [r for r in rows if r["variant"] == kind]
        ni = np.mean([r["ni_success"] for r in sub])
        su = np.mean([r["success"] for r in sub])
        vi = np.mean([r["violations"] for r in sub])
        iv = np.mean([r["interventions"] for r in sub])
        cr = np.mean([r["crossings_reduced"] for r in sub])
        print(f"  {kind:20s} ni={ni:.3f} suc={su:.3f} viols={vi:.1f} "
              f"intv={iv:.1f} cr={cr:.2f}")
    print(f"\nTotal: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
