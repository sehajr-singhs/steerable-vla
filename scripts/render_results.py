"""Render results/*.json into paper-ready tables and docs/results.json.

Every number in the manuscripts and the site regenerates from here — nothing
is hand-typed. Usage: PYTHONPATH=src python scripts/render_results.py
"""

import json
import os
import sys

ROOT = os.path.join(os.path.dirname(__file__), "..")
RESULTS = os.path.join(ROOT, "results")


def load(name):
    p = os.path.join(RESULTS, name)
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f)


def mean(rows, key):
    vals = [float(r[key]) for r in rows if key in r]
    return round(sum(vals) / len(vals), 3) if vals else None


def render_main():
    main = load("main.json")
    if not main:
        return None
    order = ["bc", "flow_flat", "ours_nofilter", "ours_full"]
    table = []
    for v in order:
        rows = [r for r in main if r["variant"] == v]
        if not rows:
            continue
        table.append({
            "variant": v,
            "success": mean(rows, "success"),
            "ni_success": mean(rows, "ni_success"),
            "crossings_reduced": mean(rows, "crossings_reduced"),
            "steps": mean(rows, "steps"),
            "jerk": mean(rows, "jerk"),
            "violations": mean(rows, "violations"),
            "interventions": mean(rows, "interventions"),
            "n_seeds": len(rows),
        })
    return table


def render_flywheel():
    out = {}
    for s in ["none", "near_miss", "relabel"]:
        d = load(f"flywheel_{s}.json")
        if d:
            out[s] = {"curve": d["curve"],
                      "final": d["curve"][-1] if d["curve"] else None,
                      "detail": d["detail"]}
    return out or None


def render_expert():
    d = load("expert.json")
    if not d:
        return None
    return {k: round(sum(x["success"] for x in v) / len(v), 3)
            for k, v in d.items()}


def main():
    out = {"main": render_main(), "flywheel": render_flywheel(),
           "expert": render_expert()}
    dst = os.path.join(ROOT, "docs", "results.json")
    with open(dst, "w") as f:
        json.dump(out, f, indent=2)
    print("wrote", dst)
    if out["main"]:
        for row in out["main"]:
            print(f"  {row['variant']:<14} ni_success={row['ni_success']} "
                  f"jerk={row['jerk']} violations={row['violations']} "
                  f"interventions={row['interventions']}")
    if out["expert"]:
        print("  expert ceiling:", out["expert"])
    if out["flywheel"]:
        for s, d in out["flywheel"].items():
            print(f"  flywheel[{s}]: {d['curve']}")


if __name__ == "__main__":
    main()
