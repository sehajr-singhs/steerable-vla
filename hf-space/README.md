---
title: Steerable VLA — Miniature Study
emoji: π
colorFrom: gray
colorTo: gray
sdk: gradio
sdk_version: 4.44.1
app_file: app.py
pinned: false
license: mit
---

# Steerable VLA — Miniature Study Dashboard

Committed results of the steerable flow-matching VLA study (cable untangling
miniature): four policy variants, the CBF safety envelope, and the data
flywheel — with the full reproducible harness in the repo.

- **Variants** — BC, flow-flat, flow+SMC (no filter), flow+SMC+CBF (ours).
- **Safety** — runtime CBF–QP filter; violations are counted when commands
  would leave the safe set.
- **Flywheel** — DAgger-style relabeling from failure states; the strategy
  decides whether the loop compounds.

Code: https://github.com/sehajr-singhs/steerable-vla
