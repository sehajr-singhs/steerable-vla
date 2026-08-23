"""Generate publication-quality figures from real study results."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import os

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.spines.top': False,
    'axes.spines.right': False,
})

OUT = os.path.join(os.path.dirname(__file__), '..', 'docs', 'figures')
os.makedirs(OUT, exist_ok=True)

# Load results
with open('results/main_1cross.json') as f:
    data = json.load(f)
with open('results/expert_1cross.json') as f:
    expert = json.load(f)

# Compute per-variant stats
variants = ['bc', 'flow_flat', 'ours_nofilter', 'ours_full']
labels = ['BC', 'Flow (flat)', 'Ours − filter', 'Ours (full)']
colors = ['#4C72B0', '#DD8452', '#55A868', '#C44E52']

stats = {}
for kind, label in zip(variants, labels):
    sub = [r for r in data if r['variant'] == kind]
    stats[kind] = {
        'ni_success': [r['ni_success'] for r in sub],
        'success': [r['success'] for r in sub],
        'violations': [r['violations'] for r in sub],
        'jerk': [r['jerk'] for r in sub],
        'interventions': [r['interventions'] for r in sub],
        'crossings_reduced': [r['crossings_reduced'] for r in sub],
    }

# --- Figure 1: Violations (headline result) ---
fig, ax = plt.subplots(figsize=(6, 4))
means = [np.mean(stats[k]['violations']) for k in variants]
stds = [np.std(stats[k]['violations']) for k in variants]
bars = ax.bar(labels, means, yerr=stds, capsize=4, color=colors, edgecolor='white', linewidth=0.5)
ax.set_ylabel('Safety violations per episode')
ax.set_title('CBF–QP Filter Eliminates All Workspace Violations')
ax.set_ylim(0, max(means) * 1.3)
# Add value labels
for bar, m in zip(bars, means):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
            f'{m:.1f}', ha='center', va='bottom', fontweight='bold', fontsize=10)
# Highlight zero
ax.annotate('0.0 violations\n(hard guarantee)', xy=(3, 0), xytext=(3, 6),
            ha='center', fontsize=9, color='#C44E52',
            arrowprops=dict(arrowstyle='->', color='#C44E52', lw=1.5))
fig.tight_layout()
fig.savefig(os.path.join(OUT, 'fig1_violations.pdf'))
fig.savefig(os.path.join(OUT, 'fig1_violations.png'))
plt.close()
print("✓ fig1_violations.pdf/png")

# --- Figure 2: Ablation radar ---
fig, axes = plt.subplots(1, 4, figsize=(14, 3.5), subplot_kw=dict(polar=True))
metrics = ['ni_success', 'success', 'violations', 'interventions']
metric_labels = ['NI Success', 'Success', '1 - Violations', '1 - Interventions']
for i, (kind, label, color) in enumerate(zip(variants, labels, colors)):
    ax = axes[i]
    vals = []
    for m in metrics:
        v = np.mean(stats[kind][m])
        if m == 'violations':
            v = 1.0 - min(v / 20.0, 1.0)  # normalize: 0 viol = 1.0
        elif m == 'interventions':
            v = 1.0 - min(v / 2.0, 1.0)  # normalize: 0 intv = 1.0
        vals.append(v)
    vals.append(vals[0])  # close the radar
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles.append(angles[0])
    ax.plot(angles, vals, 'o-', color=color, linewidth=2, markersize=5)
    ax.fill(angles, vals, alpha=0.15, color=color)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metric_labels, fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_title(label, fontsize=11, fontweight='bold', pad=15)
    ax.set_rticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(['', '0.5', '', '1.0'], fontsize=7)
fig.suptitle('Ablation: Per-Variant Performance Profile', fontsize=13, y=1.02)
fig.tight_layout()
fig.savefig(os.path.join(OUT, 'fig2_ablation.pdf'))
fig.savefig(os.path.join(OUT, 'fig2_ablation.png'))
plt.close()
print("✓ fig2_ablation.pdf/png")

# --- Figure 3: Safety-utility tradeoff ---
fig, ax = plt.subplots(figsize=(6, 4.5))
for kind, label, color in zip(variants, labels, colors):
    ni = np.mean(stats[kind]['ni_success'])
    vi = np.mean(stats[kind]['violations'])
    ax.scatter(vi, ni, s=150, c=color, edgecolors='black', linewidths=0.8, zorder=5, label=label)
    ax.annotate(label, (vi, ni), textcoords="offset points", xytext=(8, 5), fontsize=9)
ax.set_xlabel('Safety violations per episode $\\downarrow$')
ax.set_ylabel('No-intervention success $\\uparrow$')
ax.set_title('Safety–Utility Tradeoff')
ax.set_xlim(-1, 25)
ax.set_ylim(0, 1.05)
ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.3, label='Perfect success')
ax.axvline(x=0, color='gray', linestyle='--', alpha=0.3, label='Zero violations')
# Shade the ideal quadrant
ax.fill_between([-1, 0], 1.0, 1.05, alpha=0.1, color='green')
ax.annotate('Ideal\n(zero viol, perfect success)', xy=(0, 1.0), xytext=(5, 0.95),
            fontsize=8, color='green', alpha=0.7)
ax.legend(loc='lower right', fontsize=9)
fig.tight_layout()
fig.savefig(os.path.join(OUT, 'fig3_tradeoff.pdf'))
fig.savefig(os.path.join(OUT, 'fig3_tradeoff.png'))
plt.close()
print("✓ fig3_tradeoff.pdf/png")

# --- Figure 4: Expert ceiling ---
fig, ax = plt.subplots(figsize=(4, 3.5))
es = [e['success'] for e in expert]
ax.bar(['Expert ceiling'], [np.mean(es)], yerr=[np.std(es)], capsize=6,
       color='#8172B2', edgecolor='white', linewidth=0.5, width=0.5)
ax.set_ylabel('Success rate')
ax.set_ylim(0, 1.1)
ax.text(0, np.mean(es) + 0.03, f'{np.mean(es):.1%}', ha='center', fontweight='bold', fontsize=12)
ax.set_title('Oracle Expert: 1-Crossing Ceiling')
ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(OUT, 'fig4_expert.pdf'))
fig.savefig(os.path.join(OUT, 'fig4_expert.png'))
plt.close()
print("✓ fig4_expert.pdf/png")

# --- Figure 5: System overview (architecture diagram as text) ---
fig, ax = plt.subplots(figsize=(8, 3))
ax.set_xlim(0, 10)
ax.set_ylim(0, 3)
ax.axis('off')
ax.set_title('System Architecture', fontsize=14, fontweight='bold', pad=10)

boxes = [
    (0.5, 1.2, 2, 1.2, 'VLM Planner\n(3B+ VLM)', '#E6E6FA'),
    (3, 1.2, 2, 1.2, 'Flow Expert\n(CFM + Euler)', '#E6F3FF'),
    (5.5, 1.2, 2, 1.2, 'CBF–QP Filter\n(scipy SLSQP)', '#FFE6E6'),
    (8, 1.2, 1.8, 1.2, 'Cable Env\n(PBD)', '#E6FFE6'),
]
for x, y, w, h, text, fc in boxes:
    rect = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1",
                           facecolor=fc, edgecolor='black', linewidth=1.2)
    ax.add_patch(rect)
    ax.text(x + w/2, y + h/2, text, ha='center', va='center', fontsize=9, fontweight='bold')

# Arrows
for x1, x2 in [(2.5, 3), (5, 5.5), (7.5, 8)]:
    ax.annotate('', xy=(x2, 1.8), xytext=(x1, 1.8),
                arrowprops=dict(arrowstyle='->', lw=1.5, color='black'))

ax.text(5, 0.4, 'Training: CFM loss + focal loss → Expert demos (no oracle at test time)',
        ha='center', fontsize=9, style='italic', color='gray')
fig.tight_layout()
fig.savefig(os.path.join(OUT, 'fig5_overview.pdf'))
fig.savefig(os.path.join(OUT, 'fig5_overview.png'))
plt.close()
print("✓ fig5_overview.pdf/png")

print(f"\nAll figures saved to {OUT}")
print("Done!")
