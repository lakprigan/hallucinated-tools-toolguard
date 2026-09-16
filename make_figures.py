"""Generate figures for the paper from results/results.json."""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(HERE, "results", "results.json")) as f:
    R = json.load(f)

classes = R["classes"]
short = [c.split("_")[0] for c in classes]
pipes = ["no_defense", "racg_cguard", "resolution_only", "toolguard_full"]
labels = {"no_defense": "No defense",
          "racg_cguard": "Gate + contract verify\n(gating-only stack)",
          "resolution_only": "Resolution Rung\n(ours, alone)",
          "toolguard_full": "Full stack\n(ours)"}

# ---- Figure 1: grouped bar of attack success by class x pipeline ----
fig, ax = plt.subplots(figsize=(7.2, 3.4))
x = np.arange(len(classes))
w = 0.2
colors = ["#b0b0b0", "#d1495b", "#66a182", "#2e4057"]
for i, p in enumerate(pipes):
    vals = [R["attack_success"][p][c] for c in classes]
    ax.bar(x + (i - 1.5) * w, vals, w, label=labels[p], color=colors[i])
ax.set_xticks(x); ax.set_xticklabels(short)
ax.set_ylabel("Attack success rate")
ax.set_ylim(0, 1.08)
ax.set_xlabel("Hallucination class")
ax.legend(fontsize=7, ncol=2, loc="upper center")
ax.axhline(0, color="k", lw=0.5)
fig.tight_layout()
fig.savefig(os.path.join(HERE, "figures", "attack_success.png"), dpi=200)
print("wrote figures/attack_success.png")

# ---- Figure 2: the layered pipeline / where each class dies ----
fig, ax = plt.subplots(figsize=(7.2, 2.6))
ax.axis("off")
stages = ["LLM call", "Rung 0\nResolution", "Causal\ngate",
          "Contract\nverifier", "Execute"]
xs = np.linspace(0.05, 0.95, len(stages))
for i, (xx, s) in enumerate(zip(xs, stages)):
    box = "#2e4057" if s.startswith("Rung 0") else "#4a5568"
    ax.add_patch(plt.Rectangle((xx-0.085, 0.42), 0.17, 0.30,
                 fc=box, ec="k", lw=1))
    ax.text(xx, 0.57, s, ha="center", va="center", color="white", fontsize=7.5)
    if i < len(stages)-1:
        ax.annotate("", xy=(xs[i+1]-0.09, 0.57), xytext=(xx+0.09, 0.57),
                    arrowprops=dict(arrowstyle="->", lw=1.2))
# annotations for where classes are rejected
ax.text(xs[1], 0.30, "H1,H2,H3\n(+ H5 detectable)", ha="center", va="top",
        fontsize=7, color="#2e4057")
ax.text(xs[2], 0.30, "H4\n(off-frontier)", ha="center", va="top",
        fontsize=7, color="#66a182")
ax.text(xs[3], 0.30, "corrupted\ncontracts", ha="center", va="top",
        fontsize=7, color="#888")
ax.text(xs[4], 0.30, "H5 residue\n(tool confusion\n-> Paper 1)", ha="center",
        va="top", fontsize=7, color="#d1495b")
ax.set_xlim(0, 1); ax.set_ylim(0.1, 0.85)
fig.tight_layout()
fig.savefig(os.path.join(HERE, "figures", "pipeline.png"), dpi=200)
print("wrote figures/pipeline.png")
