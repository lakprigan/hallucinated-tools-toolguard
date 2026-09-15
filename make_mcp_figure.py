"""Figure for the MCP hallucination benchmark: attack success by MCP class
(M1-M5) for the naive MCP host vs. the MCP Resolution Rung, plus the per-class
rejection reason breakdown for the rung (showing each class is caught by a
distinct closed-world check)."""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(HERE, "results", "mcp_results.json")) as f:
    R = json.load(f)

classes = R["classes"]
short = [c.split("_")[0] for c in classes]
naive = [R["attack_success"]["naive_host"][c] for c in classes]
rung = [R["attack_success"]["mcp_resolution"][c] for c in classes]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.6, 3.1),
                               gridspec_kw={"width_ratios": [1.3, 1]})

x = np.arange(len(classes)); w = 0.38
ax1.bar(x - w / 2, naive, w, label="Naive MCP host", color="#d1495b")
ax1.bar(x + w / 2, rung, w, label="MCP Resolution Rung (ours)", color="#2e4057")
ax1.set_xticks(x); ax1.set_xticklabels(short, fontsize=8)
ax1.set_ylabel("attack success (fraction executed)")
ax1.set_ylim(0, 1.08)
ax1.set_title("MCP hallucination: naive host vs. resolver", fontsize=8.5)
ax1.legend(fontsize=7, loc="center right")
for xi, v in zip(x, naive):
    ax1.text(xi - w / 2, v + 0.02, f"{v:.0%}", ha="center", fontsize=6.5)

# right: which closed-world check catches each class (rejection reasons)
reason_label = {
    "reject_no_provider": "no provider (M1)",
    "reject_ambiguous": "ambiguous (M2)",
    "reject_shadowed": "shadowed (M3)",
    "reject_stale": "stale (M4)",
    "reject_signature": "signature (M5)",
    "allow": "allowed",
}
colors = {"reject_no_provider": "#8fb8de", "reject_ambiguous": "#edae49",
          "reject_shadowed": "#d1495b", "reject_stale": "#66a182",
          "reject_signature": "#2e4057", "allow": "#cccccc"}
vb = R["verdict_breakdown"]
# each class is dominated by exactly one reason; show as a labeled matrix
reasons = ["reject_no_provider", "reject_ambiguous", "reject_shadowed",
           "reject_stale", "reject_signature"]
mat = np.array([[vb[c].get(rn, 0) for rn in reasons] for c in classes], dtype=float)
mat = mat / mat.sum(axis=1, keepdims=True).clip(min=1)
im = ax2.imshow(mat, cmap="Blues", vmin=0, vmax=1, aspect="auto")
ax2.set_xticks(range(len(reasons)))
ax2.set_xticklabels([r.replace("reject_", "") for r in reasons], rotation=40,
                    ha="right", fontsize=6.5)
ax2.set_yticks(range(len(classes))); ax2.set_yticklabels(short, fontsize=7)
ax2.set_title("Which check catches each class", fontsize=8.5)
for i in range(len(classes)):
    for j in range(len(reasons)):
        if mat[i, j] > 0.5:
            ax2.text(j, i, "\u2713", ha="center", va="center", fontsize=9,
                     color="white")

fig.suptitle("MCP multi-server hallucination benchmark "
             f"(n={R['config']['n_trials']}/class, policy={R['config']['policy']})",
             fontsize=9)
fig.tight_layout(rect=[0, 0, 1, 0.94])
out = os.path.join(HERE, "figures", "mcp.png")
fig.savefig(out, dpi=200)
print("wrote", out)
