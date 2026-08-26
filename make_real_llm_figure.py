"""Figure for the real-LLM track: per-model hallucination rate + leak rates."""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(HERE, "results", "real_llm_results.json")) as f:
    R = json.load(f)

models = R["models"]
short = [m.replace("mock-", "").replace("anthropic.", "").replace("amazon.", "")
         .replace("openai.", "") for m in models]
halluc = [R["per_model"][m]["hallucination_rate"] for m in models]
prior = [R["per_model"][m]["prior_leak_rate"] for m in models]
full = [R["per_model"][m]["full_leak_rate"] for m in models]

fig, ax = plt.subplots(figsize=(7.2, 3.2))
x = np.arange(len(models)); w = 0.26
ax.bar(x - w, halluc, w, label="hallucination rate", color="#b0b0b0")
ax.bar(x, prior, w, label="prior stack leak", color="#d1495b")
ax.bar(x + w, full, w, label="full-stack leak (ours)", color="#2e4057")
ax.set_xticks(x); ax.set_xticklabels(short, rotation=20, ha="right", fontsize=7)
ax.set_ylabel("rate"); ax.set_ylim(0, 1.08)
ax.set_title(f"Real-LLM validation ({R['backend']} backend)", fontsize=9)
ax.legend(fontsize=7, ncol=3, loc="upper center")
fig.tight_layout()
out = os.path.join(HERE, "figures", "real_llm.png")
fig.savefig(out, dpi=200)
print("wrote", out)
