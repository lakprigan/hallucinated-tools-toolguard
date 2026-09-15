"""Figure for the real-LLM track: per-model hallucination rate under the two
invocation modes (schema-enforced vs. raw-JSON/MCP-bridge), plus the aggregate
leak of the prior stack vs. the full stack.

Two panels:
  (left)  per-model hallucination rate, schema vs. raw-JSON -- shows that
          hallucination survives to the runtime chiefly on the unconstrained
          raw-JSON surface, and more on weaker models.
  (right) aggregate leak: prior stack executes every emitted hallucination;
          the full stack (with the Resolution Rung) executes none.
"""
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
         .replace("openai.", "").replace("meta.", "").replace("mistral.", "")
         for m in models]
modes = R["by_mode"].keys() if "by_mode" in R else []
have_schema = "schema" in R.get("by_mode", {})
have_raw = "rawjson" in R.get("by_mode", {})


def rate(mode, m):
    return R["by_mode"][mode][m]["hallucination_rate"]


def agg(mode):
    pm = R["by_mode"][mode]
    h = sum(pm[m]["n_hallucinations"] for m in models)
    p = sum(pm[m]["leaked_prior_stack"] for m in models)
    f = sum(pm[m]["leaked_full_stack"] for m in models)
    return h, p, f


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.4, 3.1),
                               gridspec_kw={"width_ratios": [2.2, 1]})

x = np.arange(len(models)); w = 0.4
if have_schema:
    ax1.bar(x - w / 2, [rate("schema", m) for m in models], w,
            label="schema-enforced", color="#8fb8de")
if have_raw:
    ax1.bar(x + w / 2, [rate("rawjson", m) for m in models], w,
            label="raw-JSON (MCP bridge)", color="#d1495b")
ax1.set_xticks(x); ax1.set_xticklabels(short, rotation=25, ha="right", fontsize=6.5)
ax1.set_ylabel("hallucination rate"); ax1.set_ylim(0, 1.0)
ax1.set_title("Real-model hallucination rate by invocation mode", fontsize=8.5)
ax1.legend(fontsize=7, loc="upper left")

# right panel: aggregate leak across all modes/models
th = tp = tf = 0
for mode in R["by_mode"]:
    h, p, f = agg(mode)
    th += h; tp += p; tf += f
bars = ax2.bar(["prior\nstack", "full stack\n(ours)"], [tp / th, tf / th],
               color=["#d1495b", "#2e4057"])
ax2.set_ylim(0, 1.08); ax2.set_ylabel("fraction of emitted\nhallucinations executed")
ax2.set_title(f"Aggregate leak (n={th})", fontsize=8.5)
for b, v in zip(bars, [tp, tf]):
    ax2.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.03,
             f"{v}/{th}", ha="center", fontsize=8)

fig.suptitle(f"Real-LLM validation ({R['backend']} backend, "
             f"{len(models)} models)", fontsize=9)
fig.tight_layout(rect=[0, 0, 1, 0.95])
out = os.path.join(HERE, "figures", "real_llm.png")
fig.savefig(out, dpi=200)
print("wrote", out)
