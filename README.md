# You Can't Gate What Isn't There

Closed-world resolution against **tool hallucination** in LLM agents.
LLMs call tools that do not exist and pass arguments no schema declares;
selection and gating methods both presuppose the emitted call names a
real tool. This adds a training-free, closed-world **Resolution Rung**
that rejects fabricated tools and hallucinated arguments using only the
tool registry the agent already has. It runs in front of *any*
function-calling agent and composes with (but does not depend on)
downstream gating layers such as RACG (arXiv:2606.13884) and
ContractGuard (arXiv:2606.18550).

## Contribution in one line
A hallucinated call is never a gating decision, so it must be rejected
*before* any gate — a structural blind spot every selection/gating
method shares. The Resolution Rung closes it standalone.

## Layout
- `toolguard/registry.py` -- contract formalism + 100-tool registry (reused from CMTF/RACG lineage).
- `toolguard/gate.py`     -- Resolution Rung, RACG gate, ContractGuard rung, composable pipelines.
- `toolguard/mcp.py`      -- MCP multi-server deployment, qualified (server,tool) resolver, M1-M5 classes, naive-host baseline.
- `toolguard/attacks.py`  -- honest calls + five hallucination classes H1-H5 (synthetic track).
- `mcp_experiment.py`     -- MCP synthetic benchmark (naive host 1.00 vs. resolver 0.00 on M1-M5) -> `results/mcp_results.json`.
- `mcp_live_probe.py`     -- live real-model MCP-surface probe (cross-server confusion) -> `results/mcp_live_results.json`.
- `make_mcp_figure.py`    -- MCP figure.
- `tests/test_mcp.py`     -- 11 MCP unit tests.
- `toolguard/llm_client.py` -- provider-agnostic tool-use client: offline mock + live Bedrock (Converse), on-disk response cache.
- `toolguard/prompts.py`  -- Contract -> Bedrock toolSpec export + adversarial prompt suite (H1-H5).
- `experiment.py`         -- synthetic benchmark -> `results/results.json`, prints table + 6 hypotheses.
- `real_llm_experiment.py` -- real-LLM validation track -> `results/real_llm_results.json` (+ transcripts), per-model table.
- `make_figures.py` / `make_real_llm_figure.py` -- figures.
- `tests/` -- unit tests (`test_toolguard.py`, `test_real_llm.py`).
- `smoke_test.py` -- live-vs-mock smoke validation; asserts structural claims, flags H1-under-schema.
- `main.tex` / `main.pdf` -- the paper (IEEEtran, 5 pages).

## Reproduce
```bash
python3 tests/test_toolguard.py       # 8 unit tests (synthetic)
python3 tests/test_real_llm.py        # 11 unit tests (real-LLM + smoke harness)
python3 experiment.py                 # synthetic benchmark (6/6 hypotheses PASS)
python3 real_llm_experiment.py        # real-LLM track (offline mock by default)
python3 smoke_test.py                 # live-vs-mock smoke validation (mock by default)
python3 make_figures.py               # synthetic figures
python3 make_real_llm_figure.py       # real-LLM figure
tectonic main.tex                     # build main.pdf
```

## Live smoke test (validate before trusting numbers)
`smoke_test.py` runs a SMALL set of probes (few models, both invocation
surfaces, few probes/class), classifies what each model actually emits, runs
those emissions through the pipelines, and **asserts the paper's claims**,
flagging any violation:
- **[STRUCTURAL]** gating-only executes every hallucination; the full stack
  (Resolution Rung) executes none. A full-stack leak fails the run (exit 1).
- **[HYPOTHESIS]** no H1 (fabricated tool) survives under a schema-enforced API.
  A survivor is reported as a *finding*, not hidden.

With no creds it falls back to the mock and says so loudly (numbers are then
illustrative, not measured). For a real check:
```bash
pip install boto3            # + AWS creds / AWS_REGION + Bedrock model access
export TOOLGUARD_LLM_BACKEND=bedrock
export TOOLGUARD_SMOKE_MODES=both            # schema | rawjson | both
export TOOLGUARD_SMOKE_PERCLASS=2            # probes/class/model/mode
# optional: export TOOLGUARD_SMOKE_MODELS="anthropic.claude-haiku-4-5,meta.llama3-8b-instruct-v1:0"
python3 smoke_test.py
```
On a live run it also prints a **live-vs-mock hallucination-rate diff**; gaps
>= 0.25 mean the mock propensities must be recalibrated to the measured rates
before the mock table is used as an exhibit. Results are written to
`results/smoke_results.json` (+ `smoke_transcripts.json`).

## Real-LLM validation track
Runs the same registry, tool schema, and defense pipelines, but elicits tool
calls from real models under adversarial prompts, classifies what each model
actually emitted (honest / H1-H5), and runs those emissions through the prior
stack vs. the full stack. Evaluated under **two invocation surfaces**:

- **schema** — Bedrock Converse `toolConfig` (schema-enforced decoding).
  Structurally suppresses fabricated names (H1) and much of H2/H3.
- **rawjson** — tool catalog described in the prompt, model asked for a JSON
  call that a bridge parses. No decoder constraint. This is the MCP-bridge /
  custom-parser / open-weight-agent surface where hallucination survives.

Choose with `TOOLGUARD_LLM_MODE` in `{schema, rawjson, both}` (default `both`).

Default backend is an **offline deterministic mock** (no keys, reproducible).
To run against live Amazon Bedrock models:
```bash
pip install boto3            # + configure AWS credentials / AWS_REGION
export TOOLGUARD_LLM_BACKEND=bedrock
export TOOLGUARD_LLM_MODE=both       # schema | rawjson | both
export TOOLGUARD_PER_CLASS=10        # probes per class per model
python3 real_llm_experiment.py
```
Every model response is cached under `results/llm_cache/`, so re-runs are free
and the transcripts (`results/real_llm_transcripts.json`) are a reproducible
exhibit.

Live Bedrock result (10 models × 2 surfaces, per_class=10): fabricated tools
(H1) concentrate on the raw-JSON surface (34 vs. 3 under schema, the three
slipping schema only on the weakest open-weight models); open-weight models
hallucinate up to 62% and model scale does not help (a 675B model matches a
7-8B one on raw-JSON). Across **313** emitted hallucinations the gating-only
stack executes **313/313** and the full stack with the Resolution Rung executes
**0/313** — the structural result holds at scale, on the calls real models
actually emit. (An offline deterministic mock reproduces the same structure with
no keys, for CI; it is a calibrated stress-test, not the headline.)


## Headline result (attack success rate; 0.00 = fully defended)
| Pipeline | H1 | H2 | H3 | H4 | H5 | honest-rej |
|---|---|---|---|---|---|---|
| Gating-only (RACG+ContractGuard) | 1.00 | 1.00 | 1.00 | 0.00 | 1.00 | 0.00 |
| Full stack (ours) | 0.00 | 0.00 | 0.00 | 0.00 | 0.15 | 0.00 |

**Gating-only pipelines execute 100% of fabricated tools and invalid
arguments; the Resolution Rung drives every schema-detectable class to
0.00 with zero honest over-rejection.**

The 0.15 H5 residue is provably schema-valid (66/66) — a tool-confusion
problem for tool-selection methods such as causal minimal tool filtering
(CMTF, arXiv:2606.06284), not a resolution failure.
