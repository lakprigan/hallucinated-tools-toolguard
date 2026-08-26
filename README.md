# You Can't Gate What Isn't There

Hallucinated-tool detection for contract-gated LLM agents. Extends
RACG (arXiv:2606.13884) and ContractGuard (arXiv:2606.18550) with a
pre-gate **Resolution Rung** that rejects fabricated tools and
hallucinated arguments under a closed-world assumption.

## Contribution in one line
RACG/ContractGuard secure the tools an agent *has*; this secures the
calls an agent *emits*. A hallucinated call is never a gating decision,
so it must be rejected *before* the gate.

## Layout
- `toolguard/registry.py` -- contract formalism + 100-tool registry (reused from CMTF/RACG lineage).
- `toolguard/gate.py`     -- Resolution Rung, RACG gate, ContractGuard rung, composable pipelines.
- `toolguard/attacks.py`  -- honest calls + five hallucination classes H1-H5 (synthetic track).
- `toolguard/llm_client.py` -- provider-agnostic tool-use client: offline mock + live Bedrock (Converse), on-disk response cache.
- `toolguard/prompts.py`  -- Contract -> Bedrock toolSpec export + adversarial prompt suite (H1-H5).
- `experiment.py`         -- synthetic benchmark -> `results/results.json`, prints table + 6 hypotheses.
- `real_llm_experiment.py` -- real-LLM validation track -> `results/real_llm_results.json` (+ transcripts), per-model table.
- `make_figures.py` / `make_real_llm_figure.py` -- figures.
- `tests/` -- unit tests (`test_toolguard.py`, `test_real_llm.py`).
- `main.tex` / `main.pdf` -- the paper (IEEEtran, 5 pages).

## Reproduce
```bash
python3 tests/test_toolguard.py       # 8 unit tests (synthetic)
python3 tests/test_real_llm.py        # 7 unit tests (real-LLM track)
python3 experiment.py                 # synthetic benchmark (6/6 hypotheses PASS)
python3 real_llm_experiment.py        # real-LLM track (offline mock by default)
python3 make_figures.py               # synthetic figures
python3 make_real_llm_figure.py       # real-LLM figure
tectonic main.tex                     # build main.pdf
```

## Real-LLM validation track
Runs the same registry, tool schema, and defense pipelines, but elicits tool
calls from real models under adversarial prompts, classifies what each model
actually emitted (honest / H1-H5), and runs those emissions through the prior
stack vs. the full stack. Mirrors the ContractGuard six-model methodology.

Default backend is an **offline deterministic mock** (no keys, reproducible).
To run against live Amazon Bedrock models:
```bash
pip install boto3            # + configure AWS credentials / AWS_REGION
export TOOLGUARD_LLM_BACKEND=bedrock
# optional: override the model set
export TOOLGUARD_LLM_MODELS="anthropic.claude-opus-4-8,amazon.nova-lite-v1:0"
export TOOLGUARD_PER_CLASS=4         # probes per class per model
python3 real_llm_experiment.py
```
Every model response is cached under `results/llm_cache/`, so re-runs are free
and the transcripts (`results/real_llm_transcripts.json`) are a reproducible
exhibit.

Offline-mock six-model result: models hallucinate at 17-58%; the prior
RACG+ContractGuard stack executes **24/24** emitted hallucinations; the full
stack with the Resolution Rung executes **0/24** -- the structural result holds
across every model regardless of phrasing.


## Headline result (attack success rate; 0.00 = fully defended)
| Pipeline | H1 | H2 | H3 | H4 | H5 | honest-rej |
|---|---|---|---|---|---|---|
| RACG+ContractGuard (prior) | 1.00 | 1.00 | 1.00 | 0.00 | 1.00 | 0.00 |
| Full stack (ours) | 0.00 | 0.00 | 0.00 | 0.00 | 0.15 | 0.00 |

The 0.15 H5 residue is provably schema-valid (66/66) -- a tool-confusion
problem for causal minimal tool filtering (CMTF, arXiv:2606.06284), not a
resolution failure.
