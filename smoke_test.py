"""
smoke_test.py -- small LIVE Bedrock validation + live-vs-mock diff.

Purpose: before trusting the numbers in the paper, run a SMALL set of real
Bedrock probes (a few models, both invocation surfaces, few probes/class),
classify what each model actually emitted, run those emissions through the
defense pipelines, and:

  1. Print a per-(model, mode) summary of live results.
  2. Diff live vs. the offline mock on the SAME config, so you can see where the
     mock's hand-set propensities diverge from reality.
  3. Assert the paper's structural claims on the live data and FLAG violations:
       * [STRUCTURAL] every emitted hallucination is executed by the gating-only
         stack and rejected by the full stack (Resolution Rung).  A violation
         here would break the paper's core result.
       * [HYPOTHESIS] no H1 (fabricated tool) survives under the schema-enforced
         surface.  A violation is itself a *finding* (a schema API that leaks
         H1) and must be reported, not hidden.

This never fabricates numbers: with no AWS creds it falls back to the mock and
says so loudly. It is a validation gate, not a results generator.

Usage:
    # live (needs boto3 + AWS creds + Bedrock model access):
    TOOLGUARD_LLM_BACKEND=bedrock python3 smoke_test.py
    # dry run against the mock (no creds), to exercise the harness:
    python3 smoke_test.py

Env (all optional):
    TOOLGUARD_SMOKE_MODELS   comma-separated model ids (default: 3 cheap models)
    TOOLGUARD_SMOKE_PERCLASS probes per class per model per mode (default 2)
    TOOLGUARD_SMOKE_MODES    schema | rawjson | both   (default both)
"""

from __future__ import annotations

import json
import os
import random
import sys
from typing import Dict, List, Set

from toolguard.registry import build_registry, Registry
from toolguard.gate import named_pipelines, ToolCall
from toolguard.llm_client import LLMClient
from toolguard.prompts import export_toolspecs, build_prompts
from real_llm_experiment import classify, _state_all_authorized


HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")

# small, cheap default model set for a live smoke run; override via env.
DEFAULT_SMOKE_BEDROCK = [
    "anthropic.claude-haiku-4-5",
    "amazon.nova-lite-v1:0",
    "meta.llama3-8b-instruct-v1:0",
]
DEFAULT_SMOKE_MOCK = ["mock-haiku", "mock-nova-lite", "mock-llama-8b"]


def _modes() -> List[str]:
    m = os.environ.get("TOOLGUARD_SMOKE_MODES", "both").lower()
    return ["schema", "rawjson"] if m == "both" else [m]


def _models(client: LLMClient) -> List[str]:
    env = os.environ.get("TOOLGUARD_SMOKE_MODELS")
    if env:
        return [x.strip() for x in env.split(",") if x.strip()]
    return DEFAULT_SMOKE_BEDROCK if client.backend_name == "bedrock" else DEFAULT_SMOKE_MOCK


def _exposed(reg: Registry, seed: int = 20260617) -> List[str]:
    rng = random.Random(seed)
    anchors = ["send_email", "delete_file", "transfer_funds", "create_event", "read_file"]
    extra = rng.sample([n for n in reg.names() if n not in anchors], 5)
    return anchors + extra


def run_probe_set(client: LLMClient, models: List[str], modes: List[str],
                  per_class: int) -> Dict:
    reg = build_registry(100)
    pipes = named_pipelines(reg)
    exposed = _exposed(reg)
    exposed_set = set(exposed)
    tools = export_toolspecs(reg, exposed)
    probes = build_prompts(exposed, per_class=per_class)
    state = _state_all_authorized(reg)

    out: Dict[str, Dict] = {}
    for mode in modes:
        for model in models:
            key = f"{model}::{mode}"
            rec = {"n_probes": len(probes), "n_halluc": 0,
                   "classes": {}, "leaked_prior": 0, "leaked_full": 0,
                   "h1_under_schema": 0, "errors": 0, "transcript": []}
            for probe in probes:
                try:
                    resp = client.call(model, probe["prompt"], tools, mode=mode)
                except Exception as e:
                    rec["errors"] += 1
                    rec["transcript"].append({"probe": probe["class"],
                                              "error": f"{type(e).__name__}: {str(e)[:180]}"})
                    continue
                if not resp.invocations:
                    rec["transcript"].append({"probe": probe["class"], "emitted": None})
                    continue
                inv = resp.invocations[0]
                cls = classify(inv, reg, exposed_set)
                rec["classes"][cls] = rec["classes"].get(cls, 0) + 1
                rec["transcript"].append({"probe": probe["class"],
                                          "emitted_name": inv.name,
                                          "emitted_args": inv.args,
                                          "classified": cls, "mode": resp.mode})
                if cls == "honest":
                    continue
                rec["n_halluc"] += 1
                if cls == "H1_nonexistent" and mode == "schema":
                    rec["h1_under_schema"] += 1
                visible = set(exposed)
                if cls == "H4_off_frontier":
                    visible.discard(inv.name)
                call = ToolCall(inv.name, inv.args)
                if pipes["racg_cguard"].decide(call, visible, state).allowed:
                    rec["leaked_prior"] += 1
                if pipes["toolguard_full"].decide(call, visible, state).allowed:
                    rec["leaked_full"] += 1
            out[key] = rec
    return out


def check(records: Dict) -> List[str]:
    """Return a list of violation strings; empty means all structural claims held."""
    violations = []
    for key, rec in records.items():
        model, mode = key.split("::")
        h = rec["n_halluc"]
        # STRUCTURAL: prior executes all, full executes none
        if h and rec["leaked_prior"] != h:
            violations.append(
                f"[STRUCTURAL] {key}: gating-only stack rejected some hallucinations "
                f"({rec['leaked_prior']}/{h} executed); paper claims it executes ALL.")
        if rec["leaked_full"] != 0:
            violations.append(
                f"[STRUCTURAL] {key}: full stack EXECUTED {rec['leaked_full']} "
                f"hallucination(s); Resolution Rung must reject all. INVESTIGATE.")
        # HYPOTHESIS: no H1 under schema surface
        if mode == "schema" and rec["h1_under_schema"] > 0:
            violations.append(
                f"[HYPOTHESIS] {key}: {rec['h1_under_schema']} fabricated-tool (H1) "
                f"call(s) survived under a schema-enforced API. This contradicts the "
                f"paper's suppression claim -- report it as a finding.")
    return violations


def diff_against_mock(live: Dict, per_class: int, modes: List[str]) -> Dict:
    """Run the mock on the equivalent model set + config and diff halluc rates."""
    mock_models = DEFAULT_SMOKE_MOCK
    mc = LLMClient(backend="mock")
    mock = run_probe_set(mc, mock_models, modes, per_class)
    # pair live models to mock models positionally for a coarse rate comparison
    live_keys = list(live.keys())
    rows = []
    for i, lk in enumerate(live_keys):
        lmodel, mode = lk.split("::")
        # find same-index mock model for this mode
        same_mode = [k for k in mock if k.endswith("::" + mode)]
        if not same_mode:
            continue
        mk = same_mode[min(i // max(1, len(modes)), len(same_mode) - 1)]
        lr = live[lk]["n_halluc"] / max(1, live[lk]["n_probes"])
        mr = mock[mk]["n_halluc"] / max(1, mock[mk]["n_probes"])
        rows.append({"live": lk, "mock": mk, "mode": mode,
                     "live_halluc_rate": round(lr, 3),
                     "mock_halluc_rate": round(mr, 3),
                     "abs_gap": round(abs(lr - mr), 3)})
    return {"rows": rows, "mock_raw": mock}


def main():
    per_class = int(os.environ.get("TOOLGUARD_SMOKE_PERCLASS", "2"))
    modes = _modes()
    client = LLMClient()
    models = _models(client)

    print("=" * 70)
    print(f"SMOKE TEST  (backend = {client.backend_name})")
    print(f"models = {models}")
    print(f"modes  = {modes}   per_class = {per_class}")
    print("=" * 70)
    if client.backend_name != "bedrock":
        print("!! backend is MOCK: this only exercises the harness; numbers are")
        print("!! illustrative, NOT measured. Set TOOLGUARD_LLM_BACKEND=bedrock")
        print("!! (with boto3 + AWS creds + model access) for a real smoke test.\n")

    live = run_probe_set(client, models, modes, per_class)

    print(f"{'model::mode':<34} {'halluc':>6} {'prior':>6} {'full':>5} {'err':>4}")
    print("-" * 60)
    for key, rec in live.items():
        print(f"{key:<34} {rec['n_halluc']:>6} "
              f"{rec['leaked_prior']:>6} {rec['leaked_full']:>5} {rec['errors']:>4}")
    tot_h = sum(r["n_halluc"] for r in live.values())
    tot_p = sum(r["leaked_prior"] for r in live.values())
    tot_f = sum(r["leaked_full"] for r in live.values())
    print("-" * 60)
    print(f"TOTAL emitted hallucinations: {tot_h}")
    print(f"  gating-only executed : {tot_p}/{tot_h}")
    print(f"  full stack executed  : {tot_f}/{tot_h}")

    # live-vs-mock diff (only meaningful when live is actually bedrock)
    if client.backend_name == "bedrock":
        print("\n--- LIVE vs MOCK hallucination-rate diff ---")
        d = diff_against_mock(live, per_class, modes)
        print(f"{'mode':<8} {'live':>6} {'mock':>6} {'gap':>6}   live-model")
        for row in d["rows"]:
            flag = "  <-- large gap" if row["abs_gap"] >= 0.25 else ""
            print(f"{row['mode']:<8} {row['live_halluc_rate']:>6.2f} "
                  f"{row['mock_halluc_rate']:>6.2f} {row['abs_gap']:>6.2f}   "
                  f"{row['live'].split('::')[0]}{flag}")
        print("(large gaps mean the mock propensities need recalibration to the "
              "live rates before the mock table is used as an exhibit.)")

    violations = check(live)
    print("\n" + "=" * 70)
    if violations:
        print(f"FAIL: {len(violations)} claim violation(s) on the smoke data:")
        for v in violations:
            print("  - " + v)
    else:
        print("PASS: all structural claims held on the smoke data.")
        if tot_h == 0:
            print("  (note: 0 hallucinations emitted -- widen models/probes or use "
                  "raw-JSON mode to actually stress the defense.)")
    print("=" * 70)

    os.makedirs(RESULTS, exist_ok=True)
    slim = {k: {kk: vv for kk, vv in r.items() if kk != "transcript"}
            for k, r in live.items()}
    with open(os.path.join(RESULTS, "smoke_results.json"), "w") as f:
        json.dump({"backend": client.backend_name, "models": models,
                   "modes": modes, "per_class": per_class,
                   "summary": slim,
                   "totals": {"halluc": tot_h, "leaked_prior": tot_p,
                              "leaked_full": tot_f},
                   "violations": violations}, f, indent=2)
    transcripts = {k: r["transcript"] for k, r in live.items()}
    with open(os.path.join(RESULTS, "smoke_transcripts.json"), "w") as f:
        json.dump(transcripts, f, indent=2)

    sys.exit(1 if violations else 0)


if __name__ == "__main__":
    main()
