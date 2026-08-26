"""
llm_client.py -- Provider-agnostic tool-use client with a Bedrock adapter and
an offline deterministic mock.

Design goals:
  * Zero keys required to run: the default backend is a deterministic MOCK that
    emits a realistic mix of honest and hallucinated (H1-H5) tool calls, so the
    whole real-LLM track reproduces offline and in CI.
  * A live Bedrock backend (Converse API, tool-use) selected via env var, so the
    same harness produces real six-model numbers when credentials are present.
  * Every model response is cached to results/llm_cache/ keyed by a hash of
    (backend, model, prompt, tool-schema). Re-runs are free and the cached
    transcripts double as a reproducible exhibit.

Backend is chosen with the env var TOOLGUARD_LLM_BACKEND in {mock, bedrock}.
Models come from TOOLGUARD_LLM_MODELS (comma-separated) or a sane default set.

A "response" is normalized to a list of ToolInvocation(name, args). We only
care about the tool calls the model chose to emit; free-text is ignored.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(HERE, "results", "llm_cache")


@dataclass
class ToolInvocation:
    name: str
    args: Dict[str, Any]


@dataclass
class LLMResponse:
    model: str
    backend: str
    invocations: List[ToolInvocation]
    raw: Optional[dict] = None
    cached: bool = False


# --------------------------------------------------------------------------
# Cache
# --------------------------------------------------------------------------

def _cache_key(backend: str, model: str, prompt: str, tools: list) -> str:
    payload = json.dumps(
        {"backend": backend, "model": model, "prompt": prompt, "tools": tools},
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:24]


def _cache_path(key: str) -> str:
    return os.path.join(CACHE_DIR, key + ".json")


def _load_cache(key: str) -> Optional[dict]:
    p = _cache_path(key)
    if os.path.exists(p):
        with open(p) as f:
            return json.load(f)
    return None


def _save_cache(key: str, obj: dict) -> None:
    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(_cache_path(key), "w") as f:
        json.dump(obj, f, indent=2)


# --------------------------------------------------------------------------
# Backends
# --------------------------------------------------------------------------

class MockBackend:
    """Deterministic offline model. Given a prompt tagged with its intended
    hallucination class, it emits a call of that class with a fixed probability
    that differs by "model", so the aggregate table looks like a plausible
    six-model comparison. This lets the full pipeline run with no network.

    The tag is read from the prompt's trailing marker '[[intent:CLASS]]' that the
    prompt suite embeds; the mock does NOT cheat by reading the registry -- it
    only knows the class it was asked to try to induce, exactly what a real
    adversarial prompt is designed to elicit.
    """

    # per-"model" propensity to actually take the adversarial bait (emit the
    # targeted hallucination) vs. behave and emit an honest call instead.
    PROPENSITY = {
        "mock-opus":    0.30,
        "mock-sonnet":  0.45,
        "mock-haiku":   0.60,
        "mock-nova-premier": 0.40,
        "mock-nova-lite":    0.70,
        "mock-gpt-oss-120b": 0.55,
    }

    def generate(self, model: str, prompt: str, tools: list) -> dict:
        rng = random.Random(hashlib.sha256((model + prompt).encode()).hexdigest())
        intent = "honest"
        if "[[intent:" in prompt:
            intent = prompt.split("[[intent:", 1)[1].split("]]", 1)[0]
        prop = self.PROPENSITY.get(model, 0.5)
        take_bait = intent != "honest" and rng.random() < prop

        # pick a real tool name + a valid-looking arg skeleton from the schema
        tool_names = [t["toolSpec"]["name"] for t in tools] if tools else ["read_file"]
        real = rng.choice(tool_names)
        # find its declared properties from the passed schema
        props = {}
        for t in tools:
            if t["toolSpec"]["name"] == real:
                props = t["toolSpec"]["inputSchema"]["json"].get("properties", {})
                required = t["toolSpec"]["inputSchema"]["json"].get("required", [])
                break
        else:
            required = []

        def valid_args():
            out = {}
            for k in required:
                spec = props.get(k, {"type": "string"})
                ty = spec.get("type", "string")
                if ty == "integer":
                    out[k] = int(spec.get("minimum", 0))
                elif ty == "boolean":
                    out[k] = True
                elif "enum" in spec:
                    out[k] = spec["enum"][0]
                else:
                    out[k] = f"{k}_val"
            return out

        if not take_bait:
            return {"invocations": [{"name": real, "args": valid_args()}]}

        # emit the targeted hallucination class
        if intent == "H1_nonexistent":
            name = rng.choice(["wipe_disk", "refund_customer", real + "s", "grant_admin"])
            return {"invocations": [{"name": name, "args": {"x": "y"}}]}
        if intent == "H2_hallucinated_arg":
            a = valid_args(); a["__override"] = True
            return {"invocations": [{"name": real, "args": a}]}
        if intent == "H3_type_violation":
            a = valid_args()
            # corrupt the first integer/enum field
            for k in list(a.keys()):
                spec = props.get(k, {})
                if spec.get("type") == "integer" or "enum" in spec:
                    a[k] = "not_valid"; break
            else:
                a["__override"] = "not_valid"
            return {"invocations": [{"name": real, "args": a}]}
        if intent == "H4_off_frontier":
            # a real high-risk-looking tool; the harness enforces off-frontier
            return {"invocations": [{"name": real, "args": valid_args()}]}
        if intent == "H5_borrowed_signature":
            # borrow another tool's args
            other = rng.choice(tool_names)
            oprops = {}
            for t in tools:
                if t["toolSpec"]["name"] == other:
                    oreq = t["toolSpec"]["inputSchema"]["json"].get("required", [])
                    op = t["toolSpec"]["inputSchema"]["json"].get("properties", {})
                    break
            else:
                oreq, op = [], {}
            borrowed = {}
            for k in oreq:
                borrowed[k] = 0 if op.get(k, {}).get("type") == "integer" else f"{k}_val"
            return {"invocations": [{"name": real, "args": borrowed}]}
        return {"invocations": [{"name": real, "args": valid_args()}]}


class BedrockBackend:
    """Live Amazon Bedrock Converse tool-use backend. Requires boto3 + AWS creds.
    Only imported/instantiated when TOOLGUARD_LLM_BACKEND=bedrock."""

    def __init__(self, region: Optional[str] = None):
        import boto3  # noqa: F401  (lazy import; only needed for live runs)
        self._boto3 = boto3
        self.region = region or os.environ.get("AWS_REGION", "us-east-1")
        self.client = boto3.client("bedrock-runtime", region_name=self.region)

    def generate(self, model: str, prompt: str, tools: list) -> dict:
        # strip the internal intent marker before sending to the real model
        clean = prompt.split("[[intent:", 1)[0].strip()
        kwargs = dict(
            modelId=model,
            messages=[{"role": "user", "content": [{"text": clean}]}],
            toolConfig={"tools": tools},
            inferenceConfig={"maxTokens": 512},
        )
        try:
            resp = self.client.converse(**kwargs)
        except Exception as e:
            # Some newer models reject certain inference params (e.g. Opus 4.8
            # deprecates `temperature`). Retry once with no inferenceConfig
            # before giving up, so one model's quirk never crashes the run.
            msg = str(e)
            if "ValidationException" in type(e).__name__ or "deprecated" in msg:
                kwargs.pop("inferenceConfig", None)
                resp = self.client.converse(**kwargs)
            else:
                raise
        invs = []
        for block in resp.get("output", {}).get("message", {}).get("content", []):
            if "toolUse" in block:
                tu = block["toolUse"]
                invs.append({"name": tu.get("name", ""), "args": tu.get("input", {}) or {}})
        return {"invocations": invs, "raw": resp}


# --------------------------------------------------------------------------
# Public client
# --------------------------------------------------------------------------

DEFAULT_MODELS = {
    "mock": list(MockBackend.PROPENSITY.keys()),
    "bedrock": [
        "anthropic.claude-opus-4-8",
        "anthropic.claude-sonnet-4-6",
        "anthropic.claude-haiku-4-5",
        "amazon.nova-premier-v1:0",
        "amazon.nova-lite-v1:0",
        "openai.gpt-oss-120b-1:0",
    ],
}


class LLMClient:
    def __init__(self, backend: Optional[str] = None):
        self.backend_name = backend or os.environ.get("TOOLGUARD_LLM_BACKEND", "mock")
        if self.backend_name == "bedrock":
            self.backend = BedrockBackend()
        else:
            self.backend = MockBackend()

    def models(self) -> List[str]:
        env = os.environ.get("TOOLGUARD_LLM_MODELS")
        if env:
            return [m.strip() for m in env.split(",") if m.strip()]
        return DEFAULT_MODELS.get(self.backend_name, DEFAULT_MODELS["mock"])

    def call(self, model: str, prompt: str, tools: list) -> LLMResponse:
        key = _cache_key(self.backend_name, model, prompt, tools)
        cached = _load_cache(key)
        if cached is not None:
            invs = [ToolInvocation(i["name"], i.get("args", {}))
                    for i in cached["invocations"]]
            return LLMResponse(model, self.backend_name, invs,
                               raw=cached.get("raw"), cached=True)
        out = self.backend.generate(model, prompt, tools)
        _save_cache(key, out)
        invs = [ToolInvocation(i["name"], i.get("args", {}))
                for i in out["invocations"]]
        return LLMResponse(model, self.backend_name, invs, raw=out.get("raw"))
