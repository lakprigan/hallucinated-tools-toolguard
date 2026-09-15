"""Unit tests for the benchmark harness, baselines, and external catalogs."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import toolguard
from toolguard import bench, baselines
from toolguard.catalogs import (build_apibank_registry, build_mcp_manifest_deployment,
                                registry_from_manifest)
from toolguard.registry import ArgKind


def test_package_version_and_api():
    assert toolguard.__version__
    assert toolguard.BENCHMARK_VERSION == "htb-1.0"
    assert callable(toolguard.run_leaderboard)


def test_leaderboard_runs_and_ranks_ours_top():
    board = toolguard.run_leaderboard()
    assert board["benchmark_version"] == "htb-1.0"
    # our resolver should top both tracks
    assert board["single_registry"][0]["name"].startswith("resolution")
    assert board["mcp"][0]["name"].startswith("mcp-resolution")
    # no-defense should be last on both
    assert board["single_registry"][-1]["name"] == "no-defense"
    assert board["mcp"][-1]["name"] == "naive-host"


def test_schema_validate_baseline_leaks_h1():
    # the key point: JSON-schema validation WITHOUT closed-world membership
    # cannot catch a fabricated tool name.
    board = toolguard.run_leaderboard()
    sv = next(e for e in board["single_registry"] if e["name"] == "schema-validate")
    assert sv["attack_success"]["H1_nonexistent"] == 1.0
    ours = next(e for e in board["single_registry"] if e["name"].startswith("resolution-rung"))
    assert ours["attack_success"]["H1_nonexistent"] == 0.0


def test_htb_score_monotone():
    board = toolguard.run_leaderboard()
    scores = [e["htb_score"] for e in board["single_registry"]]
    assert scores == sorted(scores, reverse=True)


def test_apibank_registry_shape():
    reg = build_apibank_registry()
    assert len(reg) >= 12
    assert reg.get("Transfer") is not None
    # Transfer has a typed int amount + enum currency
    sig = reg.get("Transfer").signature
    assert sig["amount"].kind == ArgKind.INT
    assert sig["currency"].kind == ArgKind.ENUM


def test_mcp_manifest_has_natural_collisions():
    dep = build_mcp_manifest_deployment()
    # read_file collides across filesystem+slack; write_file across fs+community
    assert len(dep.providers_of("read_file")) == 2
    assert len(dep.providers_of("write_file")) == 2
    # write_file spans trust tiers (first-party + community) -> a shadow hazard
    tiers = {s.trust for s in dep.providers_of("write_file")}
    assert len(tiers) > 1


def test_manifest_jsonschema_adapter():
    reg = registry_from_manifest([
        {"name": "foo", "inputSchema": {"type": "object",
            "properties": {"n": {"type": "integer", "minimum": 1, "maximum": 5},
                           "s": {"type": "string"}}, "required": ["n"]}}])
    c = reg.get("foo")
    assert c.signature["n"].kind == ArgKind.INT and c.signature["n"].required
    assert not c.signature["s"].required


def test_accepts_coerced_disaggregates_h3():
    from toolguard.registry import ArgType
    t = ArgType(ArgKind.INT, lo=1, hi=100)
    # genuine type hallucination: not coercible
    assert not t.accepts("all of it") and not t.accepts_coerced("all of it")
    # serialization artifact: strict reject, coercible accept
    assert not t.accepts("42") and t.accepts_coerced("42")
    # out-of-range stringified: coercion still respects range
    assert not t.accepts_coerced("999")


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn(); print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} tests passed")
