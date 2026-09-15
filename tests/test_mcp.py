"""Unit tests for the MCP multi-server resolution rung (M1-M5)."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from toolguard.mcp import (MCPCall, MCPResolutionRung, MCPVerdict, MergePolicy,
                           NaiveMCPHost, Trust, build_deployment)
import mcp_experiment as X


def test_m1_fabrication_rejected():
    dep = build_deployment()
    rung = MCPResolutionRung(dep)
    d = rung.check(MCPCall("wipe_disk", {"x": "y"}))
    assert d.verdict == MCPVerdict.REJECT_NO_PROVIDER


def test_m2_same_tier_collision_rejected_ambiguous():
    dep = build_deployment()
    rung = MCPResolutionRung(dep)
    # read_file is on mail + payments (both VERIFIED) -> pure ambiguity
    d = rung.check(MCPCall("read_file", {"path": "p"}))
    assert d.verdict == MCPVerdict.REJECT_AMBIGUOUS


def test_m3_cross_tier_shadow_rejected():
    dep = build_deployment()
    rung = MCPResolutionRung(dep)
    # delete_file is on files (FIRST_PARTY) + community (COMMUNITY) -> shadow
    d = rung.check(MCPCall("delete_file", {"path": "p"}))
    assert d.verdict == MCPVerdict.REJECT_SHADOWED


def test_m3_server_pinning_allows_trusted():
    dep = build_deployment()
    rung = MCPResolutionRung(dep)
    # pinning the first-party server resolves the shadow safely
    d = rung.check(MCPCall("delete_file", {"path": "p"}, server="files"))
    assert d.allowed and d.resolved.server == "files"


def test_m3_pinning_low_trust_still_resolves_but_is_explicit():
    dep = build_deployment()
    rung = MCPResolutionRung(dep)
    # explicitly pinning the community server is allowed (the caller chose it);
    # the point is that a FLAT call must not silently land here.
    d = rung.check(MCPCall("delete_file", {"path": "p"}, server="community-tools"))
    assert d.allowed and d.resolved.server == "community-tools"


def test_m4_stale_definition_rejected():
    import random
    base = build_deployment()
    stale = X.make_stale_deployment(base, random.Random(0))
    rung = MCPResolutionRung(stale)
    d = rung.check(MCPCall("transfer_funds",
                           {"account": "a", "amount": 1, "currency": "USD"},
                           server="payments"))
    assert d.verdict == MCPVerdict.REJECT_STALE


def test_m5_cross_server_borrow_rejected():
    dep = build_deployment()
    rung = MCPResolutionRung(dep)
    # list_dir takes path; give it send_email's args
    d = rung.check(MCPCall("list_dir", {"to": "x", "subject": "s", "body": "b"},
                           server="files"))
    assert d.verdict == MCPVerdict.REJECT_SIGNATURE


def test_wrong_server_pin_rejected():
    dep = build_deployment()
    rung = MCPResolutionRung(dep)
    # transfer_funds is only on payments; pin it to mail
    d = rung.check(MCPCall("transfer_funds",
                           {"account": "a", "amount": 1, "currency": "USD"},
                           server="mail"))
    assert d.verdict == MCPVerdict.REJECT_WRONG_SERVER


def test_honest_unique_provider_allowed():
    dep = build_deployment()
    rung = MCPResolutionRung(dep)
    # list_dir is unique to files
    d = rung.check(MCPCall("list_dir", {"path": "p"}))
    assert d.allowed and d.resolved.server == "files"


def test_naive_host_executes_everything():
    dep = build_deployment()
    naive = NaiveMCPHost(dep)
    for call in [MCPCall("wipe_disk", {"x": "y"}),
                 MCPCall("read_file", {"path": "p"}),
                 MCPCall("delete_file", {"path": "p"})]:
        assert naive.check(call).allowed


def test_benchmark_full_defense():
    r = X.run()
    rung = r["attack_success"]["mcp_resolution"]
    naive = r["attack_success"]["naive_host"]
    # naive host leaks every class; rung stops every class
    assert all(v == 1.0 for v in naive.values())
    assert all(v == 0.0 for v in rung.values())
    assert r["honest_rejection"]["mcp_resolution"] == 0.0


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} tests passed")
