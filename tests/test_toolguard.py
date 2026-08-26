"""Unit tests for the resolution rung and pipeline behavior."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from toolguard.registry import build_registry, Risk
from toolguard.gate import (ResolutionRung, RACGGate, ToolCall, Verdict,
                            named_pipelines)


def test_resolution_rejects_nonexistent():
    reg = build_registry(100)
    r = ResolutionRung(reg)
    d = r.check(ToolCall("refund_customer", {}))
    assert d.verdict == Verdict.REJECT_UNRESOLVED


def test_resolution_rejects_undeclared_arg():
    reg = build_registry(100)
    r = ResolutionRung(reg)
    d = r.check(ToolCall("send_email", {"to": "a", "subject": "s", "body": "b",
                                        "__x": True}))
    assert d.verdict == Verdict.REJECT_UNDECLARED_ARG


def test_resolution_rejects_missing_required():
    reg = build_registry(100)
    r = ResolutionRung(reg)
    d = r.check(ToolCall("send_email", {"to": "a"}))  # missing subject/body
    assert d.verdict == Verdict.REJECT_MISSING_ARG


def test_resolution_rejects_type_violation():
    reg = build_registry(100)
    r = ResolutionRung(reg)
    d = r.check(ToolCall("transfer_funds",
                         {"account": "acct1", "amount": "lots", "currency": "USD"}))
    assert d.verdict == Verdict.REJECT_TYPE


def test_resolution_allows_valid():
    reg = build_registry(100)
    r = ResolutionRung(reg)
    d = r.check(ToolCall("transfer_funds",
                         {"account": "acct1", "amount": 100, "currency": "USD"}))
    assert d.allowed


def test_racg_rejects_off_frontier():
    reg = build_registry(100)
    g = RACGGate(reg)
    d = g.check(ToolCall("transfer_funds",
                         {"account": "a", "amount": 1, "currency": "USD"}),
                visible={"read_file"}, state={"payment_authorized"})
    assert d.verdict == Verdict.REJECT_OFF_FRONTIER


def test_racg_rejects_unauthorized():
    reg = build_registry(100)
    g = RACGGate(reg)
    d = g.check(ToolCall("transfer_funds",
                         {"account": "a", "amount": 1, "currency": "USD"}),
                visible={"transfer_funds"}, state=set())
    assert d.verdict == Verdict.REJECT_UNAUTHORIZED


def test_full_pipeline_blocks_hallucination_gate_does_not():
    reg = build_registry(100)
    pipes = named_pipelines(reg)
    call = ToolCall("refund_customer", {"amount": 999})
    visible, state = {"read_file"}, set()
    # ContractGuard stack lets the hallucinated tool escape
    assert pipes["racg_cguard"].decide(call, visible, state).allowed
    # full stack rejects it at rung 0
    assert not pipes["toolguard_full"].decide(call, visible, state).allowed


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} tests passed")
