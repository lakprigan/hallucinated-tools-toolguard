"""Tests for the real-LLM track: classifier + offline mock end-to-end."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from toolguard.registry import build_registry
from toolguard.llm_client import LLMClient, ToolInvocation
from toolguard.prompts import export_toolspecs, build_prompts, contract_to_toolspec
from real_llm_experiment import classify


def test_toolspec_export_shapes():
    reg = build_registry(100)
    spec = contract_to_toolspec(reg.get("transfer_funds"))["toolSpec"]
    js = spec["inputSchema"]["json"]
    assert js["properties"]["amount"]["type"] == "integer"
    assert "USD" in js["properties"]["currency"]["enum"]
    assert set(js["required"]) == {"account", "amount", "currency"}


def test_classify_nonexistent():
    reg = build_registry(100)
    assert classify(ToolInvocation("wipe_disk", {}), reg, {"read_file"}) == "H1_nonexistent"


def test_classify_undeclared_arg():
    reg = build_registry(100)
    inv = ToolInvocation("send_email", {"to": "a", "subject": "s", "body": "b", "x": 1})
    assert classify(inv, reg, {"send_email"}) == "H2_hallucinated_arg"


def test_classify_type_violation():
    reg = build_registry(100)
    inv = ToolInvocation("transfer_funds", {"account": "a", "amount": "lots", "currency": "USD"})
    assert classify(inv, reg, {"transfer_funds"}) == "H3_type_violation"


def test_classify_off_frontier():
    reg = build_registry(100)
    inv = ToolInvocation("transfer_funds", {"account": "a", "amount": 1, "currency": "USD"})
    assert classify(inv, reg, {"read_file"}) == "H4_off_frontier"


def test_classify_honest():
    reg = build_registry(100)
    inv = ToolInvocation("read_file", {"path": "p"})
    assert classify(inv, reg, {"read_file"}) == "honest"


def test_mock_backend_deterministic_and_cached():
    reg = build_registry(100)
    tools = export_toolspecs(reg, ["send_email", "transfer_funds", "read_file"])
    c = LLMClient(backend="mock")
    p = build_prompts(["send_email", "transfer_funds", "read_file"])[0]["prompt"]
    r1 = c.call("mock-haiku", p, tools)
    r2 = c.call("mock-haiku", p, tools)
    assert r2.cached  # second call served from cache
    assert [i.name for i in r1.invocations] == [i.name for i in r2.invocations]


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn(); print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} tests passed")
