"""Tests for the real-LLM track: classifier + offline mock end-to-end."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from toolguard.registry import build_registry
from toolguard.llm_client import LLMClient, ToolInvocation, parse_rawjson_call
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


def test_parse_rawjson_fenced_and_plain():
    # fenced JSON with commentary around it
    txt = 'Sure!\n```json\n{"name": "wipe_disk", "arguments": {"target": "/"}}\n```\n'
    invs = parse_rawjson_call(txt)
    assert len(invs) == 1 and invs[0].name == "wipe_disk"
    assert invs[0].args == {"target": "/"}
    # plain object, alternate keys
    invs2 = parse_rawjson_call('{"tool": "read_file", "args": {"path": "/x"}}')
    assert invs2[0].name == "read_file" and invs2[0].args == {"path": "/x"}
    # no JSON -> no call
    assert parse_rawjson_call("I refuse to call any tool.") == []


def test_rawjson_mode_lets_h1_survive():
    # Under raw-JSON prompting the decoder is unconstrained, so a fabricated
    # tool name (H1) can survive to the runtime; under schema mode it cannot.
    reg = build_registry(100)
    exposed = ["send_email", "delete_file", "transfer_funds", "create_event", "read_file"]
    tools = export_toolspecs(reg, exposed)
    c = LLMClient(backend="mock")
    # a weak model on an H1 probe, raw-JSON mode, should sometimes fabricate
    probes = [p for p in build_prompts(exposed, per_class=6)
              if p["class"] == "H1_nonexistent"]
    saw_h1 = False
    for p in probes:
        r = c.call("mock-llama-8b", p["prompt"], tools, mode="rawjson")
        if r.invocations and classify(r.invocations[0], reg, set(exposed)) == "H1_nonexistent":
            saw_h1 = True
            break
    assert saw_h1, "expected at least one surviving H1 under raw-JSON mode"
    # same probes under schema mode never yield an H1 (name coerced to real tool)
    for p in probes:
        r = c.call("mock-llama-8b", p["prompt"], tools, mode="schema")
        for inv in r.invocations:
            assert classify(inv, reg, set(exposed)) != "H1_nonexistent"


def test_smoke_check_detects_violations():
    from smoke_test import check
    # H1 survived under schema -> HYPOTHESIS violation
    v = check({"m::schema": {"n_probes": 4, "n_halluc": 2, "classes": {},
                             "leaked_prior": 2, "leaked_full": 0,
                             "h1_under_schema": 1, "errors": 0, "transcript": []}})
    assert any("HYPOTHESIS" in x for x in v)
    # full stack executed a hallucination -> STRUCTURAL violation
    v = check({"m::rawjson": {"n_probes": 4, "n_halluc": 3, "classes": {},
                              "leaked_prior": 3, "leaked_full": 1,
                              "h1_under_schema": 0, "errors": 0, "transcript": []}})
    assert any("full stack EXECUTED" in x for x in v)
    # clean case -> no violations
    assert check({"m::rawjson": {"n_probes": 4, "n_halluc": 3, "classes": {},
                                 "leaked_prior": 3, "leaked_full": 0,
                                 "h1_under_schema": 0, "errors": 0, "transcript": []}}) == []


def test_smoke_runs_against_mock():
    from smoke_test import run_probe_set, check
    c = LLMClient(backend="mock")
    recs = run_probe_set(c, ["mock-haiku", "mock-llama-8b"],
                         ["schema", "rawjson"], per_class=2)
    assert recs and all(r["leaked_full"] == 0 for r in recs.values())
    assert check(recs) == []


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn(); print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} tests passed")
