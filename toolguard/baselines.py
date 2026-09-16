"""
baselines.py -- Reference resolvers scored on the Hallucinated-Tools Benchmark.

These are the entries a submission competes against. They include our closed-world
Resolution Rung (single-registry and MCP) and several EXTERNAL / naive baselines
so the leaderboard is a genuine comparison, not a self-scored victory lap:

  single-registry:
    * no-defense            -- fail-open executor (executes everything)
    * name-allowlist        -- accept iff the tool name is registered (ignores args)
    * fuzzy-name            -- accept iff the name is within edit-distance 1 of a
                               real name (models lenient "did you mean" routers)
    * schema-validate       -- JSON-schema type check only, but NO closed-world
                               membership (accepts unknown names if args look ok)
    * resolution-rung       -- ours (membership + full signature check)
    * resolution+gate       -- ours + causal gate (also closes H4)

  MCP:
    * naive-host            -- flatten + first-provider + fail-open
    * flat-schema           -- resolve first provider, then schema-check (no
                               shadow/ambiguity/staleness handling)
    * highest-trust         -- resolve collisions to the highest-trust provider
                               (a plausible real-host policy) then schema-check
    * mcp-resolution-rung   -- ours (qualified resolve + shadow/stale/signature)

A resolver is a plain callable, so third parties add their own with one function.
"""

from __future__ import annotations

from typing import Set

from .registry import Registry, build_registry
from .gate import ToolCall, ResolutionRung, CausalGate, named_pipelines
from .mcp import (MCPCall, MCPDeployment, MCPResolutionRung, NaiveMCPHost,
                  build_deployment)

# shared registry/deployment for the stateful baselines
_REG = build_registry(100)
_DEP = build_deployment()
_RUNG = ResolutionRung(_REG)
_GATE = CausalGate(_REG)
_PIPES = named_pipelines(_REG)
_MCP_RUNG = MCPResolutionRung(_DEP)


# ---- single-registry resolvers (call, visible, state) -> allowed ---------

def r_no_defense(call: ToolCall, visible: Set[str], state: Set[str]) -> bool:
    return True


def r_name_allowlist(call: ToolCall, visible: Set[str], state: Set[str]) -> bool:
    # accept iff the name is registered; ignores arguments entirely
    return _REG.get(call.name) is not None


def _edit_distance_le1(a: str, b: str) -> bool:
    if a == b:
        return True
    la, lb = len(a), len(b)
    if abs(la - lb) > 1:
        return False
    # one substitution
    if la == lb:
        return sum(x != y for x, y in zip(a, b)) == 1
    # one insertion/deletion
    if la > lb:
        a, b = b, a; la, lb = lb, la
    i = j = 0; skipped = False
    while i < la and j < lb:
        if a[i] == b[j]:
            i += 1; j += 1
        elif skipped:
            return False
        else:
            skipped = True; j += 1
    return True


def r_fuzzy_name(call: ToolCall, visible: Set[str], state: Set[str]) -> bool:
    # lenient router: accept if the name is within edit-distance 1 of a real one
    if _REG.get(call.name) is not None:
        return True
    return any(_edit_distance_le1(call.name, n) for n in _REG.names())


def r_schema_validate(call: ToolCall, visible: Set[str], state: Set[str]) -> bool:
    # JSON-schema style check WITHOUT closed-world membership: if the name is
    # unknown, there is no schema, so a permissive validator lets it through.
    c = _REG.get(call.name)
    if c is None:
        return True  # no schema to validate against -> pass (the H1 blind spot)
    for k in call.args:
        if k not in c.signature:
            return False
    for k, t in c.signature.items():
        if t.required and k not in call.args:
            return False
    for k, v in call.args.items():
        if not c.signature[k].accepts(v):
            return False
    return True


def r_resolution_rung(call: ToolCall, visible: Set[str], state: Set[str]) -> bool:
    return _RUNG.check(call).allowed


def r_resolution_gate(call: ToolCall, visible: Set[str], state: Set[str]) -> bool:
    d = _RUNG.check(call)
    if not d.allowed:
        return False
    if _REG.get(call.name) is None:
        return False
    return _GATE.check(call, visible, state).allowed


DEFAULT_RESOLVERS = [
    ("no-defense", r_no_defense),
    ("name-allowlist", r_name_allowlist),
    ("fuzzy-name", r_fuzzy_name),
    ("schema-validate", r_schema_validate),
    ("resolution-rung (ours)", r_resolution_rung),
    ("resolution+gate (ours)", r_resolution_gate),
]


# ---- MCP resolvers (call, deployment) -> allowed -------------------------

def mr_naive_host(call: MCPCall, dep: MCPDeployment) -> bool:
    return NaiveMCPHost(dep).check(call).allowed


def mr_flat_schema(call: MCPCall, dep: MCPDeployment) -> bool:
    # resolve to the FIRST provider (or pinned) then schema-check -- no shadow,
    # ambiguity, or staleness handling. Models a schema-aware but merge-naive host.
    if call.server is not None:
        s = dep.servers.get(call.server)
        if not s or not s.connected or call.tool not in s.tools:
            return False
        c = s.tools[call.tool]
    else:
        provs = dep.providers_of(call.tool)
        if not provs:
            return False
        c = provs[0].tools[call.tool]
    for k in call.args:
        if k not in c.signature:
            return False
    for k, t in c.signature.items():
        if t.required and k not in call.args:
            return False
    for k, v in call.args.items():
        if not c.signature[k].accepts(v):
            return False
    return True


def mr_highest_trust(call: MCPCall, dep: MCPDeployment) -> bool:
    # a plausible real-host policy: on a collision, pick the highest-trust
    # provider, then schema-check. Does NOT detect shadowing (it silently
    # prefers the trusted one -- but still executes a shadowed lower-trust call
    # when the trusted one is absent) and does NOT detect staleness.
    if call.server is not None:
        s = dep.servers.get(call.server)
        if not s or not s.connected or call.tool not in s.tools:
            return False
        c = s.tools[call.tool]
    else:
        provs = dep.providers_of(call.tool)
        if not provs:
            return False
        provs.sort(key=lambda s: s.trust.value, reverse=True)
        c = provs[0].tools[call.tool]
    for k in call.args:
        if k not in c.signature:
            return False
    for k, t in c.signature.items():
        if t.required and k not in call.args:
            return False
    for k, v in call.args.items():
        if not c.signature[k].accepts(v):
            return False
    return True


def mr_resolution_rung(call: MCPCall, dep: MCPDeployment) -> bool:
    return MCPResolutionRung(dep).check(call).allowed


DEFAULT_MCP_RESOLVERS = [
    ("naive-host", mr_naive_host),
    ("flat-schema", mr_flat_schema),
    ("highest-trust", mr_highest_trust),
    ("mcp-resolution-rung (ours)", mr_resolution_rung),
]
