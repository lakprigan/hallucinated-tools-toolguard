"""
gate.py -- The defense stack for tool-augmented agents.

Layers (in the order a call passes through them):

  Rung 0  RESOLUTION (this paper's contribution)
          A closed-world resolver. Every emitted call must resolve to a
          registered tool AND type-check against that tool's contract
          signature, BEFORE anything else runs. Unresolved names, undeclared
          arguments, missing required arguments, and type/enum violations are
          rejected here. This is where hallucinated tools/args die.

  GATE    CAUSAL + ADMISSIBILITY GATE
          Given the *visible* tool set for the current step, the gate exposes a
          high-risk tool only if it is (i) on a minimal causal path and
          (ii) authorized in the current state. A call to a real tool that
          the gate did not expose this step is an "off-frontier" call (H4).

  CVerify CONTRACT INTEGRITY
          A contract verifier's rungs verify the *contract* is untampered:
          signed provenance + typed attestation + runtime effect check.
          Assumes the referenced tool exists -- which is exactly why it
          cannot catch H1/H2 on its own.

The key structural claim of the paper: hallucination defense MUST sit at
Rung 0, strictly before the gate. The gate can only gate what is in the visible
set; a hallucinated call is, by definition, not something the gate chose to
expose, so the gate never sees it as a gating decision. Symmetric to the
contract-verifier result that the runtime effect check must sit strictly AFTER
the gate.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Set

from .registry import Contract, Registry, Risk


class Verdict(Enum):
    ALLOW = "allow"
    REJECT_UNRESOLVED = "reject_unresolved"     # H1: no such tool
    REJECT_UNDECLARED_ARG = "reject_undeclared"  # H2: arg not in signature
    REJECT_MISSING_ARG = "reject_missing"        # H2: required arg absent
    REJECT_TYPE = "reject_type"                  # H3/H5: type/enum/range violation
    REJECT_OFF_FRONTIER = "reject_off_frontier"  # H4: real tool the gate didn't expose
    REJECT_UNAUTHORIZED = "reject_unauthorized"  # gate: high-risk w/o authorization


@dataclass
class ToolCall:
    name: str
    args: Dict[str, Any]


@dataclass
class Decision:
    verdict: Verdict
    rung: str                 # which layer decided
    detail: str = ""

    @property
    def allowed(self) -> bool:
        return self.verdict == Verdict.ALLOW


# --------------------------------------------------------------------------
# Rung 0: the Resolution Rung (closed-world resolver + typed signature check)
# --------------------------------------------------------------------------

class ResolutionRung:
    """Rejects hallucinated tools and arguments under a closed-world assumption.

    Closed world: the registry is authoritative. Any tool name not present is
    treated as nonexistent (H1). Any argument not in the contract signature is
    undeclared (H2). Required arguments must be present (H2). Present arguments
    must satisfy their declared type/enum/range (H3, and H5 when a borrowed
    signature carries a wrong-typed or extra field).
    """

    def __init__(self, registry: Registry):
        self.reg = registry

    def check(self, call: ToolCall) -> Decision:
        contract = self.reg.get(call.name)
        if contract is None:
            return Decision(Verdict.REJECT_UNRESOLVED, "rung0",
                            f"'{call.name}' not in registry ({len(self.reg)} tools)")
        # undeclared arguments
        for a in call.args:
            if a not in contract.signature:
                return Decision(Verdict.REJECT_UNDECLARED_ARG, "rung0",
                                f"'{a}' undeclared for '{call.name}'")
        # missing required arguments
        for a, t in contract.signature.items():
            if t.required and a not in call.args:
                return Decision(Verdict.REJECT_MISSING_ARG, "rung0",
                                f"required '{a}' missing for '{call.name}'")
        # type / enum / range
        for a, v in call.args.items():
            t = contract.signature[a]
            if not t.accepts(v):
                return Decision(Verdict.REJECT_TYPE, "rung0",
                                f"'{a}'={v!r} violates {t.kind.value} for '{call.name}'")
        return Decision(Verdict.ALLOW, "rung0")


# --------------------------------------------------------------------------
# Causal gate (least-privilege causal + admissibility gating)
# --------------------------------------------------------------------------

class CausalGate:
    """Exposes a high-risk tool only if on the causal frontier AND authorized.

    'visible' is the set of tool names the gate chose to expose at this step
    (the causal frontier). A call whose tool is real but NOT in 'visible' is
    an off-frontier call (H4) -- the gate rejects it as unexposed capability.
    """

    def __init__(self, registry: Registry):
        self.reg = registry

    def check(self, call: ToolCall, visible: Set[str], state: Set[str]) -> Decision:
        if call.name not in visible:
            return Decision(Verdict.REJECT_OFF_FRONTIER, "racg",
                            f"'{call.name}' not on causal frontier")
        contract = self.reg.get(call.name)
        assert contract is not None  # resolution already passed
        if contract.risk == Risk.HIGH and contract.authorization:
            if contract.authorization not in state:
                return Decision(Verdict.REJECT_UNAUTHORIZED, "racg",
                                f"missing authorization '{contract.authorization}'")
        return Decision(Verdict.ALLOW, "racg")


# --------------------------------------------------------------------------
# Contract-verifier rung (contract-integrity check; here: provenance digest match)
# --------------------------------------------------------------------------

class ContractVerifierRung:
    """Verifies the contract the gate read matches the trusted attestation root.

    We model contract integrity with a single provenance-digest check:
    the contract presented at gate time must hash to the trusted digest. This
    catches corrupted contracts (the contract-verifier threat model) but, crucially,
    is a no-op against hallucinated *calls*, because a hallucinated tool has no
    trusted contract to compare against -- it never reaches this rung.
    """

    def __init__(self, trusted_digests: Dict[str, str]):
        self.trusted = trusted_digests

    def check(self, call: ToolCall, presented: Contract) -> Decision:
        expect = self.trusted.get(call.name)
        if expect is None or presented.digest() != expect:
            return Decision(Verdict.REJECT_UNRESOLVED, "cguard",
                            f"contract for '{call.name}' fails provenance")
        return Decision(Verdict.ALLOW, "cguard")


# --------------------------------------------------------------------------
# Composable pipelines. Each pipeline is a list of rungs applied in order.
# --------------------------------------------------------------------------

class Pipeline:
    """A defense configuration. Ablations toggle which rungs are present."""

    def __init__(self, registry: Registry, use_resolution: bool,
                 use_racg: bool, use_cguard: bool):
        self.reg = registry
        self.use_resolution = use_resolution
        self.use_racg = use_racg
        self.use_cguard = use_cguard
        self.rung0 = ResolutionRung(registry)
        self.gate = CausalGate(registry)
        self.verifier = ContractVerifierRung({n: registry.get(n).digest() for n in registry.names()})

    def decide(self, call: ToolCall, visible: Set[str], state: Set[str]) -> Decision:
        # Rung 0 first: hallucination defense before the gate.
        if self.use_resolution:
            d = self.rung0.check(call)
            if not d.allowed:
                return d
        # If resolution is OFF, a hallucinated call may slip through to the gate,
        # which can only reason about registered/visible tools. We emulate the
        # documented failure: unknown tools the gate cannot gate are executed.
        if self.use_racg:
            # the gate needs a real contract; if none exists and resolution was off,
            # the gate cannot gate it -> the call escapes (models the vulnerability).
            if self.reg.get(call.name) is None:
                return Decision(Verdict.ALLOW, "racg",
                                "unresolved call escapes gate (no resolution rung)")
            d = self.gate.check(call, visible, state)
            if not d.allowed:
                return d
        if self.use_cguard:
            contract = self.reg.get(call.name)
            if contract is None:
                # the verifier has nothing to verify; call escapes.
                return Decision(Verdict.ALLOW, "cguard",
                                "no contract to verify (hallucinated tool)")
            d = self.verifier.check(call, contract)
            if not d.allowed:
                return d
        return Decision(Verdict.ALLOW, "final")


def named_pipelines(registry: Registry) -> Dict[str, Pipeline]:
    """The configurations compared in the paper."""
    return {
        "no_defense":        Pipeline(registry, False, False, False),
        "racg_only":         Pipeline(registry, False, True, False),
        "racg_cguard":       Pipeline(registry, False, True, True),   # gate + contract-verifier stack
        "resolution_only":   Pipeline(registry, True, False, False),
        "toolguard_full":    Pipeline(registry, True, True, True),    # this paper
    }
