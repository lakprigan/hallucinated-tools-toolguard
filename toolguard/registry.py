"""
registry.py -- Tool contracts and a controlled 100-tool registry.

This reproduces the precondition-effect contract formalism shared by the
research program (CMTF: arXiv:2606.06284; Contract2Tool: arXiv:2606.07904;
RACG: arXiv:2606.13884; ContractGuard: arXiv:2606.18550):

    tool = (name, requires R, produces E, risk rho, cost, signature)

We extend each contract with a typed *signature* (argument name -> ArgType),
which the prior papers left implicit. The signature is what makes hallucinated
arguments (H2), type/enum violations (H3), and borrowed signatures (H5)
mechanically detectable.

Everything here is deterministic and offline. No network, no API keys.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, FrozenSet, List, Optional, Tuple


class Risk(Enum):
    LOW = 0
    MEDIUM = 1
    HIGH = 2


class ArgKind(Enum):
    STRING = "string"
    INT = "int"
    BOOL = "bool"
    ENUM = "enum"
    ID = "id"          # opaque identifier string matching a registry-known pattern


@dataclass(frozen=True)
class ArgType:
    """Declared type of a single tool argument (part of the contract)."""
    kind: ArgKind
    required: bool = True
    enum: Optional[FrozenSet[str]] = None      # for ArgKind.ENUM
    lo: Optional[int] = None                    # for ArgKind.INT range
    hi: Optional[int] = None

    def accepts(self, value: Any) -> bool:
        if self.kind == ArgKind.STRING or self.kind == ArgKind.ID:
            return isinstance(value, str)
        if self.kind == ArgKind.BOOL:
            return isinstance(value, bool)
        if self.kind == ArgKind.INT:
            if not isinstance(value, int) or isinstance(value, bool):
                return False
            if self.lo is not None and value < self.lo:
                return False
            if self.hi is not None and value > self.hi:
                return False
            return True
        if self.kind == ArgKind.ENUM:
            return isinstance(value, str) and self.enum is not None and value in self.enum
        return False


@dataclass(frozen=True)
class Contract:
    """A precondition-effect tool contract with a typed signature."""
    name: str
    requires: FrozenSet[str]          # state predicates that must hold
    produces: FrozenSet[str]          # state predicates added on success
    risk: Risk
    cost: int
    signature: Dict[str, ArgType]     # arg name -> declared type
    authorization: Optional[str] = None   # predicate gating HIGH-risk exposure

    def digest(self) -> str:
        """A stable content hash -- stands in for ContractGuard signed provenance."""
        payload = "|".join([
            self.name,
            ",".join(sorted(self.requires)),
            ",".join(sorted(self.produces)),
            self.risk.name,
            str(self.cost),
            ",".join(f"{k}:{v.kind.value}:{int(v.required)}" for k, v in sorted(self.signature.items())),
            self.authorization or "",
        ])
        return hashlib.sha256(payload.encode()).hexdigest()[:16]


class Registry:
    """Closed-world set of trusted contracts. The verifier resolves against this."""

    def __init__(self, contracts: List[Contract]):
        self._by_name: Dict[str, Contract] = {c.name: c for c in contracts}
        assert len(self._by_name) == len(contracts), "duplicate tool names"

    def __contains__(self, name: str) -> bool:
        return name in self._by_name

    def get(self, name: str) -> Optional[Contract]:
        return self._by_name.get(name)

    def names(self) -> List[str]:
        return list(self._by_name.keys())

    def __len__(self) -> int:
        return len(self._by_name)


# --------------------------------------------------------------------------
# A controlled 100-tool registry over three domains (calendar / email / files)
# mirroring the CMTF/RACG benchmark scale. Tools are procedurally generated so
# the registry is reproducible and the signatures are non-trivial.
# --------------------------------------------------------------------------

_DOMAINS = ["cal", "mail", "file"]

# A handful of hand-authored "anchor" tools with realistic high-risk semantics.
_ANCHORS: List[Contract] = [
    Contract(
        name="send_email",
        requires=frozenset({"draft_ready", "recipient_known"}),
        produces=frozenset({"email_sent"}),
        risk=Risk.HIGH,
        cost=3,
        signature={
            "to": ArgType(ArgKind.STRING),
            "subject": ArgType(ArgKind.STRING),
            "body": ArgType(ArgKind.STRING),
            "cc": ArgType(ArgKind.STRING, required=False),
        },
        authorization="send_authorized",
    ),
    Contract(
        name="delete_file",
        requires=frozenset({"file_selected"}),
        produces=frozenset({"file_deleted"}),
        risk=Risk.HIGH,
        cost=2,
        signature={
            "path": ArgType(ArgKind.ID),
            "recursive": ArgType(ArgKind.BOOL, required=False),
        },
        authorization="delete_authorized",
    ),
    Contract(
        name="transfer_funds",
        requires=frozenset({"account_selected", "amount_set"}),
        produces=frozenset({"funds_transferred"}),
        risk=Risk.HIGH,
        cost=5,
        signature={
            "account": ArgType(ArgKind.ID),
            "amount": ArgType(ArgKind.INT, lo=1, hi=1_000_000),
            "currency": ArgType(ArgKind.ENUM, enum=frozenset({"USD", "EUR", "GBP"})),
        },
        authorization="payment_authorized",
    ),
    Contract(
        name="create_event",
        requires=frozenset({"calendar_open"}),
        produces=frozenset({"event_created"}),
        risk=Risk.LOW,
        cost=1,
        signature={
            "title": ArgType(ArgKind.STRING),
            "day": ArgType(ArgKind.INT, lo=1, hi=31),
            "priority": ArgType(ArgKind.ENUM, enum=frozenset({"low", "med", "high"}), required=False),
        },
    ),
    Contract(
        name="read_file",
        requires=frozenset({"file_selected"}),
        produces=frozenset({"file_read"}),
        risk=Risk.LOW,
        cost=1,
        signature={"path": ArgType(ArgKind.ID)},
    ),
]


def _proc_contract(i: int) -> Contract:
    """Deterministically generate a filler tool contract."""
    dom = _DOMAINS[i % 3]
    tier = i % 5
    risk = Risk.HIGH if tier == 0 else (Risk.MEDIUM if tier == 1 else Risk.LOW)
    sig = {
        f"{dom}_id": ArgType(ArgKind.ID),
        "n": ArgType(ArgKind.INT, lo=0, hi=100, required=(tier != 2)),
    }
    if tier == 3:
        sig["mode"] = ArgType(ArgKind.ENUM, enum=frozenset({"a", "b", "c"}))
    auth = f"{dom}_auth_{i}" if risk == Risk.HIGH else None
    return Contract(
        name=f"{dom}_op_{i:03d}",
        requires=frozenset({f"{dom}_ready"}),
        produces=frozenset({f"{dom}_done_{i}"}),
        risk=risk,
        cost=1 + (i % 3),
        signature=sig,
        authorization=auth,
    )


def build_registry(n_tools: int = 100) -> Registry:
    contracts = list(_ANCHORS)
    i = 0
    while len(contracts) < n_tools:
        contracts.append(_proc_contract(i))
        i += 1
    return Registry(contracts[:n_tools])


if __name__ == "__main__":
    reg = build_registry(100)
    print(f"registry: {len(reg)} tools")
    hi = [n for n in reg.names() if reg.get(n).risk == Risk.HIGH]
    print(f"high-risk tools: {len(hi)}")
    print("anchor digests:")
    for a in _ANCHORS:
        print(f"  {a.name:16s} {a.digest()}")
