"""
toolguard -- Closed-world resolution against tool hallucination in LLM agents
and the Model Context Protocol (MCP).

Public API (stable across 0.x):

    from toolguard import (
        __version__,
        # single-registry
        build_registry, Registry, ResolutionRung, ToolCall,
        # MCP multi-server
        build_deployment, MCPResolutionRung, MCPCall, NaiveMCPHost,
        # benchmark harness (leaderboard)
        HALLUCINATION_BENCH, run_leaderboard, score_resolver,
    )

The benchmark (`toolguard.bench`) scores ANY resolver -- ours or a third
party's -- against a fixed, versioned suite of hallucination probes (H1-H5 for a
single registry, M1-M5 for an MCP deployment) and emits a standard leaderboard
JSON plus a printable scoreboard. Install with `pip install toolguard` and run
`toolguard-bench` (or `python -m toolguard.bench`).
"""

from __future__ import annotations

__version__ = "0.2.0"
BENCHMARK_VERSION = "htb-1.0"  # Hallucinated-Tools Benchmark, dataset version

# ---- single-registry surface --------------------------------------------
from .registry import (ArgKind, ArgType, Contract, Registry, Risk,
                       build_registry)
from .gate import (ResolutionRung, CausalGate, ContractVerifierRung, Pipeline,
                   ToolCall, Verdict, Decision, named_pipelines)

# ---- MCP multi-server surface --------------------------------------------
from .mcp import (MCPServer, MCPDeployment, MCPResolutionRung, NaiveMCPHost,
                  MCPCall, MCPDecision, MCPVerdict, MergePolicy, Trust,
                  QualifiedName, build_deployment)

# ---- benchmark / leaderboard --------------------------------------------
from .bench import (HALLUCINATION_BENCH, MCP_BENCH, run_leaderboard,
                    score_resolver, score_mcp_resolver, Resolver, MCPResolver)

__all__ = [
    "__version__", "BENCHMARK_VERSION",
    "ArgKind", "ArgType", "Contract", "Registry", "Risk", "build_registry",
    "ResolutionRung", "CausalGate", "ContractVerifierRung", "Pipeline", "ToolCall",
    "Verdict", "Decision", "named_pipelines",
    "MCPServer", "MCPDeployment", "MCPResolutionRung", "NaiveMCPHost", "MCPCall",
    "MCPDecision", "MCPVerdict", "MergePolicy", "Trust", "QualifiedName",
    "build_deployment",
    "HALLUCINATION_BENCH", "MCP_BENCH", "run_leaderboard", "score_resolver",
    "score_mcp_resolver", "Resolver", "MCPResolver",
]
