"""
evaluate.py — Deterministic evaluation harness for the Bounded GraphRAG
engine (ShadowGrid).

Design goals (see project constraints):
  * No external LLM calls. Correctness is judged by deterministic
    substring/keyword matching against the raw chunk text that
    engine.query() returns, not by asking a model to grade itself.
  * Fast and repeatable: single ingestion pass, then N in-memory queries
    timed individually with time.perf_counter().
  * Exercises the specific contract points that have broken before in
    this codebase: hyphenated/numeric entities (Verification-Run-1),
    operational-prefix stripping (Project/Asset/Agent -> bare name),
    multi-hop agent<->project relationships, and the "no capitalized
    entity -> empty context, not an error" negative-path guarantee.

Usage:
    python evaluate.py
    python evaluate.py --corpus data/corpus_large.json --max-hops 2
"""

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from typing import List

from src.graph_engine import BoundedGraphRAGEngine, BoundedChunkStore
from src.ingestor import ingest_corpus


# --------------------------------------------------------------------------
# Test case definition
# --------------------------------------------------------------------------

@dataclass
class TestCase:
    name: str
    query: str
    # Substrings that MUST all appear in the returned context for a hit.
    # Leave empty (and set expect_empty=True) for negative test cases.
    expected_keywords: List[str] = field(default_factory=list)
    # If True, engine.query() is expected to return "" (no seed entity
    # resolved, or entity resolved but has no graph context).
    expect_empty: bool = False


# Ground-truth test suite, hand-verified against data/corpus_large.json's
# 5 signal documents (doc_001..doc_005). The 45 "noise_*" entries exist
# purely to stress node-count / hub-capping behavior; they contribute no
# expected keywords since their only capitalized tokens ("Infrastructure",
# "System", "Storage", ...) are filtered by ingestor.py's _STOPWORDS.
TEST_SUITE: List[TestCase] = [
    # --- Specific entity lookups -----------------------------------------
    TestCase(
        name="entity_lookup_hyphenated_numeric",
        query="Verification-Run-1",
        expected_keywords=["Verification-Run-1", "Qdrant"],
    ),
    TestCase(
        name="entity_lookup_project_prefix_strip",
        query="Project ShadowGrid",
        expected_keywords=["ShadowGrid", "PostgreSQL"],
    ),
    TestCase(
        name="entity_lookup_asset_prefix_strip",
        query="Asset Qdrant",
        expected_keywords=["Qdrant", "PostgreSQL"],
    ),
    TestCase(
        name="entity_lookup_concept_prefix_strip",
        query="Concept Vector-Embeddings",
        expected_keywords=["Vector-Embeddings", "PostgreSQL"],
    ),

    # --- Multi-hop agent/project relationships ----------------------------
    TestCase(
        name="multihop_agent_bianca",
        query="Agent Bianca",
        expected_keywords=["Bianca", "Verification-Run-1"],
    ),
    TestCase(
        name="multihop_agent_alexander",
        query="Agent Alexander",
        expected_keywords=["Alexander", "ShadowGrid"],
    ),
    TestCase(
        name="multihop_unresolved_agent",
        # "Agent Ivo" is a well-formed entity (survives extraction and
        # prefix-stripping to "Ivo") that simply never occurs in the
        # corpus. This is a distinct negative path from "no capitalized
        # entity at all" below: it exercises the empty-frontier /
        # no-matching-node case rather than the extraction case.
        query="Agent Ivo",
        expect_empty=True,
    ),

    # --- Negative test cases: no capitalized entity -----------------------
    TestCase(
        name="negative_no_entity_lowercase_sentence",
        query="how does the system work",
        expect_empty=True,
    ),
    TestCase(
        name="negative_no_entity_short_token",
        query="asd",
        expect_empty=True,
    ),
    TestCase(
        name="negative_empty_query",
        query="",
        expect_empty=True,
    ),
]


# --------------------------------------------------------------------------
# Evaluation core
# --------------------------------------------------------------------------

@dataclass
class QueryResult:
    case: TestCase
    context: str
    latency_ms: float
    found_keywords: List[str]
    missing_keywords: List[str]
    passed: bool
    recall: float


def evaluate_case(engine: BoundedGraphRAGEngine, chunk_store: BoundedChunkStore,
                   case: TestCase, max_hops: int) -> QueryResult:
    start = time.perf_counter()
    context = engine.query(case.query, chunk_store, max_hops=max_hops)
    latency_ms = (time.perf_counter() - start) * 1000.0

    if case.expect_empty:
        passed = (context == "")
        recall = 1.0 if passed else 0.0
        return QueryResult(case, context, latency_ms, [], [], passed, recall)

    found = [kw for kw in case.expected_keywords if kw in context]
    missing = [kw for kw in case.expected_keywords if kw not in context]
    recall = len(found) / len(case.expected_keywords) if case.expected_keywords else 1.0
    passed = (len(missing) == 0)
    return QueryResult(case, context, latency_ms, found, missing, passed, recall)


def run_ingestion(corpus_path: str):
    engine = BoundedGraphRAGEngine()
    chunk_store = BoundedChunkStore()

    start = time.perf_counter()
    ok = ingest_corpus(corpus_path, engine, chunk_store)
    ingest_ms = (time.perf_counter() - start) * 1000.0

    if not ok:
        print(f"FATAL: ingestion failed for corpus '{corpus_path}'.")
        sys.exit(1)

    return engine, chunk_store, ingest_ms


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------

def _truncate(text: str, width: int) -> str:
    text = text.replace("\n", " ")
    return text if len(text) <= width else text[: width - 1] + "…"


def print_diagnostic_table(results: List[QueryResult]):
    name_w, query_w, status_w, recall_w, lat_w = 32, 26, 6, 8, 10

    header = (
        f"{'Test':<{name_w}} {'Query':<{query_w}} {'Status':<{status_w}} "
        f"{'Recall':<{recall_w}} {'Latency':<{lat_w}}"
    )
    print(header)
    print("-" * len(header))

    for r in results:
        status = "PASS" if r.passed else "FAIL"
        print(
            f"{_truncate(r.case.name, name_w):<{name_w}} "
            f"{_truncate(r.case.query or '<empty>', query_w):<{query_w}} "
            f"{status:<{status_w}} "
            f"{r.recall * 100:>6.1f}% "
            f"{r.latency_ms:>8.3f}ms"
        )
        if not r.passed and not r.case.expect_empty:
            print(f"    missing keywords: {r.missing_keywords}")
        elif not r.passed and r.case.expect_empty:
            print(f"    expected empty context, got {len(r.context)} chars: "
                  f"{_truncate(r.context, 80)!r}")


def print_aggregate_metrics(results: List[QueryResult], ingest_ms: float):
    n = len(results)
    hits = sum(1 for r in results if r.passed)
    hit_rate = (hits / n * 100.0) if n else 0.0
    avg_recall = (sum(r.recall for r in results) / n * 100.0) if n else 0.0
    avg_latency = (sum(r.latency_ms for r in results) / n) if n else 0.0
    max_latency = max((r.latency_ms for r in results), default=0.0)

    print()
    print("=" * 60)
    print("AGGREGATE BENCHMARK METRICS")
    print("=" * 60)
    print(f"{'Queries evaluated':<32} {n}")
    print(f"{'Hit Rate @ K':<32} {hit_rate:.1f}%  ({hits}/{n})")
    print(f"{'Avg Context Recall':<32} {avg_recall:.1f}%")
    print(f"{'Avg Sub-graph Traversal Latency':<32} {avg_latency:.3f} ms")
    print(f"{'Max Sub-graph Traversal Latency':<32} {max_latency:.3f} ms")
    print(f"{'Ingestion Time':<32} {ingest_ms:.3f} ms")
    print("=" * 60)


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Deterministic eval harness for ShadowGrid's Bounded GraphRAG engine.")
    parser.add_argument("--corpus", default="data/corpus_large.json",
                         help="Path to the corpus JSON file (default: data/corpus_large.json)")
    parser.add_argument("--max-hops", type=int, default=2,
                         help="max_hops passed to engine.query() (default: 2)")
    args = parser.parse_args()

    print(f"Ingesting corpus: {args.corpus}")
    engine, chunk_store, ingest_ms = run_ingestion(args.corpus)
    print(f"Ingestion complete in {ingest_ms:.3f} ms "
          f"({len(engine.node_degrees)} active nodes, {len(engine.edges)} active edges)\n")

    results = [
        evaluate_case(engine, chunk_store, case, max_hops=args.max_hops)
        for case in TEST_SUITE
    ]

    print_diagnostic_table(results)
    print_aggregate_metrics(results, ingest_ms)

    failed = [r for r in results if not r.passed]
    if failed:
        print(f"\n{len(failed)} test case(s) failed.")
        sys.exit(1)
    else:
        print("\nAll test cases passed.")
        sys.exit(0)


if __name__ == "__main__":
    main()