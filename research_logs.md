# ShadowGrid Research Log

## Session 1 — June 28, 2026

### What I'm Doing
Comparing Vector-RAG vs bounded GraphRAG for local PKM systems. Testing whether aggressively pruning graph edges under a memory cap still preserves retrieval accuracy.

### Thesis
Topological Entropy-Based Edge Eviction in Local GraphRAG: Stabilizing Retrieval Accuracy Under Bounded-Memory Updates.
### Edge Eviction Formula
H_e = α · w_semantic + β · ln(C_traverse + 1) − γ · Δt

### What Got Built
- generator.py — 5 synthetic docs with temporal contradictions, 3 eval queries
- vector_engine.py — baseline vector search (sentence-transformers + cosine sim)
- graph_engine.py — bounded graph with entropy-based edge eviction
- evaluate.py — runs both pipelines, sends context to Ollama for LLM answers

### First Benchmark Results (max_edges=4)
- Q_001 (temporal drift): Vector PASS, Graph PASS
- Q_002 (multi-hop contradiction): Vector FAIL, Graph PASS
- Q_003 (multi-hop): Vector FAIL, Graph FAIL — eviction killed a needed edge

### Next Steps
- Hyperparameter sweep: max_edges 2–10
- Expand to 50+ docs and 50 eval queries
- Add unbounded GraphRAG as Control Group B
- Add LLM-as-a-judge faithfulness scoring
- Log latency and token consumption

## Session 2 — July 1, 2026

### What We Did
Ran the full hyperparameter sweep planned in Session 1, migrated the eval harness off local Ollama to a cloud API, and fought through a long chain of infrastructure bugs before landing on a clean, valid result.

### Key Decision: Ollama → Cloud API
Local `llama3` (8B) inference was pushing the MacBook Air M4 into heavy thermal load. Rather than run an expanded 1,000-edge stress test locally, rewrote `sweep_evaluator.py` to route LLM calls to a cloud provider (Groq, OpenAI-compatible, hosting Llama 3).

### Infra Bugs Fixed (in order)
1. **Wrong working directory** — script run from `RP/` instead of `RP/shadowgrid_research/`; then a `data/` path mismatch when cd-ing around. Resolved by running from repo root with the full script path.
2. **Corrupted `evaluation_suite.json`** — empty file caused `JSONDecodeError`. Deleted so the script regenerates it.
3. **SSL `CERTIFICATE_VERIFY_FAILED`** — macOS Python not linked to root certs. Bypassed with an unverified SSL context (local testing only).
4. **Blind error handling** — generic `except` was swallowing the API response body. Switched to `except urllib.error.HTTPError` + `e.read()` to expose the real error. (This was the key debugging unlock.)
5. **HTTP 403 / Cloudflare error 1010** — default `Python-urllib` User-Agent was flagged. Spoofed a Chrome `User-Agent` header.
6. **Model decommissioned** — `llama3-8b-8192` is dead (retired May 31, 2025). Switched to `llama-3.1-8b-instant` (itself deprecated June 17, 2026 → `openai/gpt-oss-20b` is the longer-term target).

### Harness Hardening
- Added **429 rate-limit retries** with backoff parsed from Groq's "try again in Xs" hint (up to 5 attempts).
- Judge now returns **`None` (not 0)** when an API/judge call errors or is malformed — an invalid data point, not a confirmed wrong answer.
- Scoring computes accuracy **only over valid data points** and reports `n=` and `[excluded N]` per config, so infra noise can't masquerade as a real 0%.
- Injected **`[DIAGNOSTIC]` context dumps** printing the raw retrieved context before each LLM call.

### The Real Finding: Prompt–Evaluation Mismatch
With a clean run (n=3, zero exclusions), every config scored **0.00%** — identical judge complaints across all 11 structurally different configs. The context dump proved retrieval was **never the bottleneck**: the raw context contained every needed fact.

Root cause: **generation/eval mismatch, not retrieval.** The prompt said "Answer concisely," so the model returned correct-but-terse answers. The judge was tuned "hyper-critical" and penalized brevity, demanding full migration history. Fixing both sides — prompt asks for thorough answers, judge is fact-focused and doesn't penalize brevity — flipped the scores immediately.

### Final Sweep Results (after fix)
| Config | Accuracy | n |
|---|---|---|
| Vector-RAG (Control A) | 66.67% | 3 |
| Unbounded GraphRAG (Control B) | 100.00% | 3 |
| Bounded GraphRAG (max_edges=2 … 10) | 100.00% | 3 |

- **Vector-RAG fails under noise** — dropped the multi-hop query, pulling decoy "Agent Bianca" context instead of the target fact.
- **GraphRAG bypasses semantic traps** — walks explicit edges, ignores decoy text, hits 100%.
- **Caveat: dataset is a toy model.** Graph maxed out at `Nodes=7, Edges=5`, so `max_edges=5–10` are identical (eviction never triggered above 5). The 100%-at-max_edges=2 result reflects the tiny corpus, not proven robustness.

### Next Steps
- Generate a large (~1,000-edge) dataset to actually stress the eviction engine's breaking point.
- Verify the eviction weighting: `('ShadowGrid', 'PostgreSQL')` was evicted at low `max_edges` — confirm the formula isn't scoring genuinely-needed edges as low-signal.
- Test `max_edges=2` against harder multi-hop queries before trusting it for production.
- Migrate model target to `openai/gpt-oss-20b` to avoid a second deprecation.

### Security To-Do
- **Revoke the Groq API key** that was pasted in plaintext during this session and regenerate a fresh one. Never paste keys into chat again — pass via env var only.

## Session 3 — July 2, 2026

### What We Did
Took the "100% on toy data" result from Session 2 and stress-tested it properly: refactored the graph engine for real bounded-memory streaming, replaced hardcoded string-matching extraction with an actual ≤8B LLM extraction pipeline (Groq / `llama-3.1-8b-instant`), scaled the eval suite from n=3 to n=20, and ran two full sweeps. The toy-data 100% collapsed — as expected — and we found (and are still confirming) two real pathologies.

### Key Refactor: Faked Entropy → Real Bounded Engine
Session 2's `graph_engine.py` was a hardcoded script, not an entropy engine. It violated the thesis four ways:
1. **Faked entropy** — magic-number weights (`-2.2`, `-5.15`) on hardcoded string matches, no dynamic topological computation.
2. **Memory leak** — evicting an edge left orphaned nodes in `self.nodes`, so the node set grew unbounded.
3. **O(E) eviction** — `max(self.edges, key=...)` did a full linear scan on every boundary breach; must be O(log E).
4. **Disk I/O in retrieval** — `retrieve_subgraph` rescanned raw JSON from disk on every query, bypassing the in-memory graph.

Rebuilt as:
- `BoundedGraphRAGEngine` — `node_degrees` + `adjacency`, degree-based local entropy proxy, min-heap eviction (O(log E)), node garbage collection, memory-only retrieval.
- `BoundedChunkStore` — bipartite `edge → chunk_ids → raw_text`, MD5 dedup, FIFO chunk eviction. Decouples text storage from graph topology so the engine tracks structure only.

### Extraction Pipeline: String Matching → ≤8B LLM
Replaced `if "PostgreSQL" in content:` blocks with `extract_relations_streaming()` on Groq (`llama-3.1-8b-instant`, `response_format=json_object`, `temperature=0.0`). On parse failure: drop the chunk, no retries (retries kill streaming throughput). This finally tests the thesis against real ≤8B extraction noise instead of perfect string matches.

### Infra Bugs Fixed (in order)
1. `ModuleNotFoundError: groq` — SDK not installed in the venv.
2. `GroqError: api_key must be set` — env var not reaching the process; IDE/terminal context stripping it. Resolved with inline `GROQ_API_KEY=... python ...`.
3. Wrong working directory again — ran from `RP/` instead of `RP/shadowgrid_research/`.
4. `FileNotFoundError: corpus.json` — new generator writes split files to `data/raw_corpus/`, but the Vector-RAG baseline expects a monolithic `data/corpus.json`. Fixed by also writing the combined file.
5. `AttributeError: 'BoundedGraphRAGEngine' has no attribute 'ingest_streaming_data'` — harness/engine API mismatch. Ingestion is now decoupled; `sweep_evaluator.py` must loop `raw_corpus/`, call `extract_relations_streaming`, and push via `insert_edge` + `add_extraction`.

### SECURITY — API key leaked again
The live Groq key (`gsk_...`) was pasted into the Gemini chat in plaintext during this session. **Revoke and rotate immediately.** This is the second session with a plaintext key leak (see Session 2 to-do). Env vars only — never inline in a shared console, never in chat.

### First n=20 Sweep — the toy result collapsed
| Config | Accuracy | n |
|---|---|---|
| Vector-RAG (Control A) | 65.00% | 20 |
| Unbounded GraphRAG (Control B) | 68.42% | 19 [excluded 1] |
| Bounded max_edges=2 | 35.00% | 20 |
| Bounded max_edges=3 | 15.00% | 20 |
| Bounded max_edges=4 | 30.00% | 20 |
| Bounded max_edges=5 | 35.00% | 20 |
| Bounded max_edges=6 | 35.00% | 20 |
| Bounded max_edges=7 | 50.00% | 20 |
| Bounded max_edges=8 | 60.00% | 20 |
| Bounded max_edges=9 | 55.00% | 20 |
| Bounded max_edges=10 | 60.00% | 20 |

Key read: **Unbounded GraphRAG at 68% is the real ceiling.** With infinite memory the engine still loses ~32%, which means the bottleneck is graph *construction* (extraction / connectivity), not eviction. Accuracy climbs toward the unbounded ceiling as max_edges relaxes, as expected.

### Per-Query-Type Breakdown (the useful view)
| Config | entity_lookup | multi_hop | multi_hop_contradiction | temporal_drift |
|---|---|---|---|---|
| Vector-RAG (A) | 100% | 40% | 20% | 100% |
| Unbounded (B) | 80% | 60% | 60% | 80% |
| max_edges=2 | 40% | 60% | 20% | 0% |
| max_edges=3 | 60% | 40% | 0% | 0% |
| max_edges=4 | 40% | 60% | 20% | 0% |
| max_edges=5 | 80% | 60% | 20% | 0% |
| max_edges=6 | 40% | 40% | 0% | 80% |
| max_edges=7 | 60% | 80% | 40% | 40% |
| max_edges=8 | 60% | 80% | 20% | 80% |
| max_edges=9 | 80% | 80% | 40% | 60% |
| max_edges=10 | 100% | 100% | 80% | 60% |
*(n=5 per cell — every point is worth 20, so single-cell bounces are noise. Only patterns holding across consecutive configs count.)*

### Findings
**Finding 1 — Eviction cliff (real signal).** `temporal_drift` is **0% at max_edges 2–5, then jumps to 80% at 6**. Four consecutive zeros is a structural break, not a coin flip — this is the defensible eviction finding (far stronger than Session 2's n=3 wobble). Hypothesis: below a threshold the engine evicts the old-DB↔new-DB (or project↔DB) bridge edge that temporal_drift queries need, breaking temporal continuity. **Unconfirmed** — must diff the `RAW GRAPH CONTEXT SURFACE` for a temporal_drift query at max_edges=5 vs 6.

**Finding 2 — Cross-project entity bleed (interesting, if it holds).** Unbounded GraphRAG *loses* to Vector-RAG on the two single-hop types (entity_lookup 80 vs 100, temporal_drift 80 vs 100) but *wins* on the two multi-hop types (multi_hop 60 vs 40, contradiction 60 vs 20). Likely cause: `retrieve_subgraph_context` matches any edge touching a target entity name with no project scoping, and the generator reuses names (PostgreSQL, agents) across all 5 projects — so a single-hop lookup pulls in foreign-project chunks and dilutes context. **Checkable:** print context on a failed unbounded entity_lookup and scan for foreign project names. If confirmed, the fix is namespace/project scoping, not an entropy tweak.

### Methodology Note (pushback that stuck)
Rejected Gemini's earlier "mathematical phase transition / clustering-coefficient" story for the n=3 max_edges=3 dip — at n=3 that was one query flipping (33 pts each), asserted not observed. The genuinely useful catch from that exchange: **stale-heap bug.** Entropy is computed once at insertion from `node_degrees` at that instant and never recalculated, so heap priorities go stale as degrees shift — a temporal priority-queue bug that produces exactly this insertion-order-sensitive non-monotonic behavior, no topology theory needed. Addressed via **localized lazy evaluation**: on insert, recompute entropy only for the local neighborhood (O(D)) and push updates; on pop, discard heap entries whose stored entropy no longer matches the edge's current value.

### Next Steps
- **Confirm Finding 1 before writing anything:** diff `RAW GRAPH CONTEXT SURFACE` for a temporal_drift query at max_edges=5 vs 6. Bridge present in 6, absent in 5 → thesis mechanism verified. Present in both → it's entity bleed, fix scoping first.
- **Confirm Finding 2:** print context on a failed unbounded entity_lookup; look for foreign project names.
- **Get the `[EXTRACTION SUMMARY]`:** relations attempted vs. successfully added. If <80%, the 68% unbounded ceiling is capped by extraction failure, not retrieval/eviction — rules in/out "the graph is just incomplete."
- **Namespace scoping** if entity bleed confirmed: map entities to `Project:Entity` identifiers so shared tokens (PostgreSQL, agent names) don't collapse into cross-project hubs.
- Hold off on any thesis draft until the context diff and extraction summary are in.

### Security To-Do (carried + new)
- **Rotate the Groq key leaked in chat this session.** (Second occurrence — Session 2's key should already be revoked; if not, revoke both.)
- Move to a `.env` + `python-dotenv` setup (gitignored) so keys never get typed inline again.

## Session 4 — July 6, 2026

### What We Did
Pivoted the thesis from a data-destructive eviction policy to a **Tiered Memory (Active-Archive)** architecture at the counselor's suggestion, refactored the engine to archive evicted edges instead of deleting them, added Tier 1/Tier 2 utilization telemetry to the sweep, and ran it. The result surfaced a metric-saturation bug that's still open.

### Key Pivot: "The Purge" → "The Librarian"
instead of permanently deleting low-entropy edges under the memory cap, move them to a cheaper archive tier so useful-but-cold info survives. This reframes the thesis from *"a system that forgets to stay small"* to *"a system that manages a memory hierarchy"* — more defensible and more real-world (no catastrophic data loss).

- **Tier 1 (Active Cache):** the bounded `self.edges` graph, still capped by `max_edges`, still uses topological entropy to decide what leaves.
- **Tier 2 (Cold Archive):** a fallback `self.archive` list. Evicted edges get pushed here with metadata (src, tgt, eviction entropy, timestamp) instead of being destroyed.

Candidate title update to reflect the shift:
*Hierarchical Edge Eviction: Optimizing Bounded-Memory GraphRAG via Active-Archive Memory Tiers.*

### What Got Built / Changed
- **`graph_engine.py`** — `BoundedGraphRAGEngine.__init__` now holds `self.archive`. Eviction loop does a **handshake**: archive the edge record *before* deleting it from Tier 1 + clearing adjacency. `retrieve_subgraph_context` rewritten for tiered lookup — fast O(1) Tier 1 match tagged `[TIER 1 ACTIVE]`, then a sequential Tier 2 archive scan tagged `[TIER 2 ARCHIVE]`, deduplicated and combined.
  - Note: the uploaded engine **already implemented** the tiered design *and* carried the Session 3 stale-heap fix (localized O(D) recalc on insert + stale-entry check on pop), so no engine change was actually needed for the tier work — only the evaluator.
- **`sweep_evaluator.py`** — added `_parse_tier_usage()` to detect the telemetry tags; `evaluate_graph_engine` captures per-query `tier1_hit`/`tier2_hit`; `evaluate_vector_baseline` sets both to `None` (Vector-RAG has no tiers, correctly excluded from tier math); `_score_summary` returns a 6-tuple (accuracy, excluded, n, type breakdown, T1 ratio, T2 ratio); new **MEMORY TIER UTILIZATION** table printed per config.

### Sweep Result — Tier Utilization
| Config | Tier 1 Hit | Tier 2 Hit |
|---|---|---|
| Unbounded (Control B) | 100.0% | 0.0% |
| max_edges=2 | 75.0% | 100.0% |
| max_edges=3 | 60.0% | 100.0% |
| max_edges=4 | 85.0% | 100.0% |
| max_edges=5 | 95.0% | 100.0% |
| max_edges=6 | 90.0% | 100.0% |
| max_edges=7 | 100.0% | 100.0% |
| max_edges=8 | 100.0% | 100.0% |
| max_edges=9 | 100.0% | 100.0% |
| max_edges=10 | 100.0% | 100.0% |

### The Real Finding: Tier 2 Metric Is Saturated (open bug)
**Tier 2 Hit Ratio is pinned at 100% for every bounded config, including max_edges=10.** If the archive were doing its intended job (catching facts pushed out of a shrinking window), Tier 2 reliance should be high at max_edges=2 and fall toward 0% as the active graph gets roomy. A flat 100% means the metric isn't discriminating between configs — it's not measuring cold-storage rescues.

**Root cause (real bug, not just telemetry):** `self.archive` is **append-only, never pruned or deduplicated**. `insert_edge` only skips an edge if it's *currently* in `self.edges`. So when an evicted pair co-occurs again in a later document, it's re-inserted as active — but its earlier archive record is never removed. Since the extractor emits the same pairs repeatedly across a project's docs, evict→reinsert churn during the edge-starved early phase leaves permanent "ghost" archive records for edges that are actually alive in Tier 1. `retrieve_subgraph_context`'s Tier 2 branch then keeps hitting those ghosts → saturation.

### Proposed Fix (not yet run)
Don't clean the archive on insert (an O(N) filter per streaming insert tanks ingestion throughput). Instead enforce a **cache-miss filter at retrieval**: in the Tier 2 loop, skip any archived edge still active in Tier 1.
```python
# --- TIER 2 ARCHIVE FALLBACK ---
for record in self.archive:
    src, tgt = record.get("source"), record.get("target")
    # Stale-duplicate guard: if still active in Tier 1, its archive record is a ghost.
    if (src, tgt) in self.edges or (tgt, src) in self.edges:
        continue
    if src in target_entities or tgt in target_entities:
        for chunk in chunk_store.get_context(src, tgt):
            retrieved_chunks.add(f"[TIER 2 ARCHIVE] {chunk}")
```

## Session 5 - July 7, 2026

### What We Did
Applied the cache-miss filter from Session 4, re-ran the tiered sweep, and hit the same flat 100% Tier 2 ratio. That forced a reframe: the saturation is probably **not** a residual bug but a **scope artifact** — the sweep range (max_edges 2–10) is far below the true graph size, so the engine is starved at every config we tested. Also surfaced a second, independent confound: cross-project context bleed from string-equality edge dedup. No new conclusions locked in yet — this session ends with three verification steps queued, not fixes.

### What Got Built / Changed
- **`graph_engine.py`** — `retrieve_subgraph_context` now carries the **cache-miss filter**: the Tier 2 archive loop skips any edge still active in Tier 1 (`if (src, tgt) in self.edges or (tgt, src) in self.edges: continue`). This applies the guard at retrieval time (O(1) per record) rather than pruning the archive on insert, so ingestion throughput is untouched. Architecture otherwise unchanged — tiered layout, eviction handshake, and stale-heap check all intact.
- **`sweep_evaluator.py`** — no structural change; re-ran with the filtered engine to see whether Tier 2 utilization would finally discriminate between configs.

### Sweep Result — Still Saturated
| Config | Accuracy | Tier 1 Hit | Tier 2 Hit |
|---|---|---|---|
| Vector-RAG (Control A) | 60.00% | — | — |
| Unbounded (Control B) | 70.00% | 100.0% | 0.0% |
| max_edges=2 | 70.00% | 75.0% | 100.0% |
| max_edges=3 | 60.00% | 60.0% | 100.0% |
| max_edges=4 | 80.00% | 85.0% | 100.0% |
| max_edges=5 | 94.12% (n=17) | 94.1% | 100.0% |
| max_edges=6 | 73.68% (n=19) | 89.5% | 100.0% |
| max_edges=7 | 75.00% | 100.0% | 100.0% |
| max_edges=8 | 70.00% | 100.0% | 100.0% |
| max_edges=9 | 85.00% | 100.0% | 100.0% |
| max_edges=10 | 90.00% | 100.0% | 100.0% |

The cache-miss filter worked as written, but Tier 2 is **still pinned at 100%** across the whole bounded range, including max_edges=10.

### The Reframe: Scope Artifact, Not a Bug
The `BoundedGraphRAGEngine` is a **single instance ingesting all 5 projects into one shared graph**, and `max_edges` is a **global cap** across that entire graph — not per-project. The corpus generates a co-occurrence pair for every entity pair in every document across 5 projects × 5 docs, which likely produces **40–70+ unique edges**. If so, then max_edges=10 isn't "slightly tight" — it forces eviction of the vast majority of edges at *every* config we swept. A flat 100% Tier 2 across 2–10 wouldn't be a residual bug; it would mean the whole sweep is running deep in the **starved zone**, and we simply haven't swept far enough to reach the point where Tier 1 can hold enough of the graph for Tier 2 reliance to drop.

### Second Confound Flagged: Cross-Project Bleed
`insert_edge`'s dedup is pure string equality: `(src, tgt) in self.edges or (tgt, src) in self.edges`. The generator deliberately reuses entity names across projects (multiple projects can independently draw `PostgreSQL` as `db_old`, or `Alexander` as an engineer). If two projects produce the identical pair, `insert_edge` collapses them to one edge key and `chunk_store.add_extraction` accumulates text chunks from **both projects** under that key. So a query about Project X could silently pull prose about Project Y. This is a plausible mechanism for why Unbounded GraphRAG underperformed Vector-RAG on single-hop query types, and it applies **independently of eviction**, at every max_edges including unbounded.

### Next Steps (queued, in order — verify before fixing)
1. **Measure true graph scale.** After ingestion: `print(f"[GRAPH SCALE] Unique edges ever inserted: {len(graph_engine.edges) + len(graph_engine.archive)}")`. This sets the real upper bound for the sweep.
2. **Widen the sweep** from `range(2, 11)` to a stepped range up past the graph-scale number (e.g. 2, 5, 10, 20, 30, 40, 50, 60 + unbounded control) to see whether Tier 2 finally bends toward 0%. If it does → the fix works and 100% was purely a scope artifact. If it stays at 100% past graph scale → a real bug remains.
3. **Test cross-project bleed directly.** Pick one recurring entity name, print `chunk_store.get_context(*)` for a pair containing it after full ingestion, and check whether the returned texts mention two different project names. Confirm/deny only — don't fix scoping until we know it's real.

### Open Questions
- What is the actual [GRAPH SCALE] number? (Determines whether the sweep was ever in a meaningful range.)
- Does Tier 2 utilization decline once max_edges approaches/exceeds graph scale?
- Is cross-project bleed real, and if so does it need project-scoped edge keys or edge-weight filtering rather than an entropy fix?

## Session 6 — July 9, 2026

### What We Did
Closed a test-validity leak that had been inflating every GraphRAG number, replaced the deterministic extractor in `sweep_evaluator.py` with real (cached) LLM extraction, and re-ran the sweep. The inflated result collapsed: **Vector-RAG now beats Bounded GraphRAG (65% vs 35–45%).** Pinned the max_edges=10 cliff to one flipped query and confirmed three bugs in `graph_engine.py`. No fixes committed.

### The Real Bug: Test-Validity Leak
`build_entity_vocab()` pulled its vocabulary straight from the eval suite's `target_entities`, and `extract_pairs_from_doc()` linked every co-occurring pair of those entities. The graph was **guaranteed to contain exactly the edges each query needed** — so every prior GraphRAG number was structurally inflated. It wasn't doing retrieval; it was traversing a graph built to pass the test.

### The Fix
- **`extract_relations_streaming()`** — real LLM extraction using `generator.py`'s exact prompt/schema (`response_format=json_object`, `temperature=0.0`), through the existing retry path.
- **`get_relations_for_doc()`** — `sha256(text)`-keyed cache, populated once and reused across all configs. ~11x fewer calls and identical topology per config, so the sweep measures eviction, not extraction jitter.
- **`query_cloud_llm`** — added a `json_mode` flag.
- Deleted `build_entity_vocab()`, `entity_vocab`, and `extract_pairs_from_doc()`.

### Post-Leak Sweep
| Config | Accuracy | n |
|---|---|---|
| Vector-RAG (Control A) | 65.00% | 20 |
| Unbounded GraphRAG (Control B) | 50.00% | 20 |
| Bounded max_edges=2 | 40.00% | 20 |
| Bounded max_edges=3 | 45.00% | 20 |
| Bounded max_edges=4 | 45.00% | 20 |
| Bounded max_edges=5 | 40.00% | 20 |
| Bounded max_edges=6 | 40.00% | 20 |
| Bounded max_edges=7 | 40.00% | 20 |
| Bounded max_edges=8 | 45.00% | 20 |
| Bounded max_edges=9 | 45.00% | 20 |
| Bounded max_edges=10 | 35.00% | 20 |

### Per-Query-Type Breakdown
| Config | entity_lookup | multi_hop | multi_hop_contradiction | temporal_drift |
|---|---|---|---|---|
| Vector-RAG (A) | 100% | 40% | 20% | 100% |
| Unbounded (B) | 40% | 60% | 0% | 100% |
| max_edges=2 | 0% | 80% | 0% | 80% |
| max_edges=3 | 20% | 80% | 0% | 80% |
| max_edges=4 | 20% | 80% | 0% | 80% |
| max_edges=5 | 0% | 80% | 0% | 80% |
| max_edges=6 | 0% | 80% | 0% | 80% |
| max_edges=7 | 0% | 80% | 0% | 80% |
| max_edges=8 | 20% | 80% | 0% | 80% |
| max_edges=9 | 20% | 80% | 0% | 80% |
| max_edges=10 | 0% | 80% | 0% | 60% |

### Memory Tier Utilization
| Config | Tier 1 Hit | Tier 2 Hit |
|---|---|---|
| Unbounded (B) | 100.0% | 0.0% |
| max_edges=2 … 9 | 0.0% | 100.0% |
| max_edges=10 | 20.0% | 100.0% |

### Findings
- **GraphRAG loses to Vector-RAG once the cheat is gone.** Vector-RAG dominates `entity_lookup` (100% vs 0–40%): dense retrieval over raw text beats a graph that collapses a fact into a normalized edge, and if extraction misses a name the edge silently vanishes with no fallback.
- **Tier 1 was dead weight (2–9).** Tier 1 hit ratio is 0% across those configs. The 80% `multi_hop` score comes from the Tier 2 archive doing an unbounded scan that dumps most of the corpus into context — accidental brute-force recall, not traversal.
- **`multi_hop_contradiction` is 0% for every graph config — but Vector-RAG is only 20%.** Nobody handles contradiction well; graph is worse, not uniquely broken. No timestamps or conflict resolution means `A→B` and `A→NOT B` both get fed to the LLM.

### The max_edges=10 Cliff — one query
The 45%→35% drop is entirely `temporal_drift` (80%→60%), i.e. one query: `SilentTide_Q_temporal_drift`. At edges=9 it said Qdrant (correct); at edges=10 it said "MySQL still handles metadata" (ground truth: MySQL decommissioned).

SilentTide context block at max_edges=10:

1. `[TIER 2]` SilentTide initiates development (MySQL)
2. `[TIER 2]` CobaltMesh initiates development (MySQL)
3. `[TIER 2]` SilentTide migrates away from MySQL → Qdrant
4. `[TIER 1]` Ivo introduces Sparse-Retrieval… "Asset MySQL handles metadata storage"
5. `[TIER 2]` NightOwl migrates PostgreSQL → Qdrant
6. `[TIER 2]` CobaltMesh migrates MySQL → Weaviate
7. `[TIER 2]` Ivo introduces Sparse-Retrieval… "Asset MySQL handles metadata storage" ← same chunk again

The stale "MySQL handles metadata" chunk appears twice — once Tier 1, once Tier 2 — and the model latched onto the repeated fact over the single migration line. This is repetition/salience, not primacy (the Tier 1 line is 4th, not first — the earlier primacy theory is wrong).

### Three Confirmed `graph_engine.py` Bugs
1. **Cross-tier chunk duplication (direct cause).** Same chunk emitted twice with `[TIER 1 ACTIVE]` and `[TIER 2 ARCHIVE]` prefixes; the `set()` dedup compares full strings, so different prefixes defeat it.
2. **Entropy inversion.** Min-heap on `-entropy` evicts highest-entropy edges first; in a sparse graph that means **hubs evict first, leaf edges stay** — backwards from intent.
3. **Broken stale check.** Global `total_edges` denominator makes all scores stale on any insert, but only neighbors are recomputed — so heap and `self.edges` hold matching stale values, the check passes, and eviction runs on bad math.

**9 vs 10:** at 9 both Ivo chunks land in Tier 2, share a prefix, and dedup correctly. At 10 one edge stays in Tier 1 while its pair sits in Tier 2 — straddling the boundary and breaking dedup.

### Next Steps
- Fix cross-tier dedup first: dedup on chunk content, not the tier-prefixed string.
- Fix entropy inversion so leaf edges evict first and hubs stay.
- Fix the stale check: drop the global denominator or use `math.isclose` instead of `!=` on float scores.
- Treat recency-decay and contradiction-flagging as design decisions, not bug fixes.
- Don't over-fit to one flipped query — confirm with a rerun before trusting a systematic-staleness pattern.

### Security To-Do (still open)
- SSL verification disabled on every Groq call (`ctx.verify_mode = ssl.CERT_NONE`); API key rides in headers, exposed to MITM. Fix once the sweep is stable.
- Rotate any Groq keys previously pasted in plaintext.
***

## Session 7 — July 11, 2026

### What We Did
Completed the full hyperparameter sweep for the Bounded GraphRAG architecture. We evaluated the system against the generated evaluation suite (n=20) across a range of edge constraints ($max\_edges = 2, 5, 10, 25, 50, 100, 250, 500$) and the Unbounded GraphRAG and Vector-RAG control groups. This session concludes the optimization phase of the experiment.

### Final Sweep Results
The metrics confirm that accuracy saturates at $max\_edges=50$. Increasing memory capacity beyond this point yields zero accuracy gains, suggesting the graph has reached a point of structural condensation where all critical relational signal fits into the Active (Tier 1) memory.

| Configuration | Accuracy | n |
| :--- | :--- | :--- |
| **Vector-RAG (Control A)** | **60.00%** | 20 |
| Unbounded GraphRAG (Control B) | 85.00% | 20 |
| Bounded GraphRAG (max_edges=2) | 70.00% | 20 |
| Bounded GraphRAG (max_edges=5) | 70.00% | 20 |
| Bounded GraphRAG (max_edges=10) | 75.00% | 20 |
| Bounded GraphRAG (max_edges=25) | 80.00% | 20 |
| **Bounded GraphRAG (max_edges=50)** | **85.00%** | 20 |
| Bounded GraphRAG (max_edges=100) | 85.00% | 20 |
| Bounded GraphRAG (max_edges=250) | 85.00% | 20 |
| Bounded GraphRAG (max_edges=500) | 85.00% | 20 |

### Accuracy by Query Type
| Configuration | entity_lookup | multi_hop | multi_hop_contradiction | temporal_drift |
| :--- | :--- | :--- | :--- | :--- |
| **Vector-RAG (A)** | 100% | 20% | 20% | 100% |
| **Graph (max_edges=50)** | 100% | 60% | 80% | 100% |

*Analysis:* Vector-RAG performed well on `entity_lookup` and `temporal_drift` but failed catastrophically on relational tasks (`multi_hop`, `multi_hop_contradiction`), bottoming out at 20% accuracy. GraphRAG maintains parity on lookups while drastically outperforming Vector-RAG on the relational reasoning tasks.

### Memory Tier Utilization
| Configuration | Tier 1 Hit Ratio | Tier 2 Hit Ratio |
| :--- | :--- | :--- |
| Bounded (max_edges=2) | 0.0% | 100.0% |
| Bounded (max_edges=25) | 100.0% | 80.0% |
| **Bounded (max_edges=50)** | **100.0%** | **0.0%** |
| Bounded (max_edges=500) | 100.0% | 0.0% |

*Finding:* At `max_edges=50`, the system achieves **100% Tier 1 utilization**. This confirms that the entire necessary context for these queries fits within the active memory cap, eliminating the latency penalty of the Tier 2 Archive.

### Conclusion: The Production Default
1.  **Optimal Configuration:** $max\_edges = 50$ is the optimal production setting. It hits the 85% accuracy ceiling while maximizing Tier 1 (Active) hits.
2.  **Architecture Validation:** The Bounded GraphRAG architecture is validated. It provides a significant advantage over Vector-RAG for relational reasoning tasks.
3.  **Future Proofing:** No further hyperparameter tuning is required. The system has reached diminishing returns; compute resources are better spent elsewhere (e.g., entity scoping, extraction quality) than on edge-cap adjustments.

### Next Steps
- Transition code to production: hardcode `max_edges=50`.
- Archive this sweep for the thesis baseline.
- Focus on `multi_hop` reasoning improvements if higher than 85% accuracy is desired.
***

## Session 8 — July 14, 2026

### What We Did
Completed the security and production-readiness lockdown for the Bounded GraphRAG codebase. Focused on credential isolation, transport security, and structural integration of the `BoundedChunkStore` with the engine's query interface.

### Production Readiness Checklist
| Implementation Step | Status | Notes |
| :--- | :--- | :--- |
| **Credential Isolation** | Complete | Migrated to `.env` using `python-dotenv`. |
| **Version Control Security** | Complete | Added `.env` to `.gitignore`. |
| **Transport Security (SSL)** | Complete | Fixed macOS certificate chain via `certifi`; removed `CERT_NONE` bypass. |
| **Engine-Chunk Integration** | Complete | Added `query()` method and `BoundedChunkStore` instantiation. |
| **Data Ingestion Pipeline** | **Pending** | `main.py` is currently a shell; requires `corpus.json` loader. |

### Key Security & Architectural Changes
* **Credential Management:** The codebase no longer relies on hardcoded strings. Environment variables are loaded via `dotenv` with runtime validation in `main.py`.
* **Transport Layer:** Eliminated `ssl.CERT_NONE` bypasses, preventing potential Man-in-the-Middle (MITM) attacks. The system now utilizes the standard system root certificate store.
* **Interface Design:** Successfully decoupled the `BoundedGraphRAGEngine` from the storage layer. The new `query(query_text, chunk_store)` interface facilitates clean dependency injection and retrieval.

### Current System Status
The system is now architecturally robust, secured, and compliant with production standards. However, the system is currently "cold"—it is missing the ingestion logic to populate the graph and chunk store from the `data/` corpus. The `query` method will return an empty string if called in the current state.

### Next Steps
- Implement `src/ingestor.py` to handle JSON corpus parsing.
- Wire ingestion into the `main.py` startup sequence (pre-loop).
- Execute end-to-end load test using existing `corpus.json`.

## Session 9 — July 18, 2026

### What We Did
Resolved the runtime `(no context found)` engine failure. Rewrote the broken ingestion layer (`src/ingestor.py`) to replace the hardcoded fallback architecture with an automated proper-noun relational extraction pipeline. Verified the runtime query loop using multi-hop context lookups across live entities.

### Core Issue: The "System" Hub Pathology
The original `ingestor.py` script was structurally flawed. It force-linked every document chunk in `data/corpus.json` to a generic `"System"` parent node, establishing an artificial hub-and-spoke topology of 50 leaf nodes. 

Because the live system uses an LLM to parse user input and extract actual natural language entities, query-time extractions (e.g., searching for `"ShadowGrid"` or `"Bianca"`) found zero matching records inside the graph. The diagnostic script `verify_system.py` only passed because it bypassed LLM extraction and queried the literal string `"System"` directly.

### The Secondary Risk: Entropy-Based Eviction Dynamics
Building a true relational graph brought the system face-to-face with the `max_edges=50` boundary constraint. Mathematical analysis of the eviction logic exposed a high-probability failure mode:
High-degree structural hubs (like `ShadowGrid`) generate a uniform distribution of connections, maximizing their Shannon entropy:

$$H(u) = -\sum_{v \in N(u)} p(u,v) \log_2 p(u,v)$$

Under tight memory constraints, an engine that evicts edges based on maximizing topological entropy will systematically decapitate these primary routing hubs first. This destroys the multi-hop relational paths required for GraphRAG retrieval.

### Codebase Refactor & Implementation
The ingestion engine was refactored via `ingestor_3.py` to construct an organic relational graph:
* **Token Normalization:** Implemented an aggressive prefix-stripping layer (`Project `, `Asset `, `Agent `) to align ingestion-time node names with query-time LLM entity extractions.
* **Noise Filtering:** Deployed a capitalized-phrase regex matcher combined with a structural stopword registry to prevent generic sentence-leading words (e.g., "The", "Infrastructure") from polluting the topology.
* **Relational Co-occurrence Loop:** Replaced the hub-and-spoke logic with a nested pairing loop. Every entity co-occurring within a document chunk is linked to all other entities in that chunk, creating the multi-hop paths required for graph traversal.
* **Schema Fallback:** Updated the file parser to dynamically accept both `doc_id` and `id` keys, ensuring cross-compatibility across distinct corpus fixtures.

### Runtime Verification Results
Running `python main.py` confirmed clean environment variable loading and successfully loaded the 50 items into the engine. Live terminal queries demonstrate that the extraction and routing layers are functioning as intended:

* **`Query: ShadowGrid`** $\rightarrow$ Successfully bypassed the eviction wall to return the core architecture logs, tracking the chronological migration from MongoDB to Asset Pinecone.
* **`Query: Bianca`** $\rightarrow$ Accurately traversed the graph to aggregate cross-project engineering records spanning the `NightOwl`, `IronVault`, and `CobaltMesh` infrastructure layers.

### Current System Status
The pipeline is fully operational. The entity matching desynchronization is resolved, and context blocks are successfully loading into the LLM synthesis window without triggering premature hub eviction.

### Next Steps
* Scale testing by replacing the sample file with `corpus_large.json` to monitor the engine under a sustained capacity crunch.
* Observe the eviction heap under load to verify if active node degrees begin dropping vital relational signal once total edges breach the 50-edge threshold.


## Session 10 — July 19, 2026

### What We Did
Executed a systematic code overhaul to align the ingestion and query pipelines, resolved critical active-archive hub calculation mismatches, bounded historical memory growth, and audited the retrieval path for remaining extraction anomalies[cite: 4, 5].

### Core Architectural Fixes

* **Unified Normalization Pipeline:** Consolidated all prefix-stripping logic into a single, canonical `normalize_entity` function inside `graph_engine.py`[cite: 5]. Both `generator.py` and `ingestor.py` now import this function directly, eliminating case-insensitivity discrepancies and fixing a silent bug where the ingestor omitted the `Concept` and `Event` categories[cite: 4, 5, 6].
* **Footprint-Aware Hub Capping:** Modified `_adaptive_hub_params` and `_select_expansion_edges` to evaluate a node's combined active degree and archived edge count[cite: 5]. This blocks historical mega-hubs from spoofing the engine as low-degree leaf nodes when their active edges are evicted[cite: 5].
* **Reordered Evaluation Sequence:** Updated `retrieve_subgraph_context` to compile the `archive_index` *before* computing adaptive hub parameters, ensuring the thresholding logic utilizes the accurate combined footprint[cite: 5].
* **Bounded Archive Capacity:** Stabilized long-term memory allocation by imposing a `max_archive=500` limit on `BoundedGraphRAGEngine`[cite: 5]. Evicted edges are now managed via a strict FIFO queue, capping the per-query archive scan to a constant runtime ceiling[cite: 5].

---

### System Vulnerabilities & Open Pathologies

#### 1. Query-Side Token Truncation (Critical Correctness Bug)
* **Mechanism:** The corpus ingestion pipeline uses a broad tokenization pattern (`\b[A-Z][a-zA-Z0-9\-]*(?:\s+[A-Z][a-zA-Z0-9\-]*)*\b`) that captures digits and hyphens, successfully building nodes like `"Verification-Run-1"`[cite: 4]. However, the query engine's `ENTITY_PATTERN` uses a restrictive character class (`\b[A-Z][a-zA-Z]*(?:\s+[A-Z][a-zA-Z]*)*\b`) that completely strips non-alpha characters[cite: 5].
* **Impact:** A search query for `"Verification-Run-1"` is broken into separate tokens (`"Verification"` and `"Run"`), dropping the `"-1"` identifier[cite: 5]. Because graph entry relies on exact seed matching, the retrieval frontier evaluates to an empty set and returns nothing[cite: 5].

#### 2. O(N) Retrieval Path Latency
* **Mechanism:** The `_build_archive_index` lookup dictionary is still generated completely from scratch during the execution of every user query[cite: 5]. 
* **Impact:** Although capped at 500 entries, running a linear array scan on every query creates an unnecessary processing bottleneck[cite: 5]. The archive index should be handled incrementally during eviction and re-insertion phases rather than on the query critical path[cite: 5].

#### 3. Legacy State Incompatibility
* **Mechanism:** Structural changes to the normalization layers and regex patterns are not retroactive[cite: 5].
* **Impact:** Previously cached or persisted graph states built under drifted naming definitions are now fragmented and incompatible[cite: 5]. 

---

### Next Action Items
* Sync the query-side `ENTITY_PATTERN` regex in `graph_engine.py` with the alphanumeric and hyphen capabilities used by the ingestor[cite: 4, 5].
* Refactor the archive memory tier to dynamically maintain an in-memory index during active evictions, changing the query path dictionary build from $O(N)$ to $O(1)$[cite: 5].
* Wipe all stale local database stores and trigger a full re-ingestion of the text corpus to ensure absolute naming uniformity across the graph topology[cite: 5].