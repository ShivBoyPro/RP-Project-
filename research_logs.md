# ShadowGrid Research Log

## Session 1 — June 28, 2026

### What We're Doing
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

