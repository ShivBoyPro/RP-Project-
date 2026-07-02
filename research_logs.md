# ShadowGrid Research Log

## Session 1 — June 28, 2026

### What We're Doing
Comparing Vector-RAG vs bounded GraphRAG for local PKM systems. Testing whether aggressively pruning graph edges under a memory cap still preserves retrieval accuracy.

### Thesis
"Topological Entropy-Based Edge Eviction in Local GraphRAG: Stabilizing Retrieval Accuracy (MAP@k) Under Bounded-Memory Streaming Updates Using Low-Parameter (≤8B) Models"

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
