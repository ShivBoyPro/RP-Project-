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
