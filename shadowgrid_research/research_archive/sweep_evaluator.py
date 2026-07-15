import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
from vector_engine import VectorRAGEngine
from graph_engine import BoundedGraphRAGEngine, BoundedChunkStore
import hashlib
import json
import re
import time
import urllib.request
import urllib.error
import ssl
import os
from dotenv import load_dotenv

# Load the keys out of the local .env file
load_dotenv()

# Safely fetch the variable from memory
api_key = os.getenv("GROQ_API_KEY")
ENTITY_PREFIX_PATTERN = re.compile(r"^(Project|Asset|Agent|Concept|Event)\s+", re.IGNORECASE)


def normalize_entity(name):
    """Strip known operational prefixes and surrounding whitespace so
    variant surface forms of the same entity collapse onto one node."""
    if not isinstance(name, str):
        return name
    return ENTITY_PREFIX_PATTERN.sub("", name).strip()


class SweepEvaluator:
    def __init__(self, eval_suite_path="data/evaluation_suite.json",
                 corpus_path="data/corpus.json",
                 cloud_model="llama-3.1-8b-instant"):
        self.eval_suite_path = eval_suite_path
        self.corpus_path = corpus_path
        self.cloud_model = cloud_model
        self.api_key = os.environ.get("GROQ_API_KEY")

        if not self.api_key:
            raise ValueError("CRITICAL FAILURE: GROQ_API_KEY environment variable is not set. Get one at console.groq.com")

        self.queries = self.load_evaluation_suite()
        self.corpus = self.load_corpus()
        # Keyed by sha256(doc content) -> list of (src, tgt) relation tuples.
        # Populated once per document on first extraction and reused across
        # every configuration in the sweep - see get_relations_for_doc.
        self.relation_cache = {}

    def load_evaluation_suite(self):
        if not os.path.exists(self.eval_suite_path):
            os.makedirs(os.path.dirname(self.eval_suite_path), exist_ok=True)
            mock_suite = [
                {"query_id": "mock_001", "type": "entity_lookup", "query": "Who was the initial engineer for ShadowGrid?", "ground_truth": "Agent Alexander", "target_entities": ["ShadowGrid", "Alexander"]},
                {"query_id": "mock_002", "type": "entity_lookup", "query": "What database platform did ShadowGrid migrate to?", "ground_truth": "Qdrant", "target_entities": ["ShadowGrid", "Qdrant"]}
            ]
            with open(self.eval_suite_path, "w") as f:
                json.dump(mock_suite, f, indent=4)
        with open(self.eval_suite_path, "r") as f:
            return json.load(f)

    def load_corpus(self):
        if not os.path.exists(self.corpus_path):
            raise FileNotFoundError(f"Corpus not found at {self.corpus_path}")
        with open(self.corpus_path, "r") as f:
            return json.load(f)

    def extract_relations_streaming(self, chunk_text, max_retries=5):
        """
        LLM-based relation extraction. Prompt/schema is identical to
        generator.py's extract_relations_streaming so ingestion here
        produces the same kind of edges the corpus was designed around,
        rather than a deterministic stand-in built from the eval suite's
        own target_entities (which let the graph "know" the answer to a
        query before any extraction ever ran).
        """
        prompt = f"""
Analyze the text and extract direct relationships between technical entities.
Output ONLY a JSON object with a single key "relations" containing an array of pairs.
Format: {{"relations": [["EntityA", "EntityB"]]}}
Do not invent relationships. Ignore generic nouns.

Strip operational prefixes from entity names before returning them: do not
include the words "Project", "Asset", "Agent", "Concept", or "Event" as part
of an entity name. For example, return "ShadowGrid" instead of "Project
ShadowGrid", and "Alexander" instead of "Agent Alexander".

Text: {chunk_text}
"""
        response = self.query_cloud_llm(prompt, max_retries=max_retries, json_mode=True)

        if response.startswith("ERROR"):
            print(f"[EXTRACTION DROPPED] LLM call failed: {response}")
            return []

        try:
            parsed = json.loads(response)
        except json.JSONDecodeError as e:
            print(f"[EXTRACTION DROPPED] Non-JSON response ({e}): {response[:200]}")
            return []

        valid_relations = []
        for pair in parsed.get("relations", []):
            if isinstance(pair, list) and len(pair) == 2:
                src, tgt = normalize_entity(pair[0]), normalize_entity(pair[1])
                if src and tgt:
                    valid_relations.append((src, tgt))
        return valid_relations

    def get_relations_for_doc(self, text, max_retries=5):
        """
        Content-hash-keyed cache in front of extract_relations_streaming.
        Without this, execute_sweep would re-run extraction on every one of
        ~13 configurations (11 of which call ingest_corpus), i.e.
        len(corpus) x 11 LLM calls for identical input text. Caching by
        doc-content hash means each document is extracted exactly once for
        the whole sweep: it cuts API cost roughly 11x, and - just as
        important - it guarantees every configuration ingests the exact
        same graph topology, so accuracy differences across max_edges
        reflect the bounding/retrieval logic being swept, not extraction
        nondeterminism or a dropped call in one run and not another.
        """
        doc_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if doc_hash in self.relation_cache:
            return self.relation_cache[doc_hash]

        relations = self.extract_relations_streaming(text, max_retries=max_retries)
        self.relation_cache[doc_hash] = relations
        return relations

    def ingest_corpus(self, graph_engine, chunk_store):
        zero_relation_docs = []
        for doc in self.corpus:
            text = doc.get("content", "")
            if not text:
                continue
            relations = self.get_relations_for_doc(text)
            if not relations:
                zero_relation_docs.append(doc.get("doc_id", "?"))
            for src, tgt in relations:
                graph_engine.insert_edge(src, tgt)
                chunk_store.add_extraction(src, tgt, text)

        if zero_relation_docs:
            print(f"[EXTRACTION SUMMARY] {len(zero_relation_docs)}/{len(self.corpus)} "
                  f"docs produced zero relations: {zero_relation_docs}")

    def query_cloud_llm(self, prompt, max_retries=5, json_mode=False):
        url = "https://api.groq.com/openai/v1/chat/completions"
        data = {
            "model": self.cloud_model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0
        }
        if json_mode:
            data["response_format"] = {"type": "json_object"}
        payload = json.dumps(data).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }

        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE

        last_error = None
        rate_limit_backoffs = 0
        max_rate_limit_backoffs = max_retries * 4  # safety ceiling so persistent 429s can't loop forever
        standard_attempts = 0
        RETRYABLE_HTTP_CODES = {500, 502, 503, 504}  # transient gateway/server errors, not client-error 4xx

        while True:
            req = urllib.request.Request(url, data=payload, headers=headers)
            try:
                with urllib.request.urlopen(req, context=ctx, timeout=30) as response:
                    res = json.loads(response.read().decode("utf-8"))
                    return res['choices'][0]['message']['content'].strip()

            except urllib.error.HTTPError as e:
                body = e.read().decode("utf-8", errors="replace")
                last_error = f"ERROR {e.code}: {body}"

                if e.code == 429:
                    if rate_limit_backoffs >= max_rate_limit_backoffs:
                        return f"ERROR: exceeded rate-limit backoff ceiling, last failure -> {last_error}"
                    match = re.search(r"try again in ([\d.]+)s", body)
                    wait = float(match.group(1)) + 0.5 if match else (2 ** rate_limit_backoffs)
                    rate_limit_backoffs += 1
                    print(f"[RATE LIMIT] 429 received, backing off {wait:.2f}s (backoff #{rate_limit_backoffs})")
                    time.sleep(wait)
                    continue  # rate-limit backoffs no longer eat into max_retries

                if e.code in RETRYABLE_HTTP_CODES:
                    standard_attempts += 1
                    if standard_attempts > max_retries:
                        return f"ERROR: exhausted {max_retries} retries on transient HTTP errors. Last failure -> {last_error}"
                    print(f"[TRANSIENT HTTP {e.code}] retrying (attempt {standard_attempts}/{max_retries})")
                    time.sleep(1 * standard_attempts)
                    continue

                # Non-retryable client error (400/401/403/404/etc) - retrying won't help
                return last_error

            except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as e:
                # Covers socket drops, DNS blips, connect/read timeouts
                last_error = f"ERROR: {e}"
                standard_attempts += 1
                if standard_attempts > max_retries:
                    return f"ERROR: exhausted {max_retries} retries on network errors. Last failure -> {last_error}"
                print(f"[NETWORK ERROR] {e} - retrying (attempt {standard_attempts}/{max_retries})")
                time.sleep(1 * standard_attempts)
                continue

            except Exception as e:
                # Unexpected/non-transient failure (e.g. JSON decode of a malformed 200) - don't retry blindly
                return f"ERROR: {e}"

    def verify_accuracy_with_judge(self, query, ground_truth, model_output, query_id=None):
        if model_output.startswith("ERROR"):
            print(f"\n--- RAG OUTPUT CALL FAILED (excluded from scoring) [query_id={query_id}] ---")
            print(model_output)
            print(f"--------------------------\n")
            return None

        judge_prompt = f"""[SYSTEM]
You are a factual accuracy validator. Your task is to compare the RAG System Output against the Ground Truth.

Grading Rules:
1. If the RAG System Output contains the correct factual answer to the Question, GRADE it 1.
2. Do not penalize for brevity unless the missing information is required to make the answer factually correct.
3. Be objective.

Query: "{query}"
Ground Truth: "{ground_truth}"
RAG System Output: "{model_output}"

Provide your evaluation in this exact format:
REASONING: <reasoning>
GRADE: <1 or 0>
"""
        raw_response = self.query_cloud_llm(judge_prompt).strip()

        print(f"\n--- JUDGE RAW THOUGHTS [query_id={query_id}] ---")
        print(raw_response)
        print(f"--------------------------\n")

        if raw_response.startswith("ERROR"):
            print(f"[EXCLUDED] Judge call failed for query_id={query_id}")
            return None

        lines = raw_response.split("\n")
        for line in reversed(lines):
            match = re.search(r"GRADE:\s*([01])(?:\.0+)?\b", line)
            if match:
                return int(match.group(1))

        print(f"[WARN] No GRADE line found in judge response for query_id={query_id} - excluding from scoring")
        return None

    @staticmethod
    def _parse_tier_usage(context):
        """
        Parses the [TIER 1 ACTIVE] / [TIER 2 ARCHIVE] telemetry tags that
        BoundedGraphRAGEngine.retrieve_subgraph_context prefixes onto each
        matched text chunk. A "hit" means at least one chunk from that tier
        made it into the context handed to the LLM for this query.
        """
        tier1_hit = "[TIER 1 ACTIVE]" in context
        tier2_hit = "[TIER 2 ARCHIVE]" in context
        return tier1_hit, tier2_hit

    @staticmethod
    def _score_summary(records):
        """
        records: list of dicts {query_id, type, score, tier1_hit, tier2_hit}
        where score is 1, 0, or None (excluded - infra/parsing failure, not
        a wrong answer) and tier1_hit/tier2_hit are None for non-graph
        (Vector-RAG) evaluations.

        Returns overall accuracy/excluded/n, a per-query-type breakdown,
        and Tier 1 / Tier 2 hit ratios computed over valid (non-excluded)
        queries - matching the same denominator convention as accuracy.
        """
        valid = [r for r in records if r["score"] is not None]
        excluded = len(records) - len(valid)
        accuracy = (sum(r["score"] for r in valid) / len(valid) * 100) if valid else float("nan")

        by_type = {}
        for r in records:
            by_type.setdefault(r["type"], []).append(r["score"])

        type_breakdown = {}
        for qtype, scores in by_type.items():
            valid_t = [s for s in scores if s is not None]
            excl_t = len(scores) - len(valid_t)
            acc_t = (sum(valid_t) / len(valid_t) * 100) if valid_t else float("nan")
            type_breakdown[qtype] = (acc_t, excl_t, len(valid_t))

        tier_records = [r for r in valid if r.get("tier1_hit") is not None]
        if tier_records:
            tier1_ratio = sum(1 for r in tier_records if r["tier1_hit"]) / len(tier_records) * 100
            tier2_ratio = sum(1 for r in tier_records if r["tier2_hit"]) / len(tier_records) * 100
        else:
            tier1_ratio = None
            tier2_ratio = None

        return accuracy, excluded, len(valid), type_breakdown, tier1_ratio, tier2_ratio

    def evaluate_vector_baseline(self):
        vector_engine = VectorRAGEngine()
        # TODO(Shiv): confirm this matches the real VectorRAGEngine API - src/vector_engine.py
        # wasn't in this upload, so `ingest` is a best guess based on naming conventions
        # used elsewhere in this file (ingest_corpus, insert_edge, add_extraction).
        # Without an explicit push here, retrieve() below runs against an empty index and
        # Control Group A silently returns empty context for every query.
        records = []

        print("\n=== STARTING VECTOR BASELINE EVALUATION ===")
        for q in self.queries:
            qid = q.get("query_id", q["query"])
            qtype = q.get("type", "unknown")

            res = vector_engine.retrieve(q['query'], k=2)
            context = "\n".join([doc['content'] for doc in res])

            print(f"\n[DIAGNOSTIC] Query: {q['query']} [id={qid}, type={qtype}]")
            print(f"--- RAW VECTOR CONTEXT SURFACE ---")
            print(context if context.strip() else "[EMPTY CONTEXT]")
            print(f"----------------------------------")

            prompt = f"Context:\n{context}\n\nQuestion: {q['query']}\nAnswer thoroughly, including all relevant background details, project names, and migration history mentioned in the context."
            output = self.query_cloud_llm(prompt)
            print(f"[LLM OUTPUT]: {output}")

            score = self.verify_accuracy_with_judge(q['query'], q['ground_truth'], output, query_id=qid)
            # Vector-RAG has no tiered memory - tier fields stay None so
            # they're excluded from tier-ratio calculations downstream.
            records.append({"query_id": qid, "type": qtype, "score": score,
                             "tier1_hit": None, "tier2_hit": None})

        return self._score_summary(records)

    def evaluate_graph_engine(self, max_edges):
        graph_engine = BoundedGraphRAGEngine(max_edges=max_edges)
        chunk_store = BoundedChunkStore()
        self.ingest_corpus(graph_engine, chunk_store)

        print(f"[SUCCESS] Streaming ingestion complete. Active Graph State: "
              f"Nodes={len(graph_engine.node_degrees)}, Edges={len(graph_engine.edges)}, "
              f"Archived Edges={len(graph_engine.archive)}")

        archived_pairs = [(a["src"], a["tgt"]) for a in graph_engine.archive]
        unique_archived_pairs = set(archived_pairs)
        overlapping_with_active = sum(
            1 for (s, t) in unique_archived_pairs
            if (s, t) in graph_engine.edges or (t, s) in graph_engine.edges
        )
        print(f"[ARCHIVE DIAGNOSTIC] {len(archived_pairs)} raw records, "
              f"{len(unique_archived_pairs)} unique pairs, "
              f"{overlapping_with_active} still overlapping with active Tier 1 "
              f"(should be 0 after the reinsert-purge fix)")

        records = []

        print(f"\n=== STARTING GRAPH RAG EVALUATION (max_edges={max_edges}) ===")
        for q in self.queries:
            qid = q.get("query_id", q["query"])
            qtype = q.get("type", "unknown")

            context = graph_engine.retrieve_subgraph_context(q['target_entities'], chunk_store)
            tier1_hit, tier2_hit = self._parse_tier_usage(context)

            print(f"\n[DIAGNOSTIC] Query: {q['query']} [id={qid}, type={qtype}]")
            print(f"--- RAW GRAPH CONTEXT SURFACE (Tier1={tier1_hit}, Tier2={tier2_hit}) ---")
            print(context if context.strip() else "[EMPTY CONTEXT]")
            print(f"---------------------------------")

            prompt = f"Context:\n{context}\n\nQuestion: {q['query']}\nAnswer thoroughly, including all relevant background details, project names, and migration history mentioned in the context."
            output = self.query_cloud_llm(prompt)
            print(f"[LLM OUTPUT]: {output}")

            score = self.verify_accuracy_with_judge(q['query'], q['ground_truth'], output, query_id=qid)
            records.append({"query_id": qid, "type": qtype, "score": score,
                             "tier1_hit": tier1_hit, "tier2_hit": tier2_hit})

        return self._score_summary(records)

    def execute_sweep(self):
        results = {}

        print("[RUNNING] Evaluating Control Group A: Baseline Vector-RAG...")
        acc, excluded, n, breakdown, t1, t2 = self.evaluate_vector_baseline()
        results["Vector-RAG (Control A)"] = (acc, excluded, n, breakdown, t1, t2)

        print("[RUNNING] Evaluating Control Group B: Unbounded GraphRAG (max_edges = inf)...")
        acc, excluded, n, breakdown, t1, t2 = self.evaluate_graph_engine(max_edges=float('inf'))
        results["Unbounded GraphRAG (Control B)"] = (acc, excluded, n, breakdown, t1, t2)

        print("[RUNNING] Executing Parameter Sweep for Bounded GraphRAG...")
        # Wide, log-scaled steps: at max_edges in the 2-10 range, BoundedGraphRAGEngine
        # evicts on almost every sentence ingested, so accuracy is flat/noisy and mostly
        # exercises Tier 2 archiving rather than the bounding tradeoff itself. Scaling up
        # toward the size of a well-connected doc sub-component (50-500 edges) is where
        # the eviction-vs-accuracy curve actually differentiates from the unbounded control.
        sweep_edges = [2, 5, 10, 25, 50, 100, 250, 500]
        for edges in sweep_edges:
            acc, excluded, n, breakdown, t1, t2 = self.evaluate_graph_engine(max_edges=edges)
            results[f"Bounded GraphRAG (max_edges={edges})"] = (acc, excluded, n, breakdown, t1, t2)
            excl_note = f" (excluded {excluded} invalid data points)" if excluded else ""
            tier_note = f" | Tier1 Hit: {t1:.1f}% | Tier2 Hit: {t2:.1f}%" if t1 is not None else ""
            print(f" -> Configuration max_edges={edges} | Accuracy: {acc:.2f}% over {n} valid queries{excl_note}{tier_note}")

        print("\n" + "=" * 65)
        print("FINAL HYPERPARAMETER SWEEP METRICS")
        print("=" * 65)
        for config, (acc, excluded, n, breakdown, t1, t2) in results.items():
            acc_str = f"{acc:.2f}%" if n > 0 else "N/A (no valid data)"
            excl_note = f"  [excluded {excluded}]" if excluded else ""
            print(f"{config:<35}: {acc_str:<20} n={n}{excl_note}")

        print("\n" + "=" * 65)
        print("ACCURACY BY QUERY TYPE")
        print("=" * 65)
        all_types = sorted({t for r in results.values() for t in r[3].keys()})
        header = f"{'Configuration':<35}" + "".join(f"{t:<24}" for t in all_types)
        print(header)
        for config, (acc, excluded, n, breakdown, t1, t2) in results.items():
            row = f"{config:<35}"
            for t in all_types:
                if t in breakdown:
                    t_acc, t_excl, t_n = breakdown[t]
                    cell = f"{t_acc:.0f}% (n={t_n})" if t_n > 0 else "N/A"
                else:
                    cell = "-"
                row += f"{cell:<24}"
            print(row)

        # Tier utilization is only meaningful for the graph configurations;
        # Vector-RAG has no tiered memory and is intentionally omitted here
        # rather than printed as a misleading "0%".
        print("\n" + "=" * 65)
        print("MEMORY TIER UTILIZATION (Tier 1 Active vs Tier 2 Archive)")
        print("=" * 65)
        print(f"{'Configuration':<35}{'Tier 1 Hit Ratio':<20}{'Tier 2 Hit Ratio':<20}")
        for config, (acc, excluded, n, breakdown, t1, t2) in results.items():
            if t1 is None:
                continue
            print(f"{config:<35}{t1:>6.1f}%{'':<13}{t2:>6.1f}%")


if __name__ == "__main__":
    evaluator = SweepEvaluator()
    evaluator.execute_sweep()