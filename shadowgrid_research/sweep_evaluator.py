import json
import os
import re
import time
import urllib.request
import urllib.error
import ssl
from src.vector_engine import VectorRAGEngine
from src.graph_engine import BoundedGraphRAGEngine


class SweepEvaluator:
    def __init__(self, eval_suite_path="data/evaluation_suite.json", cloud_model="llama-3.1-8b-instant"):
        self.eval_suite_path = eval_suite_path
        self.cloud_model = cloud_model
        self.api_key = os.environ.get("GROQ_API_KEY")

        if not self.api_key:
            raise ValueError("CRITICAL FAILURE: GROQ_API_KEY environment variable is not set. Get one at console.groq.com")

        self.queries = self.load_evaluation_suite()

    def load_evaluation_suite(self):
        if not os.path.exists(self.eval_suite_path):
            os.makedirs(os.path.dirname(self.eval_suite_path), exist_ok=True)
            mock_suite = [
                {"query": "Who was the initial engineer for ShadowGrid?", "ground_truth": "Agent Alexander", "target_entities": ["ShadowGrid", "Alexander"]},
                {"query": "What database platform did ShadowGrid migrate to?", "ground_truth": "Qdrant", "target_entities": ["ShadowGrid", "Qdrant"]}
            ]
            with open(self.eval_suite_path, "w") as f:
                json.dump(mock_suite, f, indent=4)
        with open(self.eval_suite_path, "r") as f:
            return json.load(f)

    def query_cloud_llm(self, prompt, max_retries=5):
        url = "https://api.groq.com/openai/v1/chat/completions"
        data = {
            "model": self.cloud_model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0
        }
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

        for attempt in range(max_retries):
            req = urllib.request.Request(url, data=payload, headers=headers)
            try:
                with urllib.request.urlopen(req, context=ctx) as response:
                    res = json.loads(response.read().decode("utf-8"))
                    return res['choices'][0]['message']['content'].strip()

            except urllib.error.HTTPError as e:
                body = e.read().decode("utf-8", errors="replace")
                last_error = f"ERROR {e.code}: {body}"

                if e.code == 429:
                    match = re.search(r"try again in ([\d.]+)s", body)
                    wait = float(match.group(1)) + 0.5 if match else (2 ** attempt)
                    print(f"[RATE LIMIT] 429 received, backing off {wait:.2f}s (attempt {attempt + 1}/{max_retries})")
                    time.sleep(wait)
                    continue

                return last_error

            except Exception as e:
                return f"ERROR: {e}"

        return f"ERROR: max retries exceeded, last failure -> {last_error}"

    def verify_accuracy_with_judge(self, query, ground_truth, model_output):
        if model_output.startswith("ERROR"):
            print(f"\n--- RAG OUTPUT CALL FAILED (excluded from scoring) ---")
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

        print(f"\n--- JUDGE RAW THOUGHTS ---")
        print(raw_response)
        print(f"--------------------------\n")

        if raw_response.startswith("ERROR"):
            return None

        lines = raw_response.split("\n")
        for line in reversed(lines):
            if "GRADE:" in line:
                if "1" in line and "0" not in line:
                    return 1
                if "0" in line:
                    return 0

        print("[WARN] No GRADE line found in judge response - excluding from scoring")
        return None

    @staticmethod
    def _score_summary(scores):
        valid = [s for s in scores if s is not None]
        excluded = len(scores) - len(valid)
        accuracy = (sum(valid) / len(valid) * 100) if valid else float("nan")
        return accuracy, excluded, len(valid)

    def evaluate_vector_baseline(self):
        vector_engine = VectorRAGEngine()
        scores = []

        print("\n=== STARTING VECTOR BASELINE EVALUATION ===")
        for q in self.queries:
            res = vector_engine.retrieve(q['query'], k=2)
            context = "\n".join([doc['content'] for doc in res])
            
            print(f"\n[DIAGNOSTIC] Query: {q['query']}")
            print(f"--- RAW VECTOR CONTEXT SURFACE ---")
            print(context if context.strip() else "[EMPTY CONTEXT]")
            print(f"----------------------------------")
            
            # Change this line in both evaluation methods:
            prompt = f"Context:\n{context}\n\nQuestion: {q['query']}\nAnswer thoroughly, including all relevant background details, project names, and migration history mentioned in the context."
            output = self.query_cloud_llm(prompt)
            print(f"[LLM OUTPUT]: {output}")
            
            score = self.verify_accuracy_with_judge(q['query'], q['ground_truth'], output)
            scores.append(score)

        return self._score_summary(scores)

    def evaluate_graph_engine(self, max_edges):
        graph_engine = BoundedGraphRAGEngine(max_edges=max_edges)
        graph_engine.ingest_streaming_data()
        scores = []

        print(f"\n=== STARTING GRAPH RAG EVALUATION (max_edges={max_edges}) ===")
        for q in self.queries:
            context = graph_engine.retrieve_subgraph(q['target_entities'])
            
            print(f"\n[DIAGNOSTIC] Query: {q['query']}")
            print(f"--- RAW GRAPH CONTEXT SURFACE ---")
            print(context if context.strip() else "[EMPTY CONTEXT]")
            print(f"---------------------------------")
            
            # Change this line in both evaluation methods:
            prompt = f"Context:\n{context}\n\nQuestion: {q['query']}\nAnswer thoroughly, including all relevant background details, project names, and migration history mentioned in the context."
            output = self.query_cloud_llm(prompt)
            print(f"[LLM OUTPUT]: {output}")
            
            score = self.verify_accuracy_with_judge(q['query'], q['ground_truth'], output)
            scores.append(score)

        return self._score_summary(scores)

    def execute_sweep(self):
        results = {}

        print("[RUNNING] Evaluating Control Group A: Baseline Vector-RAG...")
        acc, excluded, n = self.evaluate_vector_baseline()
        results["Vector-RAG (Control A)"] = (acc, excluded, n)

        print("[RUNNING] Evaluating Control Group B: Unbounded GraphRAG (max_edges = inf)...")
        acc, excluded, n = self.evaluate_graph_engine(max_edges=float('inf'))
        results["Unbounded GraphRAG (Control B)"] = (acc, excluded, n)

        print("[RUNNING] Executing Parameter Sweep for Bounded GraphRAG...")
        for edges in range(2, 11):
            acc, excluded, n = self.evaluate_graph_engine(max_edges=edges)
            results[f"Bounded GraphRAG (max_edges={edges})"] = (acc, excluded, n)
            excl_note = f" (excluded {excluded} invalid data points)" if excluded else ""
            print(f" -> Configuration max_edges={edges} | Accuracy: {acc:.2f}% over {n} valid queries{excl_note}")

        print("\n" + "=" * 65)
        print("FINAL HYPERPARAMETER SWEEP METRICS")
        print("=" * 65)
        for config, (acc, excluded, n) in results.items():
            acc_str = f"{acc:.2f}%" if n > 0 else "N/A (no valid data)"
            excl_note = f"  [excluded {excluded}]" if excluded else ""
            print(f"{config:<35}: {acc_str:<20} n={n}{excl_note}")


if __name__ == "__main__":
    evaluator = SweepEvaluator()
    evaluator.execute_sweep()