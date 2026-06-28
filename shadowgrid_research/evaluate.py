import json
import os
import urllib.request
import urllib.error
from src.vector_engine import VectorRAGEngine
from src.graph_engine import BoundedGraphRAGEngine

class ResearchOrchestrator:
    def __init__(self, eval_suite_path="data/evaluation_suite.json", ollama_model="qwen2.5:7b-instruct"):
        self.eval_suite_path = eval_suite_path
        self.ollama_model = ollama_model
        
        # Initialize both control and experimental engines
        self.vector_engine = VectorRAGEngine()
        self.graph_engine = BoundedGraphRAGEngine(max_edges=4)
        self.graph_engine.ingest_streaming_data()
        
        self.queries = self.load_evaluation_suite()

    def load_evaluation_suite(self):
        with open(self.eval_suite_path, "r") as f:
            return json.load(f)

    def query_local_llm(self, prompt):
        """Executes a direct, synchronous HTTP request to the local Ollama API."""
        url = "http://localhost:11434/api/generate"
        data = {
            "model": self.ollama_model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.0} # Absolute determinism for scientific benchmarking
        }
        
        req = urllib.request.Request(
            url, 
            data=json.dumps(data).encode("utf-8"), 
            headers={"Content-Type": "application/json"}
        )
        
        try:
            with urllib.request.urlopen(req) as response:
                res_data = json.loads(response.read().decode("utf-8"))
                return res_data.get("response", "").strip()
        except urllib.error.URLError as e:
            return f"[OLLAMA ERROR] Ensure Ollama is running locally. Dev trace: {e}"

    def run_benchmark(self):
        print("\n" + "="*80)
        print("STARTING HEAD-TO-HEAD RETRIEVAL BENCHMARK")
        print("="*80 + "\n")
        
        for q in self.queries:
            print(f"Query ID: {q['query_id']} | Type: {q['type']}")
            print(f"Prompt:   \"{q['query']}\"")
            print(f"Expected: \"{q['ground_truth']}\"\n")
            
            # 1. Execute Vector-RAG Path
            vector_res = self.vector_engine.retrieve(q['query'], k=2)
            vector_context = "\n".join([doc['content'] for doc in vector_res])
            
            vector_prompt = f"Context:\n{vector_context}\n\nQuestion: {q['query']}\nAnswer code-accurately and concisely based strictly on the context:"
            vector_llm_output = self.query_local_llm(vector_prompt)
            
            # 2. Execute Bounded GraphRAG Path
            graph_context = self.graph_engine.retrieve_subgraph(q['target_entities'])
            
            graph_prompt = f"Context:\n{graph_context}\n\nQuestion: {q['query']}\nAnswer code-accurately and concisely based strictly on the context:"
            graph_llm_output = self.query_local_llm(graph_prompt)
            
            # Display Comparative Performance Metrics
            print(f"[-] Vector-RAG Response:\n    {vector_llm_output}")
            print(f"[+] GraphRAG Response:\n    {graph_llm_output}")
            print("-" * 80)

if __name__ == "__main__":
    # Fallback to standard llama3 or llama3:8b if qwen2.5 is not downloaded
    orchestrator = ResearchOrchestrator(ollama_model="llama3")
    orchestrator.run_benchmark()