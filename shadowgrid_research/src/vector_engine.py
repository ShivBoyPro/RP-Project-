import os
import json
import numpy as np
from sentence_transformers import SentenceTransformer

class VectorRAGEngine:
    def __init__(self, corpus_dir="data/raw_corpus", model_name="all-MiniLM-L6-v2"):
        self.corpus_dir = corpus_dir
        # Caches the model locally on first download for offline execution
        self.model = SentenceTransformer(model_name)
        self.documents = []
        self.embeddings = []
        self.load_and_index_corpus()

    def load_and_index_corpus(self):
        """Loads raw corpus data and compiles a clean matrix of normalized embeddings."""
        if not os.path.exists(self.corpus_dir):
            raise FileNotFoundError(f"Corpus directory {self.corpus_dir} does not exist. Run generator.py first.")
            
        files = sorted([f for f in os.listdir(self.corpus_dir) if f.endswith(".json")])
        corpus_texts = []
        
        for file in files:
            with open(os.path.join(self.corpus_dir, file), "r") as f:
                doc_data = json.load(f)
                self.documents.append(doc_data)
                corpus_texts.append(doc_data["content"])
                
        if corpus_texts:
            raw_embeddings = self.model.encode(corpus_texts, convert_to_numpy=True)
            # L2 Normalize vectors to make simple dot-product equivalent to cosine similarity
            self.embeddings = raw_embeddings / np.linalg.norm(raw_embeddings, axis=1, keepdims=True)
            
        print(f"[SUCCESS] Control Group Vector Index compiled with {len(self.documents)} nodes.")

    def retrieve(self, query, k=2):
        """Executes a pure nearest-neighbor vector search."""
        if len(self.embeddings) == 0:
            return []
            
        query_emb = self.model.encode([query], convert_to_numpy=True)
        query_emb = query_emb / np.linalg.norm(query_emb, axis=1, keepdims=True)
        
        # Calculate cosine similarities via matrix multiplication
        similarities = np.dot(self.embeddings, query_emb.T).flatten()
        top_indices = np.argsort(similarities)[::-1][:k]
        
        retrieved_contexts = []
        for idx in top_indices:
            retrieved_contexts.append({
                "doc_id": self.documents[idx]["doc_id"],
                "timestamp": self.documents[idx]["timestamp"],
                "content": self.documents[idx]["content"],
                "similarity": float(similarities[idx])
            })
            
        return retrieved_contexts

if __name__ == "__main__":
    # Hard test pass to verify semantic collision
    engine = VectorRAGEngine()
    test_query = "What is the current active database stack utilized by Project ShadowGrid?"
    results = engine.retrieve(test_query, k=2)
    
    print("\n--- Raw Vector Retrieval Traces ---")
    for res in results:
        print(f"[{res['timestamp']}] {res['doc_id']} (Score: {res['similarity']:.4f})")
        print(f"Content: \"{res['content']}\"\n")