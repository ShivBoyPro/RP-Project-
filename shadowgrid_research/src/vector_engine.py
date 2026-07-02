import os
import json
import numpy as np
from sentence_transformers import SentenceTransformer

class VectorRAGEngine:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
        self.corpus_texts = []
        self.embeddings = None
        self.load_and_index_corpus()
        
    def load_and_index_corpus(self):
        corpus_path = "data/corpus_large.json" 
        if not os.path.exists(corpus_path):
            corpus_path = "data/corpus.json"
            
        if not os.path.exists(corpus_path):
            raise FileNotFoundError("Neither data/corpus_large.json nor data/corpus.json was found.")
            
        with open(corpus_path, "r") as f:
            data = json.load(f)
            
        corpus_texts = []
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict) and "content" in item:
                    corpus_texts.append(item["content"])
                elif isinstance(item, list):
                    for sub_item in item:
                        if isinstance(sub_item, dict) and "content" in sub_item:
                            corpus_texts.append(sub_item["content"])
        elif isinstance(data, dict):
            for key, val in data.items():
                if isinstance(val, dict) and "content" in val:
                    corpus_texts.append(val["content"])
                elif isinstance(val, str):
                    corpus_texts.append(val)

        if not corpus_texts:
            raise ValueError(f"Failed to extract text structures from {corpus_path}.")

        self.corpus_texts = corpus_texts
        raw_embeddings = self.model.encode(corpus_texts, convert_to_numpy=True)
        
        # Safe normalization to guard against zero-vectors
        norms = np.linalg.norm(raw_embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1e-9, norms)
        self.embeddings = raw_embeddings / norms

    def retrieve(self, query, k=2):
        if self.embeddings is None or len(self.embeddings) == 0:
            return []
            
        query_emb = self.model.encode([query], convert_to_numpy=True)
        q_norm = np.linalg.norm(query_emb, axis=1, keepdims=True)
        q_norm = np.where(q_norm == 0, 1e-9, q_norm)
        query_emb = query_emb / q_norm
        
        scores = np.dot(self.embeddings, query_emb.T).flatten()
        top_k_indices = np.argsort(scores)[::-1][:k]
        
        return [{"content": self.corpus_texts[idx], "score": float(scores[idx])} for idx in top_k_indices]