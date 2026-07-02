import json
import os

def generate_high_density_suite():
    corpus = []
    
    # Inject 45 pieces of structural background noise to drown out simple embeddings
    for i in range(1, 46):
        corpus.append({
            "id": f"noise_{i:03d}",
            "content": f"Infrastructure log sequence reference alpha-{i*100}. System operations running within normal parameters. Storage array check completed for cluster node {i}."
        })
        
    # Inject the actual tracking timeline documents with heavy overlapping keywords
    timeline = [
        {"id": "doc_001", "content": "Project ShadowGrid core architecture initialization. Initial database infrastructure deployed using standard PostgreSQL schemas. Core engineering contact assigned: Agent Alexander."},
        {"id": "doc_002", "content": "System update for ShadowGrid: Algorithmic development introduces Concept Vector-Embeddings to optimize indexing operations on top of the PostgreSQL tables."},
        {"id": "doc_003", "content": "Internal memo: Project ShadowGrid engineering overhead reassignment. Agent Alexander is officially removed from primary operations. Lead engineer status transferred to Agent Bianca."},
        {"id": "doc_004", "content": "Architecture migration proposal: Due to severe scaling limits with relational schemas, ShadowGrid has initiated a full database migration from PostgreSQL to Asset Qdrant."},
        {"id": "doc_005", "content": "Decommissioning report: Project ShadowGrid has completely shut down the PostgreSQL engine. The system is now operating exclusively on the active Qdrant database stack. Agent Bianca confirmed a scheduled verification run (Verification-Run-1)."}
    ]
    
    corpus.extend(timeline)
    
    # Ensure data directory exists before writing
    os.makedirs("data", exist_ok=True)
    
    with open("data/corpus_large.json", "w") as f:
        json.dump(corpus, f, indent=4)
        
    print(f"[SUCCESS] High-Density Corpus compiled with {len(corpus)} total documents.")

if __name__ == "__main__":
    generate_high_density_suite()