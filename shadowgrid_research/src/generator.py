import os
import json
from datetime import datetime, timedelta

class StreamingDatasetGenerator:
    def __init__(self, base_dir="data"):
        self.base_dir = base_dir
        self.corpus_dir = os.path.join(base_dir, "raw_corpus")
        self.eval_file = os.path.join(base_dir, "evaluation_suite.json")
        
        os.makedirs(self.corpus_dir, exist_ok=True)

    def generate_streaming_corpus(self):
        """
        Generates a sequence of chronologically tracked documents simulating 
        corporate project updates with embedded temporal contradictions.
        """
        start_time = datetime(2026, 1, 1, 9, 0, 0)
        
        # Timeline updates mapping specific entity states
        streaming_docs = [
            {
                "timestamp": (start_time).isoformat(),
                "doc_id": "doc_001",
                "content": "Project ShadowGrid initiates development. The core infrastructure is established to utilize a PostgreSQL instance as its primary database architecture. Core engineering contact is Agent Alexander."
            },
            {
                "timestamp": (start_time + timedelta(days=30)).isoformat(),
                "doc_id": "doc_002",
                "content": "Agent Alexander introduces Concept Vector-Embeddings to ShadowGrid. The system utilizes HuggingFace models for embedding computation. Asset PostgreSQL handles metadata storage."
            },
            {
                "timestamp": (start_time + timedelta(days=60)).isoformat(),
                "doc_id": "doc_003",
                "content": "CRITICAL INFRASTRUCTURE UPDATE: Project ShadowGrid completely migrates away from PostgreSQL. Asset PostgreSQL is fully decommissioned. The current database stack for ShadowGrid is now completely transitioned to Asset Qdrant for vector and metadata handling."
            },
            {
                "timestamp": (start_time + timedelta(days=90)).isoformat(),
                "doc_id": "doc_004",
                "content": "Agent Bianca joins Project ShadowGrid to supervise evaluation loops. Bianca updates the pipeline settings, establishing Event Verification-Run-1 on June 1st to validate accuracy."
            },
            {
                "timestamp": (start_time + timedelta(days=120)).isoformat(),
                "doc_id": "doc_005",
                "content": "Personnel reassignment: Agent Alexander is completely removed from Project ShadowGrid and moved to internal infrastructure. Agent Bianca is now the sole primary contact and lead engineer for ShadowGrid development."
            }
        ]

        # Write streaming timeline units out as distinct chronological files
        for doc in streaming_docs:
            file_path = os.path.join(self.corpus_dir, f"{doc['doc_id']}.json")
            with open(file_path, "w") as f:
                json.dump(doc, f, indent=4)
        
        print(f"[SUCCESS] Injected {len(streaming_docs)} streaming documents into {self.corpus_dir}")

    def generate_evaluation_suite(self):
        """
        Generates targeted cross-examination queries designed to break systems
        that cannot handle multi-hop navigation or state changes over time.
        """
        queries = [
            {
                "query_id": "Q_001",
                "type": "temporal_drift",
                "query": "What is the current active database stack utilized by Project ShadowGrid?",
                "ground_truth": "Asset Qdrant. The system was originally on PostgreSQL but completed a migration to Qdrant, fully decommissioning PostgreSQL.",
                "target_entities": ["ShadowGrid", "PostgreSQL", "Qdrant"]
            },
            {
                "query_id": "Q_002",
                "type": "multi_hop_contradiction",
                "query": "Who is the primary engineering contact for the project that utilizes Asset Qdrant?",
                "ground_truth": "Agent Bianca. ShadowGrid uses Qdrant, and while Alexander was the original lead, he was removed and Bianca is now the sole primary contact.",
                "target_entities": ["ShadowGrid", "Qdrant", "Alexander", "Bianca"]
            },
            {
                "query_id": "Q_003",
                "type": "multi_hop",
                "query": "Which algorithmic concept was introduced to the project managed by Agent Bianca?",
                "ground_truth": "Concept Vector-Embeddings. It was introduced to Project ShadowGrid, which is now managed by Agent Bianca.",
                "target_entities": ["Bianca", "ShadowGrid", "Vector-Embeddings"]
            }
        ]

        with open(self.eval_file, "w") as f:
            json.dump(queries, f, indent=4)
        
        print(f"[SUCCESS] Compiled evaluation suite with {len(queries)} gold-standard tasks at {self.eval_file}")

if __name__ == "__main__":
    generator = StreamingDatasetGenerator()
    generator.generate_streaming_corpus()
    generator.generate_evaluation_suite()