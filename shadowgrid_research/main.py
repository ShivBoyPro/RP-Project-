import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Verify API key
if not os.getenv("GROQ_API_KEY"):
    print("CRITICAL ERROR: GROQ_API_KEY is not set in your .env file.")
    sys.exit(1)

from src.graph_engine import BoundedGraphRAGEngine, BoundedChunkStore
from src.ingestor import ingest_corpus

def main():
    print("Initializing Bounded GraphRAG System...")
    engine = BoundedGraphRAGEngine()
    chunk_store = BoundedChunkStore()

    corpus_path = "data/corpus.json"
    
    # Run the ingestion
    success = ingest_corpus(corpus_path, engine, chunk_store)
    if not success:
        print("Failed to ingest corpus. Exiting.")
        sys.exit(1)

    print("\nSystem ready. Type 'exit' to quit.")

    # Query loop
    while True:
        try:
            user_input = input("\nQuery: ").strip()
            if user_input.lower() in ['exit', 'quit']:
                break
            if not user_input:
                continue
            
            result = engine.query(user_input, chunk_store)
            
            if result:
                print(f"\nResult:\n{result}")
            else:
                print("\n(no context found)")
                
        except KeyboardInterrupt:
            print("\nExiting.")
            break

if __name__ == "__main__":
    main()