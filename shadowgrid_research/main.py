import os
import sys
from dotenv import load_dotenv
from src.graph_engine import BoundedGraphRAGEngine, BoundedChunkStore

# Load credentials from .env
load_dotenv()


def main():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("Error: GROQ_API_KEY not found in .env.")
        sys.exit(1)

    # Production Initialization
    # Ensure graph_engine.py defaults max_edges to 50
    engine = BoundedGraphRAGEngine()
    chunk_store = BoundedChunkStore()

    # KNOWN GAP: ingestion is not wired up. Without a call that populates
    # the graph via engine.insert_edge(...) and chunk_store.add_extraction(...),
    # the graph and chunk store stay empty for the life of the process, and
    # every query() call below will return "" — not an error, just no
    # context. This is a silent-failure mode, not a crash, so it's easy to
    # miss in testing. Wire up real ingestion before treating this as
    # production-ready.
    #
    # engine.ingest("data/clean_corpus.json", chunk_store)

    # Core Production Loop
    try:
        while True:
            query = input("\nQuery: ")
            if query.lower() in ["exit", "quit"]:
                break

            try:
                response = engine.query(query, chunk_store)
            except Exception as e:
                print(f"\nError while processing query: {e}")
                continue

            if not response:
                print("\nResponse: (no context found for this query)")
            else:
                print(f"\nResponse: {response}")

    except KeyboardInterrupt:
        print("\nShutting down.")


if __name__ == "__main__":
    main()