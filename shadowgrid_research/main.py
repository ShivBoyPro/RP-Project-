import os
import sys
from dotenv import load_dotenv
from groq import Groq

# Load environment variables
load_dotenv()

# Verify API key
groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    print("CRITICAL ERROR: GROQ_API_KEY is not set in your .env file.")
    sys.exit(1)

# Initialize Groq client
client = Groq(api_key=groq_api_key)

from src.graph_engine import BoundedGraphRAGEngine, BoundedChunkStore
from src.ingestor import ingest_corpus

# System prompt to enforce strict context grounding and zero hallucination
SYSTEM_PROMPT = """You are an accurate, objective assistant. Answer the user's question using ONLY the facts provided in the Context below. 

Rules:
1. Do not use outside knowledge or assume facts not present in the Context.
2. If the Context does not contain enough information to fully answer the question, state what is known from the context and explicitly note what is missing.
3. Keep answers concise, factual, and direct."""


def synthesize_answer(query: str, context: str, client: Groq) -> str:
    """Passes retrieved graph context and query to Groq LLM for answer synthesis."""
    user_prompt = f"Context:\n{context}\n\nQuestion: {query}"

    try:
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
            max_tokens=500,
        )
        return completion.choices[0].message.content
    except Exception as e:
        return f"LLM Generation Error: {e}"


def main():
    print("Initializing Bounded GraphRAG System...")
    engine = BoundedGraphRAGEngine()
    chunk_store = BoundedChunkStore()

    corpus_path = "data/corpus_large.json"

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
            if user_input.lower() in ["exit", "quit"]:
                break
            if not user_input:
                continue

            # Step 1: Subgraph retrieval
            context = engine.query(user_input, chunk_store)

            # Step 2: Answer synthesis
            if context:
                print("\n[Retrieved Context]")
                print(context)
                print("\n" + "=" * 40)
                print("[LLM Response]")
                answer = synthesize_answer(user_input, context, client)
                print(answer)
                print("=" * 40)
            else:
                print("\n(no context found — skipping LLM call)")

        except KeyboardInterrupt:
            print("\nExiting.")
            break


if __name__ == "__main__":
    main()