import json
from src.graph_engine import BoundedGraphRAGEngine, BoundedChunkStore

def check_system():
    # Load the data
    try:
        with open('data/corpus.json', 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print("Error: data/corpus.json not found.")
        return

    engine = BoundedGraphRAGEngine()
    chunk_store = BoundedChunkStore()

    print(f"--- Diagnostic Report ---")
    print(f"Total items in JSON: {len(data)}")
    
    # Check if we can find any entities
    count = 0
    for entry in data:
        # Example logic: extract 'doc' or 'noise' as a node
        node_id = entry.get('id', 'unknown')
        content = entry.get('content', '')
        
        # Simple rule: if we have valid content, add a dummy edge 
        # just to verify the system works
        if node_id and content:
            engine.insert_edge("System", node_id)
            chunk_store.add_extraction("System", node_id, content)
            count += 1

    print(f"Successfully processed {count} items.")
    print(f"Total edges in engine: {len(engine.edges)}")
    print(f"Test Query for 'System'...")
    
    res = engine.query("System", chunk_store)
    if res:
        print("Success! Context found.")
        print(f"Preview: {res[:150]}...")
    else:
        print("Failure: No context found.")

if __name__ == "__main__":
    check_system()