import json
import os
import re

from graph_engine import normalize_entity

# Generic capitalized-phrase matcher: catches multi-word proper nouns like
# "Project ShadowGrid", "Concept Vector-Embeddings", "Agent Bianca", as well
# as single-token proper nouns like "PostgreSQL" or "Qdrant".
_ENTITY_PATTERN = re.compile(r"\b[A-Z][a-zA-Z0-9\-]*(?:\s+[A-Z][a-zA-Z0-9\-]*)*\b")

# Sentence-leading / generic words that match the capitalization pattern but
# aren't entities. Filtered out to keep the graph from filling with noise.
_STOPWORDS = {
    "The", "This", "That", "These", "Those", "Internal", "Architecture",
    "Decommissioning", "Core", "System", "Total", "Success", "Failure",
    "Preview", "Test", "Query", "Result", "Error", "Storage", "Infrastructure",
}


def extract_entities(content):
    """Pull normalized proper-noun entities out of a chunk of text."""
    if not content:
        return set()

    entities = set()
    for match in _ENTITY_PATTERN.findall(content):
        candidate = normalize_entity(match)
        if not candidate or candidate in _STOPWORDS:
            continue
        if len(candidate) < 3:
            continue
        entities.add(candidate)
    return entities


def ingest_corpus(file_path, engine, chunk_store):
    if not os.path.exists(file_path):
        print(f"Error: Corpus file not found at {file_path}")
        return False

    with open(file_path, 'r') as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            print("Error: Invalid JSON.")
            return False

    print(f"Loading {len(data)} items into engine...")

    for entry in data:
        # data/corpus.json keys documents as "doc_id"; other fixtures (e.g.
        # corpus_large.json) use "id". Accept either so ingestion doesn't
        # silently no-op against real corpus data.
        node_id = entry.get("doc_id") or entry.get("id")
        content = entry.get("content")

        if not node_id or not content:
            continue

        entities = extract_entities(content)

        if not entities:
            # No proper nouns found (e.g. pure boilerplate/log noise) — still
            # index the raw chunk under its own doc id so the content isn't
            # silently dropped, but don't fabricate a fake relational edge.
            chunk_store.add_extraction(node_id, node_id, content)
            continue

        entity_list = sorted(entities)

        # Link every entity to its source document for provenance, and index
        # the chunk content under the entity so retrieval-by-entity works.
        for entity in entity_list:
            engine.insert_edge(entity, node_id)
            chunk_store.add_extraction(entity, node_id, content)

        # Link co-occurring entities to each other — this is what gives the
        # graph actual relational structure (e.g. ShadowGrid <-> PostgreSQL,
        # ShadowGrid <-> Bianca) instead of a single "System" hub-and-spoke.
        for i in range(len(entity_list)):
            for j in range(i + 1, len(entity_list)):
                engine.insert_edge(entity_list[i], entity_list[j])

    return True