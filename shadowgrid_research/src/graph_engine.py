import hashlib
import heapq
import math
import re
from collections import defaultdict
from datetime import datetime

# Tolerance used when deciding whether a heap entry's cached entropy is still
# "close enough" to the freshly-recomputed value to trust without a re-push.
ENTROPY_STALENESS_EPSILON = 1e-9

# Entity extraction: a run of letters starting with a capital letter.
# Deliberately conservative — this will over-match sentence-initial words
# ("The", "Query") and under-match multi-word proper nouns ("New York" ->
# "New", "York" as two separate entities). Both are known limitations, not
# bugs; a real NER pass is out of scope for this bounded engine.
ENTITY_PATTERN = re.compile(r"\b[A-Z][a-zA-Z]+\b")


class BoundedChunkStore:
    def __init__(self, max_chunks=5000):
        self.max_chunks = max_chunks
        self.chunks = {}  # chunk_id -> raw_text
        self.edge_to_chunks = {}  # canonical_edge_key -> set(chunk_ids)
        self.chunk_queue = []  # FIFO queue for text chunks

    @staticmethod
    def _canonical_key(src, tgt):
        return tuple(sorted((src, tgt)))

    def add_extraction(self, src, tgt, text):
        chunk_id = hashlib.md5(text.encode('utf-8')).hexdigest()

        if chunk_id not in self.chunks:
            if len(self.chunks) >= self.max_chunks:
                victim_id = self.chunk_queue.pop(0)
                del self.chunks[victim_id]

                for e_key in list(self.edge_to_chunks.keys()):
                    self.edge_to_chunks[e_key].discard(victim_id)
                    if not self.edge_to_chunks[e_key]:
                        del self.edge_to_chunks[e_key]

            self.chunks[chunk_id] = text
            self.chunk_queue.append(chunk_id)

        key = self._canonical_key(src, tgt)
        if key not in self.edge_to_chunks:
            self.edge_to_chunks[key] = set()
        self.edge_to_chunks[key].add(chunk_id)

    def get_context(self, src, tgt):
        key = self._canonical_key(src, tgt)
        chunk_ids = self.edge_to_chunks.get(key, set())
        return [self.chunks[cid] for cid in chunk_ids if cid in self.chunks]


class BoundedGraphRAGEngine:
    def __init__(self, max_edges=50): # Hardcoded optimal 50
        self.max_edges = max_edges
        self.node_degrees = defaultdict(int)
        self.adjacency = defaultdict(set)
        self.edges = {}  # (src, tgt) -> true_current_entropy
        self.eviction_heap = []
        self.archive = []

    def _compute_local_edge_entropy(self, src, tgt):
        deg_src = self.node_degrees.get(src, 1)
        deg_tgt = self.node_degrees.get(tgt, 1)
        total_edges = max(len(self.edges), 1)

        p_src = deg_src / (2 * total_edges)
        p_tgt = deg_tgt / (2 * total_edges)

        return - (p_src * math.log2(p_src) + p_tgt * math.log2(p_tgt))

    def _garbage_collect_nodes(self, src, tgt):
        for node in (src, tgt):
            self.node_degrees[node] -= 1
            if self.node_degrees[node] <= 0:
                del self.node_degrees[node]
                if node in self.adjacency:
                    del self.adjacency[node]

    def _update_and_push(self, src, tgt):
        entropy = self._compute_local_edge_entropy(src, tgt)
        self.edges[(src, tgt)] = entropy
        heapq.heappush(self.eviction_heap, (entropy, src, tgt))

    def insert_edge(self, src, tgt):
        if src == tgt or (src, tgt) in self.edges or (tgt, src) in self.edges:
            return

        if self.archive:
            self.archive = [
                a for a in self.archive
                if not ((a["src"] == src and a["tgt"] == tgt) or (a["src"] == tgt and a["tgt"] == src))
            ]

        self.node_degrees[src] += 1
        self.node_degrees[tgt] += 1
        self.adjacency[src].add(tgt)
        self.adjacency[tgt].add(src)

        self._update_and_push(src, tgt)

        for node in (src, tgt):
            for neighbor in self.adjacency[node]:
                if neighbor == tgt and node == src:
                    continue
                e = (node, neighbor) if (node, neighbor) in self.edges else (neighbor, node)
                if e in self.edges:
                    self._update_and_push(e[0], e[1])

        while len(self.edges) > self.max_edges:
            entropy_key, cand_src, cand_tgt = heapq.heappop(self.eviction_heap)

            if (cand_src, cand_tgt) not in self.edges:
                continue

            true_entropy = self._compute_local_edge_entropy(cand_src, cand_tgt)

            if abs(true_entropy - entropy_key) > ENTROPY_STALENESS_EPSILON:
                self.edges[(cand_src, cand_tgt)] = true_entropy
                heapq.heappush(self.eviction_heap, (true_entropy, cand_src, cand_tgt))
                continue

            victim_src, victim_tgt = cand_src, cand_tgt
            live_entropy = true_entropy

            self.archive.append({
                "src": victim_src,
                "tgt": victim_tgt,
                "entropy_at_eviction": live_entropy,
                "timestamp": datetime.now().isoformat()
            })

            del self.edges[(victim_src, victim_tgt)]
            self.adjacency[victim_src].discard(victim_tgt)
            self.adjacency[victim_tgt].discard(victim_src)
            self._garbage_collect_nodes(victim_src, victim_tgt)

    def retrieve_subgraph_context(self, target_entities, chunk_store):
        # Production deduplication: Raw text-based set
        seen_chunks = set()
        target_set = set(target_entities)

        # Tier 1: Active Graph Retrieval
        valid_active_targets = target_set & set(self.node_degrees.keys())
        if valid_active_targets:
            for (src, tgt) in self.edges.keys():
                if src in valid_active_targets or tgt in valid_active_targets:
                    for text in chunk_store.get_context(src, tgt):
                        seen_chunks.add(text)

        # Tier 2: Archive Retrieval
        for archived_edge in self.archive:
            a_src, a_tgt = archived_edge["src"], archived_edge["tgt"]
            if (a_src, a_tgt) in self.edges or (a_tgt, a_src) in self.edges:
                continue
            if a_src in target_set or a_tgt in target_set:
                for text in chunk_store.get_context(a_src, a_tgt):
                    seen_chunks.add(text)

        return "\n".join(seen_chunks)

    def query(self, query_text, chunk_store):
        """
        Extract capitalized-word entities from query_text, resolve them
        against the active graph + archive, and return the combined raw
        chunk text from chunk_store.

        Returns "" (not None) when there's no query, no entities found, or
        no matching context — callers should treat empty string as "no
        context available", not as an error.
        """
        if not query_text or not query_text.strip():
            return ""

        entities = set(ENTITY_PATTERN.findall(query_text))
        if not entities:
            return ""

        return self.retrieve_subgraph_context(entities, chunk_store)