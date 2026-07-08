import hashlib
import heapq
import math
from collections import defaultdict
from datetime import datetime

class BoundedChunkStore:
    def __init__(self, max_chunks=5000):
        self.max_chunks = max_chunks
        self.chunks = {}  # chunk_id -> raw_text
        self.edge_to_chunks = {}  # (src, tgt) -> set(chunk_ids)
        self.chunk_queue = []  # FIFO queue for text chunks

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

        if (src, tgt) not in self.edge_to_chunks:
            self.edge_to_chunks[(src, tgt)] = set()
        self.edge_to_chunks[(src, tgt)].add(chunk_id)

    def get_context(self, src, tgt):
        chunk_ids = self.edge_to_chunks.get((src, tgt), set())
        return [self.chunks[cid] for cid in chunk_ids if cid in self.chunks]


class BoundedGraphRAGEngine:
    def __init__(self, max_edges=1000):
        self.max_edges = max_edges
        self.node_degrees = defaultdict(int)
        self.adjacency = defaultdict(set)  # O(1) neighborhood resolution
        self.edges = {}  # (src, tgt) -> true_current_entropy
        self.eviction_heap = []

        # TIER 2 MEMORY: Cold Storage Archive
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
        """Calculates current entropy and pushes to heap. Leaves stale duplicates to be lazy-deleted."""
        entropy = self._compute_local_edge_entropy(src, tgt)
        self.edges[(src, tgt)] = entropy
        heapq.heappush(self.eviction_heap, (-entropy, src, tgt))

    def insert_edge(self, src, tgt):
        if src == tgt or (src, tgt) in self.edges or (tgt, src) in self.edges:
            return

        # Purge any stale archive record(s) for this exact pair. If this
        # edge was previously evicted and archived, then the same entity
        # pair co-occurred again later in the stream, it's being reinserted
        # as active here - the old "this was evicted" record no longer
        # reflects reality and would otherwise cause Tier 2 lookups to
        # double-count an edge that is simultaneously active in Tier 1.
        if self.archive:
            self.archive = [
                a for a in self.archive
                if not ((a["src"] == src and a["tgt"] == tgt) or (a["src"] == tgt and a["tgt"] == src))
            ]

        # 1. Update Topology
        self.node_degrees[src] += 1
        self.node_degrees[tgt] += 1
        self.adjacency[src].add(tgt)
        self.adjacency[tgt].add(src)

        # 2. Add New Edge
        self._update_and_push(src, tgt)

        # 3. Localized Topological Recalculation O(D)
        for node in (src, tgt):
            for neighbor in self.adjacency[node]:
                if neighbor == tgt and node == src:
                    continue
                e = (node, neighbor) if (node, neighbor) in self.edges else (neighbor, node)
                if e in self.edges:
                    self._update_and_push(e[0], e[1])

        # 4. Lazy-Evaluation Eviction Loop with Tiered Handshake
        while len(self.edges) > self.max_edges:
            neg_entropy, victim_src, victim_tgt = heapq.heappop(self.eviction_heap)

            # STALE CHECK: Verify if heap record matches true current mapping
            current_true_entropy = self.edges.get((victim_src, victim_tgt))
            if current_true_entropy is None or neg_entropy != -current_true_entropy:
                continue

            # TIERED MEMORY SHIFT: Archive before deleting from Tier 1 Cache
            self.archive.append({
                "src": victim_src,
                "tgt": victim_tgt,
                "entropy_at_eviction": current_true_entropy,
                "timestamp": datetime.now().isoformat()
            })

            # Execute Eviction from Tier 1
            del self.edges[(victim_src, victim_tgt)]
            self.adjacency[victim_src].discard(victim_tgt)
            self.adjacency[victim_tgt].discard(victim_src)
            self._garbage_collect_nodes(victim_src, victim_tgt)

    def retrieve_subgraph_context(self, target_entities, chunk_store):
        matched_contexts = []
        target_set = set(target_entities)

        # --- TIER 1: ACTIVE GRAPH RETRIEVAL (Fast O(1) Memory Edge Match) ---
        valid_active_targets = target_set & set(self.node_degrees.keys())
        if valid_active_targets:
            for (src, tgt) in self.edges.keys():
                if src in valid_active_targets or tgt in valid_active_targets:
                    for text in chunk_store.get_context(src, tgt):
                        matched_contexts.append(f"[TIER 1 ACTIVE] {text}")

        # --- TIER 2: ARCHIVE RETRIEVAL (Sequential Fallback Scan) ---
        # Only counts as a genuine Tier 2 contribution if this exact pair is
        # NOT currently active in Tier 1. Without this check, an edge that
        # was evicted and later reinserted (a common pattern when the same
        # entity pair co-occurs across multiple documents) would be pulled
        # from both tiers simultaneously, inflating Tier 2 utilization even
        # when the active graph alone already covers that fact.
        for archived_edge in self.archive:
            a_src, a_tgt = archived_edge["src"], archived_edge["tgt"]
            if (a_src, a_tgt) in self.edges or (a_tgt, a_src) in self.edges:
                continue
            if a_src in target_set or a_tgt in target_set:
                for text in chunk_store.get_context(a_src, a_tgt):
                    matched_contexts.append(f"[TIER 2 ARCHIVE] {text}")

        return "\n".join(set(matched_contexts))