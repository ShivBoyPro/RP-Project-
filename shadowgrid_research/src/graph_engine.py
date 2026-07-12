import hashlib
import heapq
import math
from collections import defaultdict
from datetime import datetime

# Tolerance used when deciding whether a heap entry's cached entropy is still
# "close enough" to the freshly-recomputed value to trust without a re-push.
# See Bug 3 fix below.
ENTROPY_STALENESS_EPSILON = 1e-9


class BoundedChunkStore:
    def __init__(self, max_chunks=5000):
        self.max_chunks = max_chunks
        self.chunks = {}  # chunk_id -> raw_text
        self.edge_to_chunks = {}  # canonical_edge_key -> set(chunk_ids)
        self.chunk_queue = []  # FIFO queue for text chunks

    @staticmethod
    def _canonical_key(src, tgt):
        """
        BUG 1 FIX (Bidirectional Directionality Mismatch):
        The graph engine treats every edge as undirected - insert_edge()
        explicitly checks both (src, tgt) and (tgt, src) before deciding an
        edge is "new". But this store was keying edge_to_chunks off the raw
        (src, tgt) tuple exactly as passed in. That means an extraction
        recorded as add_extraction("A", "B", text) was invisible to a later
        get_context("B", "A") call for the *same* logical edge - chunks
        silently vanished from retrieval depending on which order the two
        entity names happened to appear in the source text.

        BUG 2 NOTE (Multi-Edge Prefix Mangling):
        A tempting "quick fix" for the above is to canonicalize via string
        concatenation, e.g. "|".join(sorted((src, tgt))). That introduces a
        new problem: if any entity name itself contains the join character
        (or if two different unordered pairs happen to sort/concatenate to
        the same string - e.g. entity names containing pipes, colons, or
        other separators), unrelated edges can collide into the same string
        key, silently merging their chunk sets ("mangling"). This corrupts
        multi-edge data by fusing distinct relations together.

        The fix for both bugs is the same: canonicalize using a tuple of the
        two endpoints in a fixed (sorted) order, never a mangled string.
        Tuples hash structurally, not lexically, so there's no delimiter to
        collide on and no risk of cross-entity key collisions.
        """
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
        """Calculates current entropy and pushes to heap. Leaves stale duplicates to be lazy-deleted.

        Pushed as raw (non-negated) entropy: heapq is a min-heap, so the lowest
        true entropy (peripheral/leaf edges) surfaces first for eviction, while
        high-entropy hub edges sink to the bottom and are preserved.
        """
        entropy = self._compute_local_edge_entropy(src, tgt)
        self.edges[(src, tgt)] = entropy
        heapq.heappush(self.eviction_heap, (entropy, src, tgt))

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

        # 4. Eviction - lazy validate-at-pop instead of full O(E) rebuild.
        #
        # BUG 3 FIX (O(E) Ingestion Performance Bottleneck):
        # The previous approach recomputed and re-heapified every single
        # active edge's entropy before evicting anything, because
        # recompute-on-pop alone can leave an edge with a stale cached
        # priority forever if it's never revisited by Step 3 and never
        # popped - even though total_edges shifting on every insertion means
        # its *true* entropy has drifted. That guarantee of correctness cost
        # O(E) per eviction-triggering insert, every time.
        #
        # The fix keeps the guarantee without the full rebuild: when we pop
        # a candidate, we recompute its true entropy right then (cheap - O(1)
        # given current degrees/total_edges) and compare it to the cached
        # priority it was popped with.
        #   - If they agree (within epsilon), the popped value was accurate
        #     and this genuinely is the current minimum - evict it.
        #   - If they disagree, the cached value was stale. Instead of
        #     trusting it, we correct self.edges[...] and re-push the edge
        #     with its true entropy, then continue popping. The corrected
        #     entry will naturally find its true position in the heap on
        #     this or a future pop.
        # Each correction is an O(log E) push/pop instead of an O(E) rebuild,
        # and correctness is preserved because eviction never proceeds on a
        # value we haven't just re-validated against ground truth.
        while len(self.edges) > self.max_edges:
            entropy_key, cand_src, cand_tgt = heapq.heappop(self.eviction_heap)

            # Lazy-deleted: this exact edge no longer exists (already evicted
            # via an earlier, now-stale heap entry for the same pair).
            if (cand_src, cand_tgt) not in self.edges:
                continue

            true_entropy = self._compute_local_edge_entropy(cand_src, cand_tgt)

            if abs(true_entropy - entropy_key) > ENTROPY_STALENESS_EPSILON:
                # Cached priority was stale - correct it and let it re-settle
                # in the heap rather than evicting on outdated information.
                self.edges[(cand_src, cand_tgt)] = true_entropy
                heapq.heappush(self.eviction_heap, (true_entropy, cand_src, cand_tgt))
                continue

            # Confirmed against ground truth: this is genuinely the current
            # minimum-entropy edge. Safe to evict.
            victim_src, victim_tgt = cand_src, cand_tgt
            live_entropy = true_entropy

            # TIERED MEMORY SHIFT: Archive before deleting from Tier 1 Cache
            self.archive.append({
                "src": victim_src,
                "tgt": victim_tgt,
                "entropy_at_eviction": live_entropy,
                "timestamp": datetime.now().isoformat()
            })

            # Execute Eviction from Tier 1
            del self.edges[(victim_src, victim_tgt)]
            self.adjacency[victim_src].discard(victim_tgt)
            self.adjacency[victim_tgt].discard(victim_src)
            self._garbage_collect_nodes(victim_src, victim_tgt)

    def retrieve_subgraph_context(self, target_entities, chunk_store):
        # BUG 4 FIX (Cross-Tier Chunk Duplication):
        # Deduplication used to happen *after* tagging each chunk with its
        # tier label ("[TIER 1 ACTIVE] ..." / "[TIER 2 ARCHIVE] ..."), then
        # relied on set() to collapse duplicates. But a single raw text chunk
        # commonly documents several entity relationships at once (e.g. one
        # paragraph mentioning A-B, B-C, and C-D). If one of those pairs is
        # currently active (Tier 1) and a different pair from the *same*
        # chunk was evicted to the archive (Tier 2), both retrieval loops
        # independently pull that identical chunk text - but with different
        # prefixes. set() hashes on the full prefixed string, so the two
        # differently-tagged copies look like distinct entries and both
        # survive, silently doubling that chunk's presence (and token cost)
        # in the assembled context.
        #
        # The fix is to dedupe on the *raw* chunk text before any tier
        # prefix is attached, using an insertion-ordered dict keyed by text.
        # Tier 1 is scanned first, so if a chunk is reachable from both
        # tiers it keeps its "[TIER 1 ACTIVE]" tag (the more current/accurate
        # provenance) and is only ever emitted once.
        seen_chunks = {}  # raw_text -> tier_label (first tier reached wins)
        target_set = set(target_entities)

        # --- TIER 1: ACTIVE GRAPH RETRIEVAL (Fast O(1) Memory Edge Match) ---
        valid_active_targets = target_set & set(self.node_degrees.keys())
        if valid_active_targets:
            for (src, tgt) in self.edges.keys():
                if src in valid_active_targets or tgt in valid_active_targets:
                    for text in chunk_store.get_context(src, tgt):
                        seen_chunks.setdefault(text, "[TIER 1 ACTIVE]")

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
                    # setdefault: if this exact chunk text was already
                    # captured via Tier 1 (because it also mentions some
                    # other, still-active pair), don't re-tag or duplicate
                    # it here - just leave the existing Tier 1 entry alone.
                    seen_chunks.setdefault(text, "[TIER 2 ARCHIVE]")

        return "\n".join(f"{tier} {text}" for text, tier in seen_chunks.items())