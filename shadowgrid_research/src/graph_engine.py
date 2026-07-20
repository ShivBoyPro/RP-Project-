import hashlib
import heapq
import math
import re
from collections import defaultdict
from datetime import datetime

# Tolerance used when deciding whether a heap entry's cached entropy is still
# "close enough" to the freshly-recomputed value to trust without a re-push.
ENTROPY_STALENESS_EPSILON = 1e-9

# Entity extraction: a run of one or more consecutive capitalized words
# ("Asset Qdrant", "New York" match as single entities, not fragments).
# Still not real NER — it will merge unrelated capitalized words that
# happen to sit next to each other ("The Project" if "The" isn't filtered,
# or two distinct proper nouns separated only by a comma-less conjunction
# like "Alice Bob" in "I spoke to Alice Bob mentioned"). The stopword
# filter below handles the common sentence-initial case but not that one.
ENTITY_PATTERN = re.compile(r"\b[A-Z][a-zA-Z]*(?:\s+[A-Z][a-zA-Z]*)*\b")

# Common sentence-initial / interrogative capitalized words that are never
# themselves entities. Stripped from the FRONT of a matched span only —
# "The Project Qdrant" -> "Project Qdrant", not "Qdrant" (mid-span words are
# left alone since we can't tell "Project" is a stopword vs. part of a name
# without knowing the ingestion-side normalization rules in generator.py).
ENTITY_LEADING_STOPWORDS = {
    "The", "A", "An", "This", "That", "These", "Those",
    "Who", "What", "When", "Where", "Why", "How", "Which",
    "Is", "Are", "Was", "Were", "Do", "Does", "Did",
    "Can", "Could", "Would", "Should", "Will",
}

# Canonical operational-prefix strip. This is the SINGLE SOURCE OF TRUTH for
# prefix normalization across the whole pipeline — generator.py and
# ingestor.py both import normalize_entity() from here instead of keeping
# their own copies. (Previously there were three independent copies that had
# already drifted: ingestor.py was missing "Concept"/"Event", and this
# module's pattern was missing generator.py's re.IGNORECASE, so "concept X"
# or "event Y" could silently fracture into duplicate nodes depending on
# which code path touched them first.)
#
# One anchored strip of an operational prefix, not a repeatable stopword.
# Kept separate from ENTITY_LEADING_STOPWORDS (which pops iteratively) so
# "Project Asset Foo" strips to "Asset Foo", not "Foo".
ENTITY_PREFIX_PATTERN = re.compile(r"^(Project|Asset|Agent|Concept|Event)\s+", re.IGNORECASE)


def normalize_entity(name):
    """Strip a known operational prefix and surrounding whitespace so variant
    surface forms of the same entity ("Project ShadowGrid" vs "ShadowGrid")
    collapse onto one graph node. Canonical implementation — see module note
    above."""
    if not isinstance(name, str):
        return name
    return ENTITY_PREFIX_PATTERN.sub("", name).strip()


def _extract_entities(query_text):
    """
    Find capitalized-word spans and strip leading stopwords from each.
    The resulting strings must match node names exactly as they were written
    into the graph by the ingestion pipeline (ingestor.py) and by relation
    extraction (generator.py). All three call sites share normalize_entity()
    from this module for the operational-prefix strip, so that part of the
    contract is enforced by import rather than by convention — but the
    ENTITY_LEADING_STOPWORDS handling below is local to query-side
    extraction and has no ingestion-side equivalent to drift out of sync
    with.
    """
    entities = set()
    for match in ENTITY_PATTERN.findall(query_text):
        words = match.split()
        while words and words[0] in ENTITY_LEADING_STOPWORDS:
            words.pop(0)
        if not words:
            continue
        candidate = " ".join(words)
        candidate = normalize_entity(candidate)
        if candidate:
            entities.add(candidate)
    return entities


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
    def __init__(self, max_edges=50, max_archive=500): # Hardcoded optimal 50
        self.max_edges = max_edges
        self.max_archive = max_archive
        self.node_degrees = defaultdict(int)
        self.adjacency = defaultdict(set)
        self.edges = {}  # (src, tgt) -> true_current_entropy
        self.eviction_heap = []
        self.archive = []  # bounded FIFO of evicted edges; see insert_edge

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

            print(f"[EVICTION TELEMETRY] Purging Edge: ({victim_src} <-> {victim_tgt})")
            print(f"  -> Active Degree of {victim_src}: {self.node_degrees.get(victim_src, 0)}")
            print(f"  -> Active Degree of {victim_tgt}: {self.node_degrees.get(victim_tgt, 0)}")

            self.archive.append({
                "src": victim_src,
                "tgt": victim_tgt,
                "entropy_at_eviction": live_entropy,
                "timestamp": datetime.now().isoformat()
            })
            # FIFO bound: without this, self.archive (and the per-query
            # _build_archive_index scan over it) grows unboundedly with
            # eviction count, and query latency degrades linearly over the
            # life of a long-running process even though max_edges caps the
            # active graph.
            if len(self.archive) > self.max_archive:
                self.archive.pop(0)

            del self.edges[(victim_src, victim_tgt)]
            self.adjacency[victim_src].discard(victim_tgt)
            self.adjacency[victim_tgt].discard(victim_src)
            self._garbage_collect_nodes(victim_src, victim_tgt)

    def _active_edges_for_node(self, node):
        """Live (uncollapsed) edges touching `node`, with cached entropy."""
        result = []
        for neighbor in self.adjacency.get(node, ()):
            e = (node, neighbor) if (node, neighbor) in self.edges else (neighbor, node)
            entropy = self.edges.get(e)
            if entropy is not None:
                result.append((neighbor, e, entropy))
        return result

    def _build_archive_index(self):
        """
        Node -> list of (neighbor, canonical_edge_key, entropy_at_eviction)
        for archived edges. self.archive itself carries src/tgt, which is
        enough structure to traverse from — the earlier claim that archived
        edges have "no adjacency to traverse" was wrong: self.adjacency
        (the live index) is what gets cleared on GC, not the archive list.
        Rebuilt fresh per query since archive mutates on every re-insertion
        (see insert_edge's archive-filtering step) and is small by
        construction (bounded by eviction volume, not graph size).
        """
        index = defaultdict(list)
        for a in self.archive:
            a_src, a_tgt = a["src"], a["tgt"]
            key = tuple(sorted((a_src, a_tgt)))
            entropy = a.get("entropy_at_eviction", 0.0)
            index[a_src].append((a_tgt, key, entropy))
            index[a_tgt].append((a_src, key, entropy))
        return index

    def _edges_for_node(self, node, archive_index):
        """Active + archived edges touching `node`, combined for traversal."""
        return self._active_edges_for_node(node) + archive_index.get(node, [])

    def _select_expansion_edges(self, edges_with_entropy, hub_degree_threshold, hub_fanout_cap, archive_index):
        """
        Fan-out control for hub nodes. Below threshold, expand everything.
        Above threshold, keep only the `hub_fanout_cap` neighbors with the
        LOWEST total structural footprint — active degree PLUS archived
        edge count, not active degree alone.

        Active-degree-only ranking has a real blind spot: a node whose
        edges are almost entirely evicted shows active degree 0 (or is
        absent from node_degrees entirely) and gets ranked as if it were a
        specific leaf, even when it's actually a historical mega-hub with
        a dozen archived spokes waiting behind it. Confirmed by test: a
        12-spoke evicted hub outranked a genuine 1-edge leaf under
        degree-only sorting and would have dumped its spokes into the next
        hop. Counting archived edges via archive_index fixes this specific
        case.

        Still heuristic: this is structural specificity, not query
        relevance, and archive_index is rebuilt fresh per query so this
        cost scales with archive size (bounded by eviction volume).
        """
        if len(edges_with_entropy) <= hub_degree_threshold:
            return edges_with_entropy

        def total_footprint(node):
            return self.node_degrees.get(node, 0) + len(archive_index.get(node, []))

        ranked = sorted(edges_with_entropy, key=lambda item: total_footprint(item[0]))
        return ranked[:hub_fanout_cap]

    def _adaptive_hub_params(self, archive_index=None):
        """
        Data-driven fallback for hub_degree_threshold / hub_fanout_cap,
        used when the caller doesn't supply explicit values. Hardcoded
        8/5 defaults were untuned against any real degree distribution;
        this instead reads the graph's actual current degrees.

        threshold = 75th percentile of node footprint, floored at 3 so a
        node with an ordinary 2-3 edge fanout never gets treated as a hub
        just because the sample is small.
        fanout_cap = sqrt(node count), clamped to [3, 8].

        Footprint here is ACTIVE + ARCHIVED edges per node — the same
        combined footprint _select_expansion_edges ranks candidates on via
        total_footprint(). Previously this threshold was computed from
        active degree alone while the classification check it feeds
        (`len(edges_with_entropy) <= hub_degree_threshold`, where
        edges_with_entropy already includes archived edges) compared
        against that mismatched, artificially-low threshold. On a graph
        with heavy eviction history, active-only degrees skew low (nodes
        get fully deleted from node_degrees once their active degree hits
        0), dragging the percentile down toward the floor of 3 — at which
        point ordinary active nodes with a couple of archived edges tipped
        over 3 total and got wrongly fanout-capped as hubs. Using the same
        footprint definition on both sides of the comparison fixes that.

        Below 5 nodes there isn't enough of a distribution to derive a
        percentile from — percentile-of-2-samples is noise, not signal
        (confirmed by test: a 2-node graph pushed threshold down to 1 and
        capped normal fanout). Falls back to permissive fixed defaults in
        that regime instead of adapting to nonsense.

        Still a heuristic with no relevance signal — this controls HOW MANY
        edges expand, not WHICH ones are right for the query. Recomputed
        per call since node_degrees/archive change as edges are
        inserted/evicted; this is not free but both are bounded (active by
        max_edges, archive by max_archive) so the node list is always small.
        """
        if archive_index is None:
            archive_index = {}
        all_nodes = set(self.node_degrees.keys()) | set(archive_index.keys())
        degrees = sorted(
            self.node_degrees.get(node, 0) + len(archive_index.get(node, []))
            for node in all_nodes
        )
        n = len(degrees)
        if n < 5:
            return 8, 5
        idx = int(0.75 * (n - 1))
        threshold = max(degrees[idx], 3)
        fanout_cap = max(3, min(8, int(math.sqrt(n))))
        return threshold, fanout_cap

    def retrieve_subgraph_context(self, target_entities, chunk_store,
                                   max_hops=2, hub_degree_threshold=None, hub_fanout_cap=None):
        seen_chunks = set()
        visited_edges = set()
        target_set = set(target_entities)
        archive_index = self._build_archive_index()

        if hub_degree_threshold is None or hub_fanout_cap is None:
            auto_threshold, auto_cap = self._adaptive_hub_params(archive_index)
            hub_degree_threshold = auto_threshold if hub_degree_threshold is None else hub_degree_threshold
            hub_fanout_cap = auto_cap if hub_fanout_cap is None else hub_fanout_cap

        # Seed frontier from BOTH active nodes and archive-only nodes — a
        # query entity that's currently evicted from the live graph but
        # still present in archived edges should still be a valid starting
        # point, not silently dropped the way the old Tier-1/Tier-2 split
        # dropped it unless it happened to co-occur with an active node.
        frontier = target_set & (set(self.node_degrees.keys()) | set(archive_index.keys()))
        visited_nodes = set(frontier)

        for _ in range(max_hops):
            if not frontier:
                break
            next_frontier = set()
            for node in frontier:
                candidates = self._edges_for_node(node, archive_index)
                selected = self._select_expansion_edges(candidates, hub_degree_threshold, hub_fanout_cap, archive_index)
                for neighbor, e_key, _entropy in selected:
                    canon = tuple(sorted(e_key))
                    if canon in visited_edges:
                        continue
                    visited_edges.add(canon)
                    for text in chunk_store.get_context(e_key[0], e_key[1]):
                        seen_chunks.add(text)
                    if neighbor not in visited_nodes:
                        next_frontier.add(neighbor)
            visited_nodes |= next_frontier
            frontier = next_frontier

        return "\n".join(seen_chunks)

    def query(self, query_text, chunk_store, max_hops=2, hub_degree_threshold=None, hub_fanout_cap=None):
        """
        Extract multi-word capitalized entities from query_text (stopword-
        stripped), resolve them against the active graph + archive via
        bounded BFS, and return the combined raw chunk text from chunk_store.

        max_hops: how many edge-traversals out from the seed entities.
        hub_degree_threshold: node degree above which fanout capping kicks
            in. None (default) -> computed per-call from the live degree
            distribution; see _adaptive_hub_params.
        hub_fanout_cap: max neighbors expanded per hub node per hop. None
            (default) -> same adaptive computation.

        Returns "" (not None) when there's no query, no entities found, or
        no matching context — callers should treat empty string as "no
        context available", not as an error.
        """
        if not query_text or not query_text.strip():
            return ""

        entities = _extract_entities(query_text)
        if not entities:
            return ""

        return self.retrieve_subgraph_context(
            entities, chunk_store,
            max_hops=max_hops,
            hub_degree_threshold=hub_degree_threshold,
            hub_fanout_cap=hub_fanout_cap,
        )