import hashlib
import heapq
import math
import re
from collections import defaultdict
from datetime import datetime

# Tolerance used when deciding whether a heap entry's cached entropy is still
# "close enough" to the freshly-recomputed value to trust without a re-push.
ENTROPY_STALENESS_EPSILON = 1e-9

# Entity extraction: a run of one or more consecutive capitalized "words",
# where each word may contain letters, digits, and hyphens ("Asset Qdrant",
# "New York", "Verification-Run-1", "Qdrant-2" all match as single spans).
# This MUST stay byte-for-byte in sync with ingestor.py's _ENTITY_PATTERN —
# they are two independent copies of the same contract (query-side vs.
# ingestion-side), and a prior drift (this pattern lacked \d and \-) is what
# caused Session 10's token-truncation bug: ingestion built nodes like
# "Verification-Run-1" but query-side extraction fragmented the same text
# into "Verification" + "Run", silently dropping the "-1" and producing an
# empty seed frontier. Still not real NER — it will merge unrelated
# capitalized words that happen to sit next to each other ("The Project" if
# "The" isn't filtered, or two distinct proper nouns separated only by a
# comma-less conjunction like "Alice Bob" in "I spoke to Alice Bob
# mentioned"). The stopword filter below handles the common sentence-initial
# case but not that one.
ENTITY_PATTERN = re.compile(r"\b[A-Z][a-zA-Z0-9\-]*(?:\s+[A-Z][a-zA-Z0-9\-]*)*\b")

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
    Find capitalized-word spans (letters/digits/hyphens per word, per
    ENTITY_PATTERN above) and strip leading stopwords from each. The
    resulting strings must match node names exactly as they were written
    into the graph by the ingestion pipeline (ingestor.py) and by relation
    extraction (generator.py). All three call sites share normalize_entity()
    from this module for the operational-prefix strip, so that part of the
    contract is enforced by import rather than by convention — but the
    ENTITY_LEADING_STOPWORDS handling below is local to query-side
    extraction and has no ingestion-side equivalent to drift out of sync
    with. Hyphenated/numeric tokens ("Verification-Run-1") contain no
    internal whitespace, so they pass through match.split() as a single
    word and are never fragmented by the stopword-stripping loop.
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
    def __init__(self, max_edges=50, max_archive=500):  # Hardcoded optimal 50
        self.max_edges = max_edges
        self.max_archive = max_archive
        self.node_degrees = defaultdict(int)
        self.adjacency = defaultdict(set)
        self.edges = {}  # (src, tgt) -> true_current_entropy
        self.eviction_heap = []

        # --- Archive tier (evicted edges), maintained incrementally ---
        # self.archive: FIFO-ordered list of archive entry dicts. This is an
        # append/pop(0) audit log, not a lookup structure — it exists so we
        # know eviction ORDER for the max_archive cap. It may contain
        # entries that are logically stale (superseded by a later
        # re-eviction, or dropped because the edge went active again); that
        # is fine and expected, see _evict_edge / insert_edge below.
        self.archive = []

        # self._archive_by_key: canonical_edge_key -> the CURRENT archive
        # entry dict for that edge (or absent if the edge isn't currently
        # archived). O(1) existence/identity check used to detect stale
        # self.archive entries and to reconcile archive state when an edge
        # is re-inserted as active.
        self._archive_by_key = {}

        # self.archive_index: node -> {canonical_edge_key: (neighbor, entropy)}.
        # This is the structure query-time traversal reads (via
        # _archived_edges_for_node). It is kept in sync with
        # _archive_by_key on every eviction, FIFO-prune, and re-insertion
        # event (see _add_to_archive_index / _remove_from_archive_index),
        # so retrieve_subgraph_context never rebuilds it from scratch.
        self.archive_index = defaultdict(dict)

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

    # ------------------------------------------------------------------
    # Archive index maintenance (O(1) dict ops — see class-level comment
    # on self.archive_index for what each structure is for).
    # ------------------------------------------------------------------

    def _add_to_archive_index(self, entry):
        src, tgt = entry["src"], entry["tgt"]
        key = tuple(sorted((src, tgt)))
        entropy = entry.get("entropy_at_eviction", 0.0)
        self.archive_index[src][key] = (tgt, entropy)
        self.archive_index[tgt][key] = (src, entropy)

    def _remove_from_archive_index(self, entry):
        src, tgt = entry["src"], entry["tgt"]
        key = tuple(sorted((src, tgt)))
        for node in (src, tgt):
            bucket = self.archive_index.get(node)
            if bucket is None:
                continue
            bucket.pop(key, None)
            if not bucket:
                del self.archive_index[node]

    def insert_edge(self, src, tgt):
        if src == tgt or (src, tgt) in self.edges or (tgt, src) in self.edges:
            return

        # If this exact edge currently has a live archived representation,
        # it's about to become active again — drop the stale archive copy.
        # This used to be `self.archive = [a for a in self.archive if ...]`,
        # a full linear scan + list rebuild on EVERY insert_edge call
        # (i.e. on the ingestion critical path, not just queries). It's now
        # two dict lookups. Any duplicate/stale copies left sitting in the
        # self.archive FIFO audit list are harmless no-ops — see the
        # identity check in _evict_edge's FIFO-prune step below.
        key = tuple(sorted((src, tgt)))
        stale = self._archive_by_key.pop(key, None)
        if stale is not None:
            self._remove_from_archive_index(stale)

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

            self._evict_edge(cand_src, cand_tgt, true_entropy)

    def _evict_edge(self, victim_src, victim_tgt, live_entropy):
        """
        Move an edge from the active graph into the bounded archive tier.
        Updates self.archive_index incrementally (O(1) dict ops) as part of
        the eviction itself, instead of leaving that bookkeeping to a
        per-query rebuild (the old _build_archive_index, see below).
        """
        print(f"[EVICTION TELEMETRY] Purging Edge: ({victim_src} <-> {victim_tgt})")
        print(f"  -> Active Degree of {victim_src}: {self.node_degrees.get(victim_src, 0)}")
        print(f"  -> Active Degree of {victim_tgt}: {self.node_degrees.get(victim_tgt, 0)}")

        entry = {
            "src": victim_src,
            "tgt": victim_tgt,
            "entropy_at_eviction": live_entropy,
            "timestamp": datetime.now().isoformat()
        }
        key = tuple(sorted((victim_src, victim_tgt)))

        # A prior archived copy of this exact edge (evicted, reinserted,
        # evicted again) is superseded by this fresher entry — drop its
        # index rows before installing the new ones.
        old = self._archive_by_key.get(key)
        if old is not None:
            self._remove_from_archive_index(old)

        self._archive_by_key[key] = entry
        self._add_to_archive_index(entry)
        self.archive.append(entry)

        # FIFO bound on the audit list: without this, self.archive grows
        # unboundedly with eviction count over the life of a long-running
        # process. Popping the FIFO head is O(1). We only need to also drop
        # the popped entry from archive_index/_archive_by_key if it's still
        # the CURRENT entry for that edge (identity check via `is`) — a
        # fresher re-eviction of the same edge may have already superseded
        # it in _archive_by_key, in which case this popped object is a
        # stale audit record only, and index bookkeeping for it already
        # happened above when the fresher entry was installed.
        if len(self.archive) > self.max_archive:
            victim_entry = self.archive.pop(0)
            v_key = tuple(sorted((victim_entry["src"], victim_entry["tgt"])))
            if self._archive_by_key.get(v_key) is victim_entry:
                del self._archive_by_key[v_key]
                self._remove_from_archive_index(victim_entry)

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

    def _archived_edges_for_node(self, node):
        """
        Archived edges touching `node`, read directly from the incrementally
        maintained self.archive_index — no scan of self.archive required.
        Replaces the old pattern of calling _build_archive_index() and then
        indexing into its result.
        """
        return [
            (neighbor, key, entropy)
            for key, (neighbor, entropy) in self.archive_index.get(node, {}).items()
        ]

    def _build_archive_index(self):
        """
        NOT on the query path anymore (Session 10 Objective 2). Kept only as
        an O(archive_size) reference implementation for tests/debugging that
        want to assert self.archive_index is consistent with self.archive —
        e.g. `assert engine._build_archive_index() == {n: {k: v for _, k, v
        in ...} ...}`-style checks. retrieve_subgraph_context and every
        query-time helper below read self.archive_index directly instead.
        """
        index = defaultdict(dict)
        for a in self.archive:
            a_src, a_tgt = a["src"], a["tgt"]
            key = tuple(sorted((a_src, a_tgt)))
            entropy = a.get("entropy_at_eviction", 0.0)
            index[a_src][key] = (a_tgt, entropy)
            index[a_tgt][key] = (a_src, entropy)
        return index

    def _edges_for_node(self, node):
        """Active + archived edges touching `node`, combined for traversal."""
        return self._active_edges_for_node(node) + self._archived_edges_for_node(node)

    def _select_expansion_edges(self, edges_with_entropy, hub_degree_threshold, hub_fanout_cap):
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
        hop. Counting archived edges via self.archive_index fixes this
        specific case.

        Still heuristic: this is structural specificity, not query
        relevance. Reading self.archive_index is now O(1) per node lookup
        (dict), not a per-query archive scan.
        """
        if len(edges_with_entropy) <= hub_degree_threshold:
            return edges_with_entropy

        def total_footprint(node):
            return self.node_degrees.get(node, 0) + len(self.archive_index.get(node, {}))

        ranked = sorted(edges_with_entropy, key=lambda item: total_footprint(item[0]))
        return ranked[:hub_fanout_cap]

    def _adaptive_hub_params(self):
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
        per call since node_degrees/archive_index change as edges are
        inserted/evicted; both are bounded (active by max_edges, archive by
        max_archive) so the node list is always small, and archive_index
        reads are O(1) dict lookups rather than a rebuilt structure.
        """
        all_nodes = set(self.node_degrees.keys()) | set(self.archive_index.keys())
        degrees = sorted(
            self.node_degrees.get(node, 0) + len(self.archive_index.get(node, {}))
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

        # No archive_index rebuild here — self.archive_index is already
        # current, maintained incrementally by insert_edge/_evict_edge on
        # every ingestion event. This is the Session 10 Objective 2 fix:
        # the old `archive_index = self._build_archive_index()` line (an
        # O(archive_size) scan) ran on every single call to this method.

        if hub_degree_threshold is None or hub_fanout_cap is None:
            auto_threshold, auto_cap = self._adaptive_hub_params()
            hub_degree_threshold = auto_threshold if hub_degree_threshold is None else hub_degree_threshold
            hub_fanout_cap = auto_cap if hub_fanout_cap is None else hub_fanout_cap

        # Seed frontier from BOTH active nodes and archive-only nodes — a
        # query entity that's currently evicted from the live graph but
        # still present in archived edges should still be a valid starting
        # point, not silently dropped the way the old Tier-1/Tier-2 split
        # dropped it unless it happened to co-occur with an active node.
        frontier = target_set & (set(self.node_degrees.keys()) | set(self.archive_index.keys()))
        visited_nodes = set(frontier)

        for _ in range(max_hops):
            if not frontier:
                break
            next_frontier = set()
            for node in frontier:
                candidates = self._edges_for_node(node)
                selected = self._select_expansion_edges(candidates, hub_degree_threshold, hub_fanout_cap)
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