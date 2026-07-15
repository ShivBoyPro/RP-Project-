import os
import json
import random
import re
import time
from datetime import datetime, timedelta
from groq import Groq
from graph_engine import BoundedChunkStore, BoundedGraphRAGEngine
import os
from dotenv import load_dotenv

# Load the keys out of the local .env file
load_dotenv()

# Safely fetch the variable from memory
api_key = os.getenv("GROQ_API_KEY")

# ---------------------------------------------------------------------------
# Deterministic world model
#
# Instead of hand-writing 50 documents and 20 queries independently (which
# risks the ground_truth in the eval suite silently drifting out of sync
# with what the corpus actually says), both are generated FROM the same
# underlying "world state" per project. The eval suite's ground_truth is
# read directly off this world state, so it's guaranteed correct relative
# to whatever documents were generated - not retyped by hand.
#
# Entity pools are deliberately reused across projects (e.g. every project
# starts on "PostgreSQL") so that querying "who uses Qdrant now" or "who is
# project X's primary contact" is genuinely ambiguous without correctly
# resolving cross-project relations - this is what creates real structural
# noise/density, rather than noise docs that are trivially distinguishable
# from signal docs.
# ---------------------------------------------------------------------------

PROJECT_NAMES = ["ShadowGrid", "IronVault", "NightOwl", "CobaltMesh", "SilentTide"]
ENGINEER_POOL = ["Alexander", "Bianca", "Desmond", "Farah", "Grigori", "Hana", "Ivo", "Junko"]
OLD_DB_POOL = ["PostgreSQL", "MySQL", "MongoDB"]
NEW_DB_POOL = ["Qdrant", "Pinecone", "Weaviate"]
CONCEPT_POOL = ["Vector-Embeddings", "Sparse-Retrieval", "Cross-Encoder-Rerank", "Hybrid-Search"]
EVENT_POOL = ["Verification-Run-1", "Audit-Cycle-Alpha", "Compliance-Sweep-2", "Load-Test-Beta"]


class ProjectWorld:
    """A single project's deterministic timeline: two engineers (one
    removed, one becomes primary), a DB migration, a concept introduction,
    and a validation event. Every field is drawn from the shared pools so
    the same entity name (e.g. 'PostgreSQL') recurs across multiple,
    otherwise-unrelated projects.
    """

    def __init__(self, project_name, rng, start_time):
        self.project_name = project_name
        self.engineer_initial = rng.choice(ENGINEER_POOL)
        remaining_engineers = [e for e in ENGINEER_POOL if e != self.engineer_initial]
        self.engineer_final = rng.choice(remaining_engineers)
        self.db_old = rng.choice(OLD_DB_POOL)
        self.db_new = rng.choice(NEW_DB_POOL)
        self.concept = rng.choice(CONCEPT_POOL)
        self.event = rng.choice(EVENT_POOL)
        self.start_time = start_time

    def documents(self):
        t = self.start_time
        docs = [
            {
                "timestamp": t.isoformat(),
                "doc_id": f"{self.project_name.lower()}_001",
                "content": (
                    f"Project {self.project_name} initiates development. The core infrastructure is "
                    f"established to utilize a {self.db_old} instance as its primary database "
                    f"architecture. Core engineering contact is Agent {self.engineer_initial}."
                ),
            },
            {
                "timestamp": (t + timedelta(days=30)).isoformat(),
                "doc_id": f"{self.project_name.lower()}_002",
                "content": (
                    f"Agent {self.engineer_initial} introduces Concept {self.concept} to "
                    f"{self.project_name}. The system utilizes HuggingFace models for embedding "
                    f"computation. Asset {self.db_old} handles metadata storage."
                ),
            },
            {
                "timestamp": (t + timedelta(days=60)).isoformat(),
                "doc_id": f"{self.project_name.lower()}_003",
                "content": (
                    f"CRITICAL INFRASTRUCTURE UPDATE: Project {self.project_name} completely "
                    f"migrates away from {self.db_old}. Asset {self.db_old} is fully "
                    f"decommissioned. The current database stack for {self.project_name} is now "
                    f"completely transitioned to Asset {self.db_new} for vector and metadata handling."
                ),
            },
            {
                "timestamp": (t + timedelta(days=90)).isoformat(),
                "doc_id": f"{self.project_name.lower()}_004",
                "content": (
                    f"Agent {self.engineer_final} joins Project {self.project_name} to supervise "
                    f"evaluation loops. {self.engineer_final} updates the pipeline settings, "
                    f"establishing Event {self.event} to validate accuracy."
                ),
            },
            {
                "timestamp": (t + timedelta(days=120)).isoformat(),
                "doc_id": f"{self.project_name.lower()}_005",
                "content": (
                    f"Personnel reassignment: Agent {self.engineer_initial} is completely removed "
                    f"from Project {self.project_name} and moved to internal infrastructure. Agent "
                    f"{self.engineer_final} is now the sole primary contact and lead engineer for "
                    f"{self.project_name} development."
                ),
            },
        ]
        return docs

    def queries(self):
        return [
            {
                "query_id": f"{self.project_name}_Q_temporal_drift",
                "type": "temporal_drift",
                "query": f"What is the current active database stack utilized by Project {self.project_name}?",
                "ground_truth": (
                    f"Asset {self.db_new}. The system was originally on {self.db_old} but completed "
                    f"a migration to {self.db_new}, fully decommissioning {self.db_old}."
                ),
                "target_entities": [self.project_name, self.db_old, self.db_new],
            },
            {
                "query_id": f"{self.project_name}_Q_multi_hop_contradiction",
                "type": "multi_hop_contradiction",
                "query": f"Who is the primary engineering contact for the project that utilizes Asset {self.db_new}?",
                "ground_truth": (
                    f"Agent {self.engineer_final}. {self.project_name} uses {self.db_new}, and while "
                    f"{self.engineer_initial} was the original lead, they were removed and "
                    f"{self.engineer_final} is now the sole primary contact."
                ),
                "target_entities": [self.project_name, self.db_new, self.engineer_initial, self.engineer_final],
            },
            {
                "query_id": f"{self.project_name}_Q_multi_hop_concept",
                "type": "multi_hop",
                "query": f"Which algorithmic concept was introduced to the project managed by Agent {self.engineer_final}?",
                "ground_truth": (
                    f"Concept {self.concept}. It was introduced to Project {self.project_name}, which "
                    f"is now managed by Agent {self.engineer_final}."
                ),
                "target_entities": [self.engineer_final, self.project_name, self.concept],
            },
            {
                "query_id": f"{self.project_name}_Q_entity_lookup",
                "type": "entity_lookup",
                "query": f"Who was the initial engineer for Project {self.project_name}?",
                "ground_truth": f"Agent {self.engineer_initial}.",
                "target_entities": [self.project_name, self.engineer_initial],
            },
        ]


class StreamingDatasetGenerator:
    def __init__(self, base_dir="data", num_projects=5, noise_docs=25, seed=42):
        self.base_dir = base_dir
        self.corpus_dir = os.path.join(base_dir, "raw_corpus")
        self.eval_file = os.path.join(base_dir, "evaluation_suite.json")
        self.num_projects = num_projects
        self.noise_docs = noise_docs
        self.rng = random.Random(seed)  # seeded for reproducibility across runs

        os.makedirs(self.corpus_dir, exist_ok=True)

        if num_projects > len(PROJECT_NAMES):
            raise ValueError(f"Only {len(PROJECT_NAMES)} project names defined; requested {num_projects}")

        self.worlds = []
        base_start = datetime(2026, 1, 1, 9, 0, 0)
        for i, name in enumerate(PROJECT_NAMES[:num_projects]):
            # Stagger project start times so timestamps interleave rather
            # than each project's 5 docs sitting in one contiguous block.
            project_start = base_start + timedelta(days=self.rng.randint(0, 20) + i * 5)
            self.worlds.append(ProjectWorld(name, self.rng, project_start))

    def _generate_noise_docs(self):
        noise = []
        for i in range(1, self.noise_docs + 1):
            noise.append({
                "timestamp": (datetime(2026, 1, 1, 9, 0, 0) + timedelta(hours=self.rng.randint(0, 4000))).isoformat(),
                "doc_id": f"noise_{i:03d}",
                "content": (
                    f"Infrastructure log sequence reference alpha-{i * 100}. System operations "
                    f"running within normal parameters. Storage array check completed for cluster "
                    f"node {i}."
                ),
            })
        return noise

    def generate_streaming_corpus(self):
        all_docs = []
        for world in self.worlds:
            all_docs.extend(world.documents())
        all_docs.extend(self._generate_noise_docs())

        # Sort by timestamp so the corpus reads as one interleaved stream
        # across projects, matching how a real ingestion feed would arrive.
        all_docs.sort(key=lambda d: d["timestamp"])

        for doc in all_docs:
            file_path = os.path.join(self.corpus_dir, f"{doc['doc_id']}.json")
            with open(file_path, "w") as f:
                json.dump(doc, f, indent=4)

        print(f"[SUCCESS] Injected {len(all_docs)} streaming documents into {self.corpus_dir} "
              f"({sum(len(w.documents()) for w in self.worlds)} signal, {self.noise_docs} noise)")

        monolithic_path = os.path.join(self.base_dir, "corpus.json")
        with open(monolithic_path, "w") as f:
            json.dump(all_docs, f, indent=4)
        print(f"[SUCCESS] Exported monolithic baseline corpus to {monolithic_path}")

    def generate_evaluation_suite(self):
        queries = []
        for world in self.worlds:
            queries.extend(world.queries())

        with open(self.eval_file, "w") as f:
            json.dump(queries, f, indent=4)

        print(f"[SUCCESS] Compiled evaluation suite with {len(queries)} gold-standard tasks "
              f"({self.num_projects} projects x 4 query types) at {self.eval_file}")


_client = None


def _get_client():
    """Lazily instantiate the Groq client on first use, rather than at
    module import time. Instantiating at import time meant GROQ_API_KEY had
    to already be bound in the environment before this module could even be
    imported - breaking test suites, deployment runners, or any script that
    sets up env vars after import."""
    global _client
    if _client is None:
        _client = Groq()
    return _client


# Operational prefixes that appear in source text but should not survive
# into the graph as part of the entity name, or "Project ShadowGrid" and
# "ShadowGrid" fracture into two distinct nodes even though every downstream
# query and ground_truth string uses the bare form.
ENTITY_PREFIX_PATTERN = re.compile(r"^(Project|Asset|Agent|Concept|Event)\s+", re.IGNORECASE)


def normalize_entity(name):
    """Strip known operational prefixes and surrounding whitespace so
    variant surface forms of the same entity collapse onto one node."""
    if not isinstance(name, str):
        return name
    return ENTITY_PREFIX_PATTERN.sub("", name).strip()


def extract_relations_streaming(chunk_text, max_retries=5, client=None):
    """
    Extracts entity relation pairs via Groq. Includes 429 retry/backoff -
    at 50 docs, sequential extraction calls will hit the TPM rate limit
    partway through; without backoff, later documents in the corpus would
    silently contribute zero edges (extraction "dropped"), skewing the
    graph toward whichever projects happened to be ingested first.

    `client` can be passed explicitly (e.g. from a test harness); otherwise
    a lazily-initialized module-level client is used.
    """
    if client is None:
        client = _get_client()

    prompt = f"""
Analyze the text and extract direct relationships between technical entities.
Output ONLY a JSON object with a single key "relations" containing an array of pairs.
Format: {{"relations": [["EntityA", "EntityB"]]}}
Do not invent relationships. Ignore generic nouns.

Strip operational prefixes from entity names before returning them: do not
include the words "Project", "Asset", "Agent", "Concept", or "Event" as part
of an entity name. For example, return "ShadowGrid" instead of "Project
ShadowGrid", and "Alexander" instead of "Agent Alexander".

Text: {chunk_text}
"""
    last_error = None
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0.0,
            )

            parsed = json.loads(response.choices[0].message.content)

            valid_relations = []
            for pair in parsed.get("relations", []):
                if isinstance(pair, list) and len(pair) == 2:
                    src, tgt = normalize_entity(pair[0]), normalize_entity(pair[1])
                    if src and tgt:
                        valid_relations.append((src, tgt))

            return valid_relations

        except Exception as e:
            body = str(e)
            last_error = body

            # Rate limiting: honor the server's requested backoff if given.
            match = re.search(r"try again in ([\d.]+)s", body)
            if match:
                wait = float(match.group(1)) + 0.5
                print(f"[RATE LIMIT] backing off {wait:.2f}s (attempt {attempt + 1}/{max_retries})")
                time.sleep(wait)
                continue
            if "429" in body or "rate_limit" in body:
                wait = 2 ** attempt
                print(f"[RATE LIMIT] backing off {wait:.2f}s (attempt {attempt + 1}/{max_retries})")
                time.sleep(wait)
                continue

            # Transient connectivity/server-side failures: dropped packets,
            # DNS blips, connection resets, timeouts, or a 502/503 from the
            # gateway. These are not evidence the request itself is bad, so
            # retry with backoff rather than silently discarding the
            # document's edges (which would corrupt the graph's structural
            # footprint based purely on infrastructure flakiness).
            is_transient = isinstance(e, (
                ConnectionError, TimeoutError, OSError,
            )) or any(
                marker in body
                for marker in (
                    "Connection", "connection", "timeout", "Timeout",
                    "502", "503", "504", "Bad Gateway", "Service Unavailable",
                    "Gateway Timeout", "Temporary failure", "reset by peer",
                )
            )
            if is_transient:
                wait = 2 ** attempt
                print(f"[TRANSIENT ERROR] backing off {wait:.2f}s (attempt {attempt + 1}/{max_retries}): {e}")
                time.sleep(wait)
                continue

            # Non-retryable failure (e.g. malformed request, auth error,
            # unparseable response schema): retrying won't help.
            print(f"[EXTRACTION DROPPED] Model failure: {e}")
            return []

    print(f"[EXTRACTION DROPPED] Max retries exceeded, last failure -> {last_error}")
    return []


def run_pipeline(num_projects=5, noise_docs=25):
    generator = StreamingDatasetGenerator(num_projects=num_projects, noise_docs=noise_docs)
    generator.generate_streaming_corpus()
    generator.generate_evaluation_suite()

    corpus_path = "data/corpus.json"
    if not os.path.exists(corpus_path):
        print(f"Corpus file {corpus_path} not found.")
        return

    chunk_store = BoundedChunkStore(max_chunks=2000)
    engine = BoundedGraphRAGEngine(max_edges=1000)

    # Read the monolithic corpus rather than listing raw_corpus/ alphabetically.
    # Filenames are prefixed by project (e.g. cobaltmesh_001.json), so an
    # alphabetical directory listing would process each project's documents
    # as one contiguous block, defeating the interleaved timeline the
    # generator staggered on purpose. corpus.json preserves true
    # chronological (timestamp-sorted) order.
    with open(corpus_path, "r") as f:
        docs = json.load(f)

    docs_with_zero_relations = []

    for idx, doc in enumerate(docs, 1):
        raw_text = doc.get("content", "")

        relations = extract_relations_streaming(raw_text)

        doc_id = doc.get("doc_id", f"doc_{idx}")
        print(f"[EXTRACTION] doc={doc_id} -> "
              f"{relations if relations else '[]  <-- NO RELATIONS EXTRACTED'}")
        print(f"  source text: {raw_text}")

        if not relations:
            docs_with_zero_relations.append(doc_id)

        for src, tgt in relations:
            engine.insert_edge(src, tgt)
            chunk_store.add_extraction(src, tgt, raw_text)

        if idx % 10 == 0:
            print(f"[PROGRESS] {idx}/{len(docs)} documents processed | "
                  f"Nodes={len(engine.node_degrees)}, Edges={len(engine.edges)}")

    print(f"[STREAMING COMPLETE] Active Graph State: Nodes={len(engine.node_degrees)}, Edges={len(engine.edges)}")
    print(f"[EXTRACTION SUMMARY] {len(docs_with_zero_relations)}/{len(docs)} documents produced "
          f"zero relations ({len(docs_with_zero_relations) / len(docs) * 100:.1f}% extraction failure rate)")
    if docs_with_zero_relations:
        print(f"  Zero-relation doc_ids: {docs_with_zero_relations}")


if __name__ == "__main__":
    run_pipeline(num_projects=5, noise_docs=25)