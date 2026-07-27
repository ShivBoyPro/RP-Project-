import hashlib
from typing import Dict, Set, Tuple, Optional, List

class BoundedChunkStore:
    def __init__(self):
        # Maps chunk_id -> (text, timestamp)
        self.chunks: Dict[str, Tuple[str, Optional[str]]] = {}
        
        # Maps graph edges (src, tgt) -> set of chunk_ids
        self.edge_to_chunks: Dict[Tuple[str, str], Set[str]] = {}

    def _hash_text(self, text: str) -> str:
        """Generate a deterministic hash ID for a text chunk."""
        return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]

    def add_extraction(self, src: str, tgt: str, text: str, timestamp: Optional[str] = None) -> str:
        """Add an extracted text chunk linked to a specific edge, storing its timestamp."""
        chunk_id = self._hash_text(text)
        
        # Store chunk text and timestamp if not already present
        if chunk_id not in self.chunks:
            self.chunks[chunk_id] = (text, timestamp)
            
        # Map the edge to the chunk ID
        edge_key = (src, tgt)
        if edge_key not in self.edge_to_chunks:
            self.edge_to_chunks[edge_key] = set()
        self.edge_to_chunks[edge_key].add(chunk_id)
        
        return chunk_id

    def get_chunks_for_edge(self, edge: Tuple[str, str]) -> List[Tuple[str, Optional[str]]]:
        """Retrieve all text chunks and timestamps associated with a given edge."""
        chunk_ids = self.edge_to_chunks.get(edge, set())
        return [self.chunks[cid] for cid in chunk_ids if cid in self.chunks]

    def remove_edge(self, edge: Tuple[str, str]):
        """Clean up edge-to-chunk mappings when an edge is evicted from memory."""
        if edge in self.edge_to_chunks:
            del self.edge_to_chunks[edge]