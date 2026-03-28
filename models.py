"""
models.py - data models für UStG RAG

data structures:
-) LegalReference: Zitierbare Referenz (§ X Abs. Y Z n lit. a)
-) LegalChunk: Chunk with Parent/Child-relationships
-) RetrievalResult: result with score + source
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Set
from enum import Enum


class SourceType(str, Enum):
    """Quellentyp"""
    USTG = "UStG"
    ANHANG = "Anhang"
    USTR = "UStR"


class ChunkLevel(str, Enum):
    """hierarchy of a chunk"""
    PARAGRAPH = "paragraph"   # § X
    ABSATZ = "absatz"         # § X Abs. Y
    ZIFFER = "ziffer"         # § X Abs. Y Z n
    LITERA = "litera"         # § X Abs. Y Z n lit. a


@dataclass
class LegalReference:
    """
    Structured reference to a provision of the UStG.
    
    Examples:
        § 12 Abs. 2 Z 2 lit. a UStG 1994
        Art. 1 Abs. 1 Anhang UStG 1994
        UStR 2000 Rz 1501
    """
    source: SourceType
    paragraph: str                    # "12", "Art1"
    absatz: Optional[str] = None      # "2"
    ziffer: Optional[str] = None      # "2"
    litera: Optional[str] = None      # "a"
    randzahl: Optional[str] = None    # "1501" (UStR only)
    
    @property
    def level(self) -> ChunkLevel:
        if self.litera:
            return ChunkLevel.LITERA
        if self.ziffer:
            return ChunkLevel.ZIFFER
        if self.absatz:
            return ChunkLevel.ABSATZ
        return ChunkLevel.PARAGRAPH
    
    @property
    def canonical_id(self) -> str:
        """
        Unique ID for Dedup und Lookup.
        Format: "ustg:12:2:2:a" or "ustr:rz1501"
        """
        prefix = self.source.value.lower()
        if self.source == SourceType.USTR and self.randzahl:
            return f"{prefix}:rz{self.randzahl}"
        
        parts = [prefix, self.paragraph]
        if self.absatz:
            parts.append(self.absatz)
        if self.ziffer:
            parts.append(self.ziffer)
        if self.litera:
            parts.append(self.litera)
        return ":".join(parts)
    
    @property
    def parent_id(self) -> Optional[str]:
        """ID of the higher-level chunk"""
        prefix = self.source.value.lower()
        if self.litera:
            return f"{prefix}:{self.paragraph}:{self.absatz}:{self.ziffer}"
        if self.ziffer:
            return f"{prefix}:{self.paragraph}:{self.absatz}"
        if self.absatz:
            return f"{prefix}:{self.paragraph}"
        return None  # § has no parent
    
    @property
    def citation(self) -> str:
        """human-readable citation"""
        if self.source == SourceType.USTR:
            if self.randzahl:
                return f"UStR 2000 Rz {self.randzahl}"
            return f"UStR 2000 § {self.paragraph}"
        
        parts = []
        if self.source == SourceType.ANHANG:
            parts.append(f"Art. {self.paragraph.replace('Art', '')}")
        else:
            parts.append(f"§ {self.paragraph}")
        
        if self.absatz:
            parts.append(f"Abs. {self.absatz}")
        if self.ziffer:
            parts.append(f"Z {self.ziffer}")
        if self.litera:
            parts.append(f"lit. {self.litera}")
        
        if self.source == SourceType.ANHANG:
            parts.append("Anhang UStG 1994")
        else:
            parts.append("UStG 1994")
        
        return " ".join(parts)
    
    def is_ancestor_of(self, other: LegalReference) -> bool:
        """Check if  self is an ancestor of other"""
        if self.source != other.source or self.paragraph != other.paragraph:
            return False
        
        # § is the parent of everything below it!
        if self.level == ChunkLevel.PARAGRAPH:
            return other.level != ChunkLevel.PARAGRAPH
        
        if self.level == ChunkLevel.ABSATZ and self.absatz == other.absatz:
            return other.level in (ChunkLevel.ZIFFER, ChunkLevel.LITERA)
        
        if (self.level == ChunkLevel.ZIFFER and 
            self.absatz == other.absatz and 
            self.ziffer == other.ziffer):
            return other.level == ChunkLevel.LITERA
        
        return False


@dataclass
class LegalChunk:
    """A chunk with complete hierarchical information. Core principle:
    each chunk contains its own text and the introductory sentence of the parent paragraph/item, so that context is never lost. """
    ref: LegalReference
    text: str                           #  own chunk text
    text_with_context: str = ""         # Text + introductory of the parent
    title: str = ""                     # heading/title of the paragraph
    
    # hierarchy
    parent_id: Optional[str] = None        # canonical_id of the parent
    children_ids: List[str] = field(default_factory=list)
    
    # introductory sentence
    intro_sentence: str = ""
    
    # UStR-specific
    judikatur: List[str] = field(default_factory=list)
    linked_ustg_refs: List[str] = field(default_factory=list)  # §§ that are mentioned in this UStR chunk
    
    @property
    def chunk_id(self) -> str:
        return self.ref.canonical_id
    
    @property
    def source_type(self) -> SourceType:
        return self.ref.source
    
    @property
    def citation(self) -> str:
        return self.ref.citation
    
    @property
    def search_text(self) -> str:
        """Text for Embedding/BM25 enriched with context"""
        if self.text_with_context:
            return self.text_with_context
        return self.text
    
    def to_context_string(self) -> str:
        """Formatting chunk for LLM context"""
        parts = [self.citation]
        if self.title:
            parts.append(f"({self.title})")
        parts.append(f"\n{self.text_with_context or self.text}")
        if self.judikatur:
            parts.append(f"\nJudication: {', '.join(self.judikatur[:3])}")
        return " ".join(parts)


@dataclass
class RetrievalResult:
    """A search result with scores"""
    chunk: LegalChunk
    dense_score: float = 0.0       # Cosine similarity (BGE-M3)
    sparse_score: float = 0.0      # BM25 score
    rerank_score: float = 0.0      # Cross-encoder score
    combined_score: float = 0.0    # Final weighted score
    boost_reason: str = ""         # boost reasong ("explicit_ref", "linked_ustr", etc.)
    
    @property
    def chunk_id(self) -> str:
        return self.chunk.chunk_id


@dataclass
class ChunkStore:
    """Central storage of all chunks with indexed access. Enables fast lookup by ID, §, source, etc."""
    chunks: Dict[str, LegalChunk] = field(default_factory=dict)
    
    # Indices
    _by_paragraph: Dict[str, List[str]] = field(default_factory=lambda: {})
    _by_source: Dict[SourceType, List[str]] = field(default_factory=lambda: {})
    _children: Dict[str, List[str]] = field(default_factory=lambda: {})
    
    def add(self, chunk: LegalChunk):
        cid = chunk.chunk_id
        self.chunks[cid] = chunk
        
    # Index by paragraph
        para = chunk.ref.paragraph
        if para not in self._by_paragraph:
            self._by_paragraph[para] = []
        self._by_paragraph[para].append(cid)
        
    # Index by source
        src = chunk.source_type
        if src not in self._by_source:
            self._by_source[src] = []
        self._by_source[src].append(cid)
        
    # Index parent - children
        if chunk.parent_id:
            if chunk.parent_id not in self._children:
                self._children[chunk.parent_id] = []
            self._children[chunk.parent_id].append(cid)
    
    def remove(self, chunk_id: str):
        """Remove a chunk and clean up all indices"""
        chunk = self.chunks.pop(chunk_id, None)
        if not chunk:
            return
        # Clean paragraph index
        para = chunk.ref.paragraph
        if para in self._by_paragraph:
            self._by_paragraph[para] = [i for i in self._by_paragraph[para] if i != chunk_id]
        # Clean source index
        src = chunk.source_type
        if src in self._by_source:
            self._by_source[src] = [i for i in self._by_source[src] if i != chunk_id]
        # Clean children index
        if chunk.parent_id and chunk.parent_id in self._children:
            self._children[chunk.parent_id] = [
                i for i in self._children[chunk.parent_id] if i != chunk_id
            ]
    
    def get(self, chunk_id: str) -> Optional[LegalChunk]:
        return self.chunks.get(chunk_id)
    def get_by_paragraph(self, para: str) -> List[LegalChunk]:
        ids = self._by_paragraph.get(para, [])
        return [self.chunks[i] for i in ids if i in self.chunks]
    def get_by_source(self, source: SourceType) -> List[LegalChunk]:
        ids = self._by_source.get(source, [])
        return [self.chunks[i] for i in ids if i in self.chunks]
    def get_children(self, chunk_id: str) -> List[LegalChunk]:
        ids = self._children.get(chunk_id, [])
        return [self.chunks[i] for i in ids if i in self.chunks]
    
    def get_parent(self, chunk_id: str) -> Optional[LegalChunk]:
        chunk = self.chunks.get(chunk_id)
        if chunk and chunk.parent_id:
            return self.chunks.get(chunk.parent_id)
        return None
    
    def get_with_ancestors(self, chunk_id: str) -> List[LegalChunk]:
        """Retrieves a chunk plus all its parent elements (for context construction)."""
        result = []
        current_id = chunk_id
        while current_id:
            chunk = self.chunks.get(current_id)
            if not chunk:
                break
            result.append(chunk)
            current_id = chunk.parent_id
        return list(reversed(result))  # from § to Abs to Z to lit
    
    def all_chunks(self) -> List[LegalChunk]:
        return list(self.chunks.values())
    
    @property
    def size(self) -> int:
        return len(self.chunks)
    
    def stats(self) -> Dict:
        by_source = {}
        by_level = {}
        for chunk in self.chunks.values():
            src = chunk.source_type.value
            by_source[src] = by_source.get(src, 0) + 1
            lvl = chunk.ref.level.value
            by_level[lvl] = by_level.get(lvl, 0) + 1
        return {"total": self.size, "by_source": by_source, "by_level": by_level}
