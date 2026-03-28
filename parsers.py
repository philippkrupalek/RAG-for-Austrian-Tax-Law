"""
parsers.py

Parser for UStG, Anhang (Annex), and UStR

-) Each hierarchy level (section, paragraph, item, lit.) becomes its own chunk
-) Parent/child relationships are explicitly captured
-) Introductory sentences are preserved in the context of child chunks
-) UStR chunks include linked_ustg_refs (which sections are referenced)
"""

import re
import html
from typing import List, Dict, Optional, Tuple
from pathlib import Path

from models import (
    LegalChunk, LegalReference, ChunkStore,
    SourceType, ChunkLevel
)

# RTF 
try:
    from striprtf.striprtf import rtf_to_text
    RTF_AVAILABLE = True
except ImportError:
    RTF_AVAILABLE = False
    print("striprtf not available - pip install striprtf")

# HTML
try:
    from bs4 import BeautifulSoup
    HTML_AVAILABLE = True
except ImportError:
    HTML_AVAILABLE = False
    print("BeautifulSoup not available - pip install beautifulsoup4")


# UStG / ANHANG PARSER

class UStGParser:
    """Hierarchical parser for UStG 1994 and the Annex in RTF, Creates parent/child chunks. Hierarchy:
    Section X
    - para. Y
    -- item n
    --- lit. a
    Introductory-sentence: If a paragraph reads:
    "(2) The tax is reduced to 10% for the following supplies:
    1. Supplies of food
    2. Supplies of books"
    """
    
    # Patterns
    PAT_PARAGRAPH = re.compile(r'^§\s*(\d+[a-z]?)\.\s+(.*)', re.DOTALL)
    PAT_ARTIKEL = re.compile(r'^Art(?:ikel)?\.?\s+(\d+[a-z]?)\b[\.\s]*(.*)', re.DOTALL)
    # valid Absatz identifiers: § 19 Abs. 1a, § 3a Abs. 11a, § 11 Abs. 1a
    PAT_ABSATZ = re.compile(r'^\s*\((\d+[a-z]?)\)\s*(.*)', re.DOTALL)
    # (?!\d{4}[\.\s]) vPrevents matches before 4-digit years.
    PAT_ZIFFER = re.compile(r'^\s*(\d{1,2})\.\s+(?!\d{4}[\.\s])(.*)', re.DOTALL)
    PAT_LITERA = re.compile(r'^\s*([a-z]{1,2})\)\s+(.*)', re.DOTALL)
    PAT_SUB_LITERA = re.compile(r'^\s*(aa|bb|cc|dd|ee)\)\s+(.*)', re.DOTALL)
    
    def __init__(self, source_type: SourceType = SourceType.USTG):
        self.source_type = source_type
        self.store = ChunkStore()
        
        # parsing state
        self._current_para: Optional[str] = None
        self._current_abs: Optional[str] = None
        self._current_ziff: Optional[str] = None
        self._current_lit: Optional[str] = None
        self._current_title: str = ""
        
        # Accumulators for intro sentences
        self._para_intro_lines: List[str] = []
        self._abs_intro_lines: List[str] = []
        self._ziff_intro_lines: List[str] = []
        # Track last title line
        self._pending_title: Optional[str] = None
        # For appending continuation lines
        self._last_chunk_id: Optional[str] = None
    
    def parse(self, rtf_path: str) -> ChunkStore:
        """Parse RTF file and return ChunkStore"""
        if not RTF_AVAILABLE:
            print("striprtf required for RTF parsing")
            return self.store
        
        path = Path(rtf_path)
        if not path.exists():
            print(f"File not found: {rtf_path}")
            return self.store
        
        with open(rtf_path, 'r', encoding='utf-8', errors='ignore') as f:
            rtf_content = f.read()
        
        text = rtf_to_text(rtf_content)
        lines = text.split('\n')
        
        for line in lines:
            self._process_line(line)
        
        # build text_with_context for all chunks
        self._build_context_texts()
        # link parent/children
        self._link_hierarchy()
        
        return self.store
    
    def _process_line(self, line: str):
        """Process single line and update state"""
        stripped = line.strip()
        if not stripped:
            return
        
        # Try matching in order of hierarchy
        # §-Paragraph/Article
        match = self.PAT_PARAGRAPH.match(stripped) if self.source_type == SourceType.USTG else None
        if not match:
            match = self.PAT_ARTIKEL.match(stripped) if self.source_type == SourceType.ANHANG else None
        if not match and self.source_type == SourceType.USTG:
            match = self.PAT_ARTIKEL.match(stripped)  # UStG can also inclue "Art."
        
        if match:
            self._flush_intro_to_parent()
            para_num = match.group(1)
            rest = match.group(2).strip() if match.group(2) else ""
            if self.source_type == SourceType.ANHANG:
                para_num = f"Art{para_num}"
            self._current_para = para_num
            self._current_abs = None
            self._current_ziff = None
            self._current_lit = None
            self._current_title = self._pending_title or ""
            self._pending_title = None
            self._para_intro_lines = []
            self._abs_intro_lines = []
            self._ziff_intro_lines = []
            
            # Check if rest starts with "(1)". In this case split into §-level chunk (title/header only) + Abs.1 (the actual text).
            # without this "§ 12. (1) Der Unternehmer kann" would become a single chunk, and Abs.1 would never be created.
            abs1_match = self.PAT_ABSATZ.match(rest) if rest else None
            
            if abs1_match:
            # Create §-level chunk with just the title (no Abs.1 text)
                ref = self._make_ref()
                chunk = LegalChunk(
                    ref=ref,
                    text=self._current_title or f"§ {para_num}",
                    title=self._current_title,
                )
                self.store.add(chunk)
                self._last_chunk_id = chunk.chunk_id
                
            # Now process the "(1)" part as a separate Abs.1 chunk
                self._process_line(rest)
            else:
            # No "(1)" in rest, normal §-level chunk
                ref = self._make_ref()
                chunk = LegalChunk(
                    ref=ref,
                    text=rest if rest else stripped,
                    title=self._current_title,
                )
                self.store.add(chunk)
                self._last_chunk_id = chunk.chunk_id
                
                if rest:
                    self._para_intro_lines.append(rest)
            return

        # Absatz (1), (2) etc.
        match = self.PAT_ABSATZ.match(stripped)
        if match and self._current_para:
            self._flush_intro_to_parent()
            abs_num = match.group(1)
            rest = match.group(2).strip()

            self._current_abs = abs_num
            self._current_ziff = None
            self._current_lit = None
        # M5: _abs_intro_lines is reset when a new paragraph begins and contains, only the opening line of the paragraph (not all subsequent lines)
            self._ziff_intro_lines = []

            ref = self._make_ref()
            text = f"({abs_num}) {rest}"
            chunk = LegalChunk(
                ref=ref,
                text=text,
                title=self._current_title,
                parent_id=self._make_parent_ref().canonical_id,
            )
            self.store.add(chunk)
            self._last_chunk_id = chunk.chunk_id

            # Store only the paragraph opening line as the intro (no appending afterward)
            self._abs_intro_lines = [text]
            return
        
        # digits 1., 2. etc.
        match = self.PAT_ZIFFER.match(stripped)
        if match and self._current_abs:
            ziff_num = match.group(1)
            rest = match.group(2).strip()

            self._current_ziff = ziff_num
            self._current_lit = None
            self._ziff_intro_lines = []
            
            ref = self._make_ref()
            text = f"{ziff_num}. {rest}"
            chunk = LegalChunk(
                ref=ref,
                text=text,
                title=self._current_title,
                parent_id=self._make_parent_ref().canonical_id,
                intro_sentence=self._get_intro_sentence(self._abs_intro_lines),
            )
            self.store.add(chunk)
            self._last_chunk_id = chunk.chunk_id
            
            self._ziff_intro_lines.append(text)
            return
        
        # Litera a), b), etc.
        match = self.PAT_LITERA.match(stripped)
        if match and self._current_abs:
            lit = match.group(1)
            rest = match.group(2).strip()
            
            if self._current_ziff:
                # litera under ziffer (digit)
                self._current_lit = lit
                
                ref = self._make_ref()
                text = f"{lit}) {rest}"
                
                # Combine Absatz-Intro + Ziffer-Intro
                combined_intro = self._get_intro_sentence(self._abs_intro_lines)
                ziff_intro = self._get_intro_sentence(self._ziff_intro_lines)
                if ziff_intro:
                    combined_intro = f"{combined_intro} → {ziff_intro}" if combined_intro else ziff_intro
                
                chunk = LegalChunk(
                    ref=ref,
                    text=text,
                    title=self._current_title,
                    parent_id=self._make_parent_ref().canonical_id,
                    intro_sentence=combined_intro,
                )
                self.store.add(chunk)
                self._last_chunk_id = chunk.chunk_id
                return
            else:
                # FLitera directly under Absatz (no Ziffer), Some provisions go §X Abs.Y lit.a without a Ziffer (digit)
                self._current_ziff = None
                self._current_lit = lit
                
                # special reference (no ziffer level)
                ref = LegalReference(
                    source=self.source_type,
                    paragraph=self._current_para,
                    absatz=self._current_abs,
                    litera=lit,
                )
                text = f"{lit}) {rest}"
                chunk = LegalChunk(
                    ref=ref,
                    text=text,
                    title=self._current_title,
                    parent_id=self._make_abs_ref().canonical_id if self._current_abs else None,
                    intro_sentence=self._get_intro_sentence(self._abs_intro_lines),
                )
                self.store.add(chunk)
                self._last_chunk_id = chunk.chunk_id
                return
        
        # Title detection
        if (5 < len(stripped) < 120 and
            stripped[0].isupper() and
            not stripped.endswith('.') and
            not stripped.startswith('(') and
            '§' not in stripped and
            not re.match(r'^\d+\.?\s', stripped)):  # Not a Ziffer continuation
            self._pending_title = stripped
            return
        
        # append to the last chunk
        if self._last_chunk_id and self._current_para:
            chunk = self.store.get(self._last_chunk_id)
            if chunk:
                chunk.text += " " + stripped
                if self._current_ziff and not self._current_lit:
                    self._ziff_intro_lines.append(stripped)
    
    def _make_ref(self) -> LegalReference:
        """Create reference from current state"""
        return LegalReference(
            source=self.source_type,
            paragraph=self._current_para,
            absatz=self._current_abs,
            ziffer=self._current_ziff,
            litera=self._current_lit,
        )
    
    def _make_parent_ref(self) -> LegalReference:
        """Create reference for parent of current state"""
        if self._current_lit:
            return LegalReference(
                source=self.source_type,
                paragraph=self._current_para,
                absatz=self._current_abs,
                ziffer=self._current_ziff,
            )
        if self._current_ziff:
            return LegalReference(
                source=self.source_type,
                paragraph=self._current_para,
                absatz=self._current_abs,
            )
        if self._current_abs:
            return LegalReference(
                source=self.source_type,
                paragraph=self._current_para,
            )
        return LegalReference(
            source=self.source_type,
            paragraph=self._current_para,
        )
    
    def _make_abs_ref(self) -> LegalReference:
        """Create reference for current Absatz"""
        return LegalReference(
            source=self.source_type,
            paragraph=self._current_para,
            absatz=self._current_abs,
        )
    
    def _flush_intro_to_parent(self):
        """Save intro text to parent chunks before starting new section"""
        pass  
    
    def _get_intro_sentence(self, lines: List[str]) -> str:
        """
        Extract the intro sentence from lines, takes the first line and truncates
        """
        if not lines:
            return ""
        intro = lines[0]
        # truncate to first 200 chars to keep manageable
        if len(intro) > 200:
            intro = intro[:200] + "..."
        return intro
    
    def _build_context_texts(self):
        """
        text_with_context für jeden Chunk aufbauen:
        - § chunks: nur ihr Text
        - Abs chunks: §-Titel + §-Eröffnungstext + Abs-Text (M6)
        - Z chunks: Abs-Intro + Z-Text
        - lit chunks: Abs-Intro + Z-Intro + lit-Text
        """
        for chunk in self.store.all_chunks():
            if chunk.source_type not in (SourceType.USTG, SourceType.ANHANG):
                continue

            level = chunk.ref.level

            if level == ChunkLevel.PARAGRAPH:
                chunk.text_with_context = chunk.text

            elif level == ChunkLevel.ABSATZ:
                # Abs. Chunks get §-title and § opening text as context
                parent = self.store.get(chunk.parent_id) if chunk.parent_id else None
                if parent:
                    parent_header = parent.title or ""
                    parent_intro = parent.text[:120].strip() if parent.text else ""
                    if parent_header and parent_intro and parent_header not in parent_intro:
                        ctx_prefix = f"{parent_header}: {parent_intro}"
                    elif parent_intro:
                        ctx_prefix = parent_intro
                    elif parent_header:
                        ctx_prefix = parent_header
                    else:
                        ctx_prefix = ""
                    
                    # If Abs. text is nearly empty (e.g. "(1) " from §11), pull first children's text
                    abs_text = chunk.text
                    if len(abs_text.strip()) < 10:
                        children = self.store.get_children(chunk.chunk_id)
                        if children:
                            child_texts = " ".join(c.text[:150] for c in children[:3])
                            abs_text = f"{abs_text.strip()} {child_texts}"
                    
                    chunk.text_with_context = f"{ctx_prefix} → {abs_text}" if ctx_prefix else abs_text
                else:
                    chunk.text_with_context = chunk.text

            elif level in (ChunkLevel.ZIFFER, ChunkLevel.LITERA):
                if chunk.intro_sentence:
                    chunk.text_with_context = f"{chunk.intro_sentence} → {chunk.text}"
                else:
                    chunk.text_with_context = chunk.text
    
    def _link_hierarchy(self):
        """Set children_ids on parent chunks"""
        for chunk in self.store.all_chunks():
            if chunk.parent_id:
                parent = self.store.get(chunk.parent_id)
                if parent and chunk.chunk_id not in parent.children_ids:
                    parent.children_ids.append(chunk.chunk_id)


# UStR parser

class UStRParser:
    """Parser for UStR 2000 XML. Goal: Extraction of linked_ustg_refs. 
    Which sections of the UStG are mentioned in this marginal number? Enables targeted UStR assignment."""
    
    # extract paragraph references from UStR text
    PAT_USTG_REF = re.compile(
        r'§\s*(\d+[a-z]?)(?:\s+Abs\.?\s*(\d+))?(?:\s+Z\.?\s*(\d+))?(?:\s+lit\.?\s*([a-z]))?',
        re.IGNORECASE
    )
    
    PAT_SECTION = re.compile(r'^(\d+[a-z]?)')
    
    def parse(self, xml_path: str) -> ChunkStore:
        """Parse UStR XML and return ChunkStore"""
        store = ChunkStore()
        
        if not HTML_AVAILABLE:
            print("BeautifulSoup required")
            return store
        
        path = Path(xml_path)
        if not path.exists():
            print(f"File not found: {xml_path}")
            return store
        
        try:
            with open(xml_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            chunks = self._parse_content(content)
            for chunk in chunks:
                store.add(chunk)
            
        except Exception as e:
            print(f"UStR parse error: {e}")
            import traceback
            traceback.print_exc()
        
        return store
    
    @staticmethod
    def _truncate_at_sentence(text: str, max_chars: int = 3000) -> str:
        """Cut at sentence boundaries instead of a fixed character limit. Split on “. ” and keep full sentences within max_chars.
        If no sentence boundary is found, fall back to a hard cut."""
        
        if len(text) <= max_chars:
            return text
        # finding the last '. ' in max_chars
        cutoff = text.rfind('. ', 0, max_chars)
        if cutoff > 0:
            return text[:cutoff + 1]  #include period
        return text[:max_chars]  # fallback

    def _parse_content(self, content: str) -> List[LegalChunk]:
        """Parse XML content into chunks"""
        chunks = []
        
        # Regex parsing 
        segment_pattern = re.compile(
            r'<segbez>([^<]*)</segbez>.*?<txt>([^<]*(?:<(?!/txt>)[^<]*)*)</txt>',
            re.DOTALL
        )
        
        rz_div_pattern = re.compile(
            r'<div class="Randzahl" randzahl="(\d+)"[^>]*>\d+</div>(.*?)(?=<div class="Randzahl"|<h[45]>|$)',
            re.DOTALL
        )
        
        for match in segment_pattern.finditer(content):
            title = match.group(1).strip()
            txt_content = match.group(2)
            
            if not title or not txt_content:
                continue
            
            # To find paragraph number, trying multiple title patterns.
            # -) 3a Leistungsort...", r'^(\d+[a-z]?)'
            # -) "Zu § 3a UStG", fallback and search
            # -) "§ 3a UStG", fallback and search
            
            para_match = self.PAT_SECTION.match(title)
            if para_match:
                para_num = para_match.group(1)
            else:
                par_fallback = re.search(r'§\s*(\d+[a-z]?)', title)
                para_num = par_fallback.group(1) if par_fallback else "0"
            
            try:
                txt_unescaped = html.unescape(txt_content)
            except:
                txt_unescaped = txt_content
            
            # Extract judication references
            judikatur = re.findall(r'(VwGH|EuGH|BFH)[^<]{5,50}', txt_unescaped)[:5]
            
            rz_matches = list(rz_div_pattern.finditer(txt_unescaped))
            
            if rz_matches:
                for rz_match in rz_matches:
                    rz_num = rz_match.group(1)
                    rz_content = rz_match.group(2)

                    clean_text = re.sub(r'<[^>]+>', ' ', rz_content)
                    clean_text = re.sub(r'\s+', ' ', clean_text).strip()

                    if len(clean_text) > 30:
                        linked_refs = self._extract_ustg_refs(clean_text)
                        truncated = self._truncate_at_sentence(clean_text)

    # hierarchical context for UStR, similar to UStG chunks. Before: "[Regarding § 12 UStG] para. 1234: text" Now: section title + § + para. as full context.
    # This helps the reranker compare UStG and UStR more fairly.
                        
                        first_sentence = truncated.split('.')[0].strip()
                        if len(first_sentence) > 100:
                            first_sentence = first_sentence[:100]
                        context_header = f"[{title} | Zu § {para_num} UStG | Rz {rz_num}]"
                        text_with_ctx = f"{context_header} {truncated}"

                        ref = LegalReference(
                            source=SourceType.USTR,
                            paragraph=para_num,
                            randzahl=rz_num,
                        )

                        chunk = LegalChunk(
                            ref=ref,
                            text=truncated,
                            text_with_context=text_with_ctx,
                            title=title,
                            judikatur=judikatur,
                            linked_ustg_refs=linked_refs,
                        )
                        chunks.append(chunk)
            else:
                clean_text = re.sub(r'<[^>]+>', ' ', txt_unescaped)
                clean_text = re.sub(r'\s+', ' ', clean_text).strip()

                if len(clean_text) > 50:
                    linked_refs = self._extract_ustg_refs(clean_text)
                    truncated = self._truncate_at_sentence(clean_text)

                    # same principal for Chunks without Rz-division
                    context_header = f"[{title} | Zu § {para_num} UStG]"
                    text_with_ctx = f"{context_header} {truncated}"

                    ref = LegalReference(
                        source=SourceType.USTR,
                        paragraph=para_num,
                    )

                    chunk = LegalChunk(
                        ref=ref,
                        text=truncated,
                        text_with_context=text_with_ctx,
                        title=title,
                        judikatur=judikatur,
                        linked_ustg_refs=linked_refs,
                    )
                    chunks.append(chunk)
        
        return chunks
    
    def _extract_ustg_refs(self, text: str) -> List[str]:
        """Extract UStG paragraph references from UStR text at Abs granularity.
        Examples:
            "§ 12 Abs. 1 Z 1 UStG" = ["12", "12.1"]
            "§ 4 und § 6 UStG" → ["4", "6"]"""
        
        refs: set = set()
        for match in self.PAT_USTG_REF.finditer(text):
            para = match.group(1)
            refs.add(para)  # add plain paragraph
            if match.group(2):  # Abs present
                refs.add(f"{para}.{match.group(2)}")
        return sorted(list(refs))


# combined parser

def parse_all_sources(
    ustg_path: Optional[str] = None,
    anhang_path: Optional[str] = None,
    ustr_path: Optional[str] = None,
) -> ChunkStore:
    """
    Parse all sources and return a unified ChunkStore.
    -) ustg_path: Path to UStG RTF file
    -) anhang_path: Path to Anhang RTF file  
    -) ustr_path: Path to UStR XML file
    Returns a chunkstore with all chunks indexed
    """
    combined_store = ChunkStore()
    
    # Parse UStG
    if ustg_path and Path(ustg_path).exists():
        print("Parsing UStG 1994...")
        parser = UStGParser(source_type=SourceType.USTG)
        ustg_store = parser.parse(ustg_path)
        for chunk in ustg_store.all_chunks():
            combined_store.add(chunk)
        stats = ustg_store.stats()
        print(f"   {stats['total']} chunks | Levels: {stats['by_level']}")
    
    # Parse Anhang
    if anhang_path and Path(anhang_path).exists():
        print("Parsing Anhang (Binnenmarkt)...")
        parser = UStGParser(source_type=SourceType.ANHANG)
        anhang_store = parser.parse(anhang_path)
        for chunk in anhang_store.all_chunks():
            combined_store.add(chunk)
        stats = anhang_store.stats()
        print(f"   {stats['total']} chunks | Levels: {stats['by_level']}")
    
    # Parse UStR
    if ustr_path and Path(ustr_path).exists():
        print("Parsing UStR 2000...")
        parser = UStRParser()
        ustr_store = parser.parse(ustr_path)
        for chunk in ustr_store.all_chunks():
            combined_store.add(chunk)
        print(f"   {ustr_store.size} chunks")
        
        # Report linked refs stats
        linked_count = sum(1 for c in ustr_store.all_chunks() if c.linked_ustg_refs)
        total_links = sum(len(c.linked_ustg_refs) for c in ustr_store.all_chunks())
        print(f"   {linked_count} UStR-Chunks mit §-Verknüpfungen ({total_links} Links)")
    
    print(f"\n Overall: {combined_store.size} Chunks")
    print(f"   {combined_store.stats()}")
    
    # merge short chunks into parents
    merged = _merge_short_chunks(combined_store)
    if merged > 0:
        print(f"   {merged} kurze Chunks (<50 chars) in Parent-Chunks integriert")
        print(f"   After merge: {combined_store.size} Chunks")
    
    return combined_store


# minimum text length for a chunk to exist as a single chunk
MIN_CHUNK_LENGTH = 50


def _merge_short_chunks(store: ChunkStore, min_length: int = MIN_CHUNK_LENGTH) -> int:
    """Merge short chunks (<min_length chars) into their parent.Short chunks (e.g. “5. die Datenverarbeitung;”) lack context and match too broadly. 
    Merging them into their parent keeps the info but improves retrieval. Only applies to UStG/Annex chunks (UStR already has a minimum length). It then returns Number of merged chunks."""
    
    merged_count = 0
    
    # Collect short chunks
    short_chunks = [
        c for c in store.all_chunks()
        if (len(c.text.strip()) < min_length 
            and c.parent_id 
            and c.source_type in (SourceType.USTG, SourceType.ANHANG)
            and c.ref.level != ChunkLevel.ABSATZ)  # don't merge Abs chunks
    ]
    
    for chunk in short_chunks:
        parent = store.get(chunk.parent_id)
        if not parent:
            continue
        
        # Append short chunk's text to parent
        parent.text += "\n" + chunk.text
        
        # update parent's text_with_context
        if parent.text_with_context:
            parent.text_with_context += "\n" + chunk.text
        
        # remove the short chunk
        store.remove(chunk.chunk_id)
        
        # also remove from parent's children_ids
        if chunk.chunk_id in parent.children_ids:
            parent.children_ids.remove(chunk.chunk_id)
        
        merged_count += 1
    
    return merged_count