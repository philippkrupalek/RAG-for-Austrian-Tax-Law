"""
retriever.py _ Hybrid Retrieval für UStG RAG

-) Reference Extraction: §§/Art from query
-) Candidate Retrieval: Dense (BGE-M3) Top-50 + BM25 Top-50
-) Boost if chunk_id matches reference, Boost if UStR linked_ustg_paragraphs includes question-§§
-) Rerank: Cross-Encoder Top-50 - Top 12-18
-) Source Balancing: UStG primary, UStR bridge, Anhang conditional
"""

import re
import json
import math
import numpy as np
from typing import List, Dict, Optional, Set, Tuple
from collections import defaultdict
from dataclasses import dataclass

from models import (
    LegalChunk, LegalReference, ChunkStore, RetrievalResult,
    SourceType, ChunkLevel
)

# reference extractor

class ReferenceExtractor:
    """Extracts only explicit § and Art references from queries with no guessing, no keyword mapping, retrieval system handles semantics."""
    
    PAT_PARA = re.compile(r'§\s*(\d+[a-z]?)', re.IGNORECASE)
    PAT_ART = re.compile(r'Art(?:ikel)?\.?\s*(\d+[a-z]?)', re.IGNORECASE)
    PAT_ABS = re.compile(r'Abs(?:atz)?\.?\s*(\d+)', re.IGNORECASE)
    PAT_ZIFF = re.compile(r'Z(?:iffer)?\.?\s*(\d+)', re.IGNORECASE)
    PAT_LIT = re.compile(r'lit(?:era)?\.?\s*([a-z])', re.IGNORECASE)
    PAT_RZ = re.compile(r'Rz\.?\s*(\d+)', re.IGNORECASE)
    
    @staticmethod
    def extract(query: str) -> Dict:
        result = {
            'paragraphs': [],
            'artikel': [],
            'details': [],
            'randzahlen': [],
        }
        
        for m in ReferenceExtractor.PAT_PARA.finditer(query):
            para = m.group(1)
            if para not in result['paragraphs']:
                result['paragraphs'].append(para)
        
        for m in ReferenceExtractor.PAT_ART.finditer(query):
            art = f"Art{m.group(1)}"
            if art not in result['artikel']:
                result['artikel'].append(art)
        
        for m in ReferenceExtractor.PAT_RZ.finditer(query):
            rz = m.group(1)
            if rz not in result['randzahlen']:
                result['randzahlen'].append(rz)
        
        detail_pattern = re.compile(
            r'§\s*(\d+[a-z]?)'
            r'(?:\s+Abs(?:atz)?\.?\s*(\d+[a-z]?))?'
            r'(?:\s+Z(?:iffer)?\.?\s*(\d+))?'
            r'(?:\s+lit(?:era)?\.?\s*([a-z]))?',
            re.IGNORECASE
        )
        
        for m in detail_pattern.finditer(query):
            para = m.group(1)
            detail = {
                'para': para,
                'abs': m.group(2),
                'ziff': m.group(3),
                'lit': m.group(4),
            }
            if detail['abs'] is not None:
                result['details'].append(detail)
        
        return result
    
    @staticmethod
    def is_eu_relevant(query: str) -> bool:
        query_lower = query.lower()
        
        exact_keywords = ['eu']
        for kw in exact_keywords:
            if re.search(r'\b' + re.escape(kw) + r'\b', query_lower):
                return True
        
        substring_keywords = [
            'europa', 'europäisch', 'innergemeinschaftlich',
            'ig-erwerb', 'ig-lieferung', 'ig erwerb', 'ig lieferung',
            'binnenmarkt', 'gemeinschaftsgebiet',
            'mitgliedstaat', 'grenzüberschreitend', 'uid-nummer', 'uid nummer',
            'versandhandel', 'fernverkauf', 'dreiecksgeschäft',
            'konsignationslager', 'anhang',
        ]
        
        for kw in substring_keywords:
            if kw in query_lower:
                return True
        
        if re.search(r'art(?:ikel)?\.?\s*\d', query_lower):
            return True
        
        return False


# BM25 index

class BM25Index:
    """BM25 implementation for lexical matching."""
    
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.doc_freqs = {}
        self.doc_lens = []
        self.avg_dl = 0.0
        self.n_docs = 0
        self.tf_matrix = []
        self.chunk_ids = []
        self._tokenize_cache = {}
    
    def build(self, chunks: List[LegalChunk]):
        self.n_docs = len(chunks)
        self.doc_freqs = defaultdict(int)
        self.tf_matrix = []
        self.doc_lens = []
        self.chunk_ids = []
        
        for chunk in chunks:
            tokens = self._tokenize(chunk.search_text)
            tf = defaultdict(int)
            for token in tokens:
                tf[token] += 1
            
            self.tf_matrix.append(dict(tf))
            self.doc_lens.append(len(tokens))
            self.chunk_ids.append(chunk.chunk_id)
            
            for term in set(tokens):
                self.doc_freqs[term] += 1
        
        self.avg_dl = sum(self.doc_lens) / max(self.n_docs, 1)
    
    def search(self, query: str, top_k: int = 50) -> List[Tuple[str, float]]:
        tokens = self._tokenize(query)
        scores = []
        
        for i in range(self.n_docs):
            score = 0.0
            tf = self.tf_matrix[i]
            dl = self.doc_lens[i]
            
            for term in tokens:
                if term not in tf:
                    continue
                
                term_tf = tf[term]
                df = self.doc_freqs.get(term, 0)
                
                idf = math.log((self.n_docs - df + 0.5) / (df + 0.5) + 1.0)
                
                tf_component = (term_tf * (self.k1 + 1)) / (
                    term_tf + self.k1 * (1 - self.b + self.b * dl / self.avg_dl)
                )
                
                score += idf * tf_component
            
            if score > 0:
                scores.append((self.chunk_ids[i], score))
        
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]
    
    _LEGAL_BIGRAMS = [
        'vorsteuerabzug', 'steuerschuld', 'reverse charge', 'innergemeinschaftlich',
        'steuerpflicht', 'steuerbefreiung', 'leistungsempfänger', 'leistungserbringer',
        'unrichtiger steuerausweis', 'unberechtigter steuerausweis',
        'steuerschuld kraft rechnungslegung', 'sollbesteuerung', 'istbesteuerung',
        'vorsteuerberichtigung', 'eigenverbrauch', 'bauleistung', 'subunternehmer',
        'dreiecksgeschäft', 'konsignationslager', 'fernverkauf', 'versandhandel',
    ]

    _STOPWORDS = frozenset({
        'z', 'a', 'zu', 'in', 'an', 'am', 'im', 'um', 'auf', 'ab', 'aus',
        'der', 'die', 'das', 'des', 'dem', 'den', 'ein', 'eine', 'einer',
        'ist', 'sind', 'wird', 'werden', 'war', 'hat', 'haben', 'kann',
        'und', 'oder', 'bzw', 'sowie', 'als', 'wie', 'für', 'von',
        'mit', 'bei', 'nach', 'vor', 'über', 'unter', 'durch', 'je', 'ob',
        'zu', 'lit', 'abs', 'rz',
    })

    def _tokenize(self, text: str) -> List[str]:
        text_lower = text.lower()

        remaining = text_lower
        for compound in sorted(self._LEGAL_BIGRAMS, key=len, reverse=True):
            if compound in remaining:
                remaining = remaining.replace(compound, compound.replace(' ', '_'))

        raw = re.findall(r'§\s*\d+[a-z]?|\d+[a-z]?|[a-zäöüß_]+', remaining)
        tokens = []
        for t in raw:
            t = re.sub(r'§\s*', '§', t)
            if t.startswith('§'):
                tokens.append(t)
            elif t not in self._STOPWORDS and len(t) > 1:
                tokens.append(t)
        return tokens

# hybrid retriever

class HybridRetriever:
    """Dense + BM25 + Rerank + Source Balancing"""
    
    DENSE_WEIGHT = 0.6
    SPARSE_WEIGHT = 0.4
    RERANK_WEIGHT = 0.3
    DENSE_WEIGHT_RERANKED = 0.4
    SPARSE_WEIGHT_RERANKED = 0.3
    
    EXPLICIT_REF_BOOST = 2.0
    EXACT_REF_BOOST = 3.0
    USTR_LINKED_BOOST = 1.5
    
    DENSE_CANDIDATES = 50
    BM25_CANDIDATES = 50
    RERANK_TOP_K = 50

    MAX_USTG = 15
    MAX_USTR = 4
    MAX_ANHANG = 4
    
    MIN_RERANK_SCORE = 0.1
    RERANK_CONTEXT_CHARS = 2048
    
    def __init__(self, chunk_store: ChunkStore, device: str = 'cpu'):
        self.store = chunk_store
        self.device = device
        self.ref_extractor = ReferenceExtractor()
        
        self.dense_index = None
        self.bm25_index = None
        self.reranker = None
        self.embeddings = None
        
        self._chunk_list = []
    
    def build(self, embedding_model: str = "BAAI/bge-m3"):
        """Build all indices (with disk caching)"""
        from langchain_community.embeddings import HuggingFaceEmbeddings
        from langchain_community.vectorstores import FAISS
        from langchain_core.documents import Document
        import hashlib
        import pickle
        from pathlib import Path
        
        all_chunks = self.store.all_chunks()
        if not all_chunks:
            print("No chunks")
            return
        
        hash_input = "|".join(
            f"{c.chunk_id}:{c.text[:80]}" for c in sorted(all_chunks, key=lambda c: c.chunk_id)
        )
        content_hash = hashlib.sha256(hash_input.encode()).hexdigest()[:16]
        
        cache_dir = Path(__file__).parent / "index_cache"
        cache_dir.mkdir(exist_ok=True)
        faiss_cache = cache_dir / "faiss_index"
        bm25_cache = cache_dir / "bm25_index.pkl"
        hash_file = cache_dir / "content_hash.txt"
        chunks_cache = cache_dir / "chunk_list.pkl"
        
        cache_valid = (
            hash_file.exists() and
            hash_file.read_text().strip() == content_hash and
            faiss_cache.exists() and
            bm25_cache.exists() and
            chunks_cache.exists()
        )
        
        print(f"\n Building hybrid retrieval ({len(all_chunks)} chunks)...")
        print(f"   Loading {embedding_model}...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={'device': self.device},
            encode_kwargs={
                'normalize_embeddings': True,
                'batch_size': 8,   # reduces VRAM down to 3-4GB 
            }
        )
        
        if cache_valid:
            print(f"   Loading cached indices (hash={content_hash})...")
            self.dense_index = FAISS.load_local(
                str(faiss_cache), self.embeddings,
                allow_dangerous_deserialization=True
            )
            with open(bm25_cache, 'rb') as f:
                self.bm25_index = pickle.load(f)
            with open(chunks_cache, 'rb') as f:
                self._chunk_list = pickle.load(f)
            print(f"   Loaded from cache: {len(self._chunk_list)} chunks")
        else:
            print(f"   Building dense index...")
            docs = []
            self._chunk_list = all_chunks
            for chunk in all_chunks:
                doc = Document(
                    page_content=chunk.search_text,
                    metadata={
                        'chunk_id': chunk.chunk_id,
                        'source': chunk.source_type.value,
                        'ref': chunk.citation,
                        'paragraph': chunk.ref.paragraph,
                    }
                )
                docs.append(doc)
            
            self.dense_index = FAISS.from_documents(docs, self.embeddings)
            print(f"   Dense index: {len(docs)} vectors")
            
            print(f"   Building BM25 index")
            self.bm25_index = BM25Index()
            self.bm25_index.build(all_chunks)
            print(f"   BM25 index: {self.bm25_index.n_docs} docs, {len(self.bm25_index.doc_freqs)} terms")
            
            print(f"   Saving indices to cache...")
            self.dense_index.save_local(str(faiss_cache))
            with open(bm25_cache, 'wb') as f:
                pickle.dump(self.bm25_index, f)
            with open(chunks_cache, 'wb') as f:
                pickle.dump(self._chunk_list, f)
            hash_file.write_text(content_hash)
            print(f"   Cache saved (hash={content_hash})")
        
        self._reranker_loaded = False
        print(f"   Hybrid retrieval system ready!")
    
    def _load_reranker(self):
        """Lazy-load cross-encoder reranker"""
        if self._reranker_loaded:
            return
        
        try:
            from sentence_transformers import CrossEncoder
            print("   Loading cross-encoder reranker..")
            self.reranker = CrossEncoder(
                'BAAI/bge-reranker-v2-m3',
                device=self.device,
                max_length=1024,
            )
            self._reranker_loaded = True
            print("   Reranker loaded")
        except Exception as e:
            print(f"   Reranker not available: {e}")
            print(f"   -> Falling back to score-only ranking")
            self._reranker_loaded = True
            self.reranker = None

    def retrieve(
        self,
        query: str,
        rewritten_query: Optional[str] = None,
        top_k: int = 22,
        use_reranking: bool = True,
        ref_query: Optional[str] = None,
        rerank_query: Optional[str] = None,
        force_dense_only: bool = False,   # only Dense (BGE-M3), BM25 deactivated
        force_sparse_only: bool = False,  # only nur BM25, Dense deactivated
    ) -> List[RetrievalResult]:
        """Full hybrid retrieval pipeline. Args:
        -) query: Search query (typically statement + question for better semantic search)
        -) rewritten_query: LLM-rewritten query (used for additional search pass)
        -) top_k: Number of results to return
        _) use_reranking: Enable cross-encoder reranking (S3+)
        -) ref_query: Used only for §-reference extraction (default: query). Pass case.question when query contains the full statement.
        -) rerank_query: Used only for cross-encoder reranker (default: ref_query or query). bge-reranker-v2-m3 is optimized for short queries, long statements as reranker query degrade scores.
        -) force_dense_only: dense retrieval only (BGE-M3), BM25 disabled
        -) force_sparse_only: BM25 retrieval only, dense disabled
        """
        print(f"\n{'='*60}")
        print(f"Query: '{query[:80]}{'...' if len(query) > 80 else ''}'")

        _ref_source = ref_query if ref_query else query
        refs = self.ref_extractor.extract(_ref_source)
        eu_relevant = self.ref_extractor.is_eu_relevant(_ref_source)

        _rerank_source = rerank_query if rerank_query else (_ref_source)

        explicit_paras = set(refs['paragraphs'] + refs['artikel'])
        print(f"   Explicit: §§ {refs['paragraphs']}, Art {refs['artikel']}, Rz {refs['randzahlen']}")
        print(f"   EU-relevant: {eu_relevant}")

        # Step 2: Dense retrieval
        dense_results = self._dense_search(query, k=self.DENSE_CANDIDATES)

        # Step 3: BM25 retrieval
        bm25_results = self._bm25_search(query, k=self.BM25_CANDIDATES)

        # enforce single modality
        if force_dense_only:
            bm25_results = {}   # BM25 off
        if force_sparse_only:
            dense_results = {}  # Dense iff

        # Step 3b: If rewritten query available, additional search runs
        if rewritten_query and rewritten_query != query:
            if not force_sparse_only:
                dense_rw = self._dense_search(rewritten_query, k=self.DENSE_CANDIDATES)
                for cid, score in dense_rw.items():
                    dense_results[cid] = max(dense_results.get(cid, 0.0), score)
            if not force_dense_only:
                bm25_rw = self._bm25_search(rewritten_query, k=self.BM25_CANDIDATES)
                for cid, score in bm25_rw.items():
                    bm25_results[cid] = max(bm25_results.get(cid, 0.0), score)

        # Step 4: Merge
        candidates = self._merge_candidates(dense_results, bm25_results)
        
        # Step 4a: Filter out §-level chunks (paragraph-only, no Abs/Z)
        # These contain only the section title and crowd out specific Abs/Z chunks.
        # Exception: UStR chunks are Randzahlen and always paragraph-level.
        pre_filter = len(candidates)
        candidates = [
            r for r in candidates
            if r.chunk.ref.level != ChunkLevel.PARAGRAPH
            or r.chunk.source_type == SourceType.USTR
        ]
        filtered = pre_filter - len(candidates)
        if filtered:
            print(f"   Filtered {filtered} §-level chunks (too coarse)")
        
        print(f"   Merged candidates: {len(candidates)}")

        # Then inject UStG chunks for explicit §-references
        MAX_INJECT_PER_PARA = 12
        if explicit_paras:
            candidate_ids = {r.chunk_id for r in candidates}
            injected = 0
            for para in explicit_paras:
                para_injected = 0
                # Sort by level with Abs first, then Z, then lit, so Abs-level chunks get injected before the limit is hit
                para_chunks = [c for c in self.store.get_by_paragraph(para)
                               if c.ref.level != ChunkLevel.PARAGRAPH
                               and c.source_type in (SourceType.USTG, SourceType.ANHANG)
                               and c.chunk_id not in candidate_ids]
                level_order = {ChunkLevel.ABSATZ: 0, ChunkLevel.ZIFFER: 1, ChunkLevel.LITERA: 2}
                para_chunks.sort(key=lambda c: level_order.get(c.ref.level, 3))
                
                for chunk in para_chunks:
                    if para_injected >= MAX_INJECT_PER_PARA:
                        break
                    result = RetrievalResult(
                        chunk=chunk,
                        dense_score=0.0,
                        sparse_score=0.0,
                        combined_score=0.1,
                    )
                    candidates.append(result)
                    candidate_ids.add(chunk.chunk_id)
                    injected += 1
                    para_injected += 1
            if injected:
                print(f"   💉 Injected {injected} UStG chunks (Abs+) for §§ {list(explicit_paras)}")

        # Step 4c: Topic-keyword injection (implicit § references) when query mentions a legal topic but no explicit § is given, inject all sub-chunks from the most likely paragraph. §3a has 16 Absätze
        if not explicit_paras:
            TOPIC_PARA_MAP = {
                'leistungsort': ['3a'],
                'ort der leistung': ['3a'],
                'ort der sonstigen leistung': ['3a'],
                'ort der lieferung': ['3'],
                'lieferort': ['3'],
                'steuerbefreiung': ['6'],
                'befreit': ['6'],
                'steuerfrei': ['6'],
                'bemessungsgrundlage': ['4'],
                'kleinunternehmer': ['6'],
                'reihengeschäft': ['3'],
                'werklieferung': ['3'],
                'eigenverbrauch': ['3a', '1'],
                'differenzbesteuerung': ['24'],
                'steuerschuldnerschaft': ['19'],
            }
            query_lower = query.lower()
            topic_paras = set()
            for keyword, paras in TOPIC_PARA_MAP.items():
                if keyword in query_lower:
                    topic_paras.update(paras)
            if topic_paras:
                candidate_ids = {r.chunk_id for r in candidates}
                topic_injected = 0
                for para in topic_paras:
                    # Include ALL sub-levels (Abs, Z, lit), bc §3a needs Abs.9, 11, 12, 13 which are deep in the hierarchy
                    para_chunks = [c for c in self.store.get_by_paragraph(para)
                                   if c.ref.level != ChunkLevel.PARAGRAPH
                                   and c.source_type in (SourceType.USTG, SourceType.ANHANG)
                                   and c.chunk_id not in candidate_ids]
                    level_order = {ChunkLevel.ABSATZ: 0, ChunkLevel.ZIFFER: 1, ChunkLevel.LITERA: 2}
                    para_chunks.sort(key=lambda c: level_order.get(c.ref.level, 3))
                    # §3a has 16 Absätze need at least 20 slots
                    max_inject = 20
                    for chunk in para_chunks[:max_inject]:
                        result_obj = RetrievalResult(
                            chunk=chunk,
                            dense_score=0.0,
                            sparse_score=0.0,
                            combined_score=0.05,
                        )
                        candidates.append(result_obj)
                        candidate_ids.add(chunk.chunk_id)
                        topic_injected += 1
                if topic_injected:
                    print(f"   🏷️ Topic-injected {topic_injected} chunks for §§ {list(topic_paras)}")

        # Step 5: Boost explicit refs + UStR coupling
        candidates = self._apply_boosts(candidates, refs, eu_relevant)

        # Step 5b: Keyword-based boost for § 3a Absätze
        candidates = self._apply_keyword_boost_3a(query, candidates)

        # Step 6: Reranking with Cross-Encoder when activated
        if use_reranking:
            candidates = self._rerank(_rerank_source, candidates)
        else:
            candidates.sort(key=lambda r: r.combined_score, reverse=True)

        # Step 7: Source balancing
        has_explicit_arts = bool(refs.get('artikel'))
        final = self._balance_sources(candidates, eu_relevant, top_k, has_explicit_arts=has_explicit_arts)

        # Step 7b: §-level expansion. Replace §-level UStG chunks (depth=1) with their Abs-level children. §-level chunks like "3a" or "12" NEVER match expected refs that have Abs
        # (e.g. "§ 3a Abs. 9") because retrieval matching requires depth >= 2.
        # So When a §-level chunk appears, swap it for its highest-scoring Abs-level child chunks, which do match and contain the actual legal text.
        final = self._expand_paragraph_chunks(final, candidates, top_k)

        # Step 8: Paragraph diversity deduplication
        # Problem was; multiple chunks from same §+Abs fill all top-K slots, blocking other paragraphs from appearing in Recall@K window.
        # To fix, cap same-paragraph chunks, keep best-scoring per Abs, push excess to the end so diverse paragraphs rank higher.
        final = self._deduplicate_by_paragraph(final, top_k)

        dist = defaultdict(int)
        for r in final:
            dist[r.chunk.source_type.value] += 1
        print(f"   Final: {dict(dist)} ({len(final)} total)")

        return final

    def backfill_paragraphs(
        self,
        missing_paras: Set[str],
        existing_results: List[RetrievalResult],
        query: str,
        max_per_para: int = 4,
        cited_refs: Optional[List] = None,
    ) -> List[RetrievalResult]:
        """2-Pass Backfill: Retrieve chunks for §§ that the LLM cited
        but that were missing from the original retrieval context."""
        existing_ids = {r.chunk_id for r in existing_results}
        backfill = []

        for para in missing_paras:
            chunks = self.store.get_by_paragraph(para)
            candidates = [c for c in chunks if c.chunk_id not in existing_ids]

            if not candidates:
                continue

            if cited_refs:
                precise_refs = [r for r in cited_refs if r.paragraph == para and r.absatz]
                if precise_refs:
                    exact_matches = []
                    rest = []
                    for c in candidates:
                        matched = False
                        for ref in precise_refs:
                            if (c.ref.absatz == ref.absatz and
                                (ref.ziffer is None or c.ref.ziffer == ref.ziffer) and
                                (ref.litera is None or c.ref.litera == ref.litera)):
                                exact_matches.append(c)
                                matched = True
                                break
                        if not matched:
                            rest.append(c)
                    candidates = exact_matches + rest

            if self.reranker and len(candidates) > max_per_para:
                pairs = [
                    (query, (c.text_with_context or c.search_text)[:self.RERANK_CONTEXT_CHARS])
                    for c in candidates
                ]
                try:
                    scores = self.reranker.predict(pairs)
                    scored = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
                    candidates = [c for c, s in scored[:max_per_para]]
                except Exception:
                    candidates = candidates[:max_per_para]
            else:
                candidates = candidates[:max_per_para]

            for chunk in candidates:
                result = RetrievalResult(
                    chunk=chunk,
                    dense_score=0.0,
                    sparse_score=0.0,
                    combined_score=0.05,
                    boost_reason="backfill_2pass",
                )
                backfill.append(result)
                existing_ids.add(chunk.chunk_id)

        return backfill

    def _dense_search(self, query: str, k: int) -> Dict[str, float]:
        """Dense similarity search, returns {chunk_id: score}"""
        if not self.dense_index:
            return {}

        results = self.dense_index.similarity_search_with_score(query, k=k)
        scores = {}
        for doc, score in results:
            cid = doc.metadata['chunk_id']
            similarity = max(0, 1 - score / 2)
            scores[cid] = similarity

        return scores

    def _bm25_search(self, query: str, k: int) -> Dict[str, float]:
        """BM25 lexical search, returns {chunk_id: score}"""
        if not self.bm25_index:
            return {}

        results = self.bm25_index.search(query, top_k=k)

        if not results:
            return {}

        max_score = results[0][1] if results[0][1] > 0 else 1.0
        return {cid: score / max_score for cid, score in results}

    def _merge_candidates(
        self,
        dense: Dict[str, float],
        sparse: Dict[str, float]
    ) -> List[RetrievalResult]:
        """Merge dense and sparse results into a unified candidate list."""
        all_ids = set(dense.keys()) | set(sparse.keys())

        candidates = []
        for cid in all_ids:
            chunk = self.store.get(cid)
            if not chunk:
                continue

            d_score = dense.get(cid, 0.0)
            s_score = sparse.get(cid, 0.0)

            combined = (
                self.DENSE_WEIGHT * d_score +
                self.SPARSE_WEIGHT * s_score
            )

            result = RetrievalResult(
                chunk=chunk,
                dense_score=d_score,
                sparse_score=s_score,
                combined_score=combined,
            )
            candidates.append(result)

        return candidates

    def _apply_boosts(
        self,
        candidates: List[RetrievalResult],
        refs: Dict,
        eu_relevant: bool,
    ) -> List[RetrievalResult]:
        """Apply boost factors for explicit refs and UStR coupling"""

        explicit_paras = set(refs['paragraphs'])
        explicit_arts = set(refs['artikel'])
        details = refs.get('details', [])

        for result in candidates:
            chunk = result.chunk
            boosts = []

            if chunk.source_type in (SourceType.USTG, SourceType.ANHANG):
                if chunk.ref.paragraph in explicit_paras:
                    result.combined_score *= self.EXPLICIT_REF_BOOST
                    boosts.append("explicit_§")

                if chunk.ref.paragraph in explicit_arts:
                    result.combined_score *= self.EXPLICIT_REF_BOOST
                    boosts.append("explicit_art")

            if chunk.source_type in (SourceType.USTG, SourceType.ANHANG):
                for detail in details:
                    if (chunk.ref.paragraph == detail['para'] and
                        chunk.ref.absatz == detail.get('abs') and
                        detail.get('abs') is not None):

                        if (chunk.ref.ziffer == detail.get('ziff') and
                            chunk.ref.litera == detail.get('lit') and
                            detail.get('ziff') is not None):
                            result.combined_score *= self.EXACT_REF_BOOST
                            boosts.append("exact_ref")
                        else:
                            result.combined_score *= 1.5
                            boosts.append("partial_ref")

            if chunk.source_type == SourceType.USTR and chunk.linked_ustg_refs:
                if explicit_paras & set(chunk.linked_ustg_refs):
                    result.combined_score *= self.USTR_LINKED_BOOST
                    boosts.append("ustr_linked")

            if (chunk.ref.randzahl and
                chunk.ref.randzahl in refs.get('randzahlen', [])):
                result.combined_score *= self.EXACT_REF_BOOST
                boosts.append("rz_match")

            if boosts:
                result.boost_reason = ", ".join(boosts)

        return candidates

    # § 3a Keyword Boost 
    KEYWORD_3A_BOOST = 2.5

    KEYWORD_3A_MAP = {
        '3a.6': {
            'keywords': ['sonstige leistung', 'empfängerort', 'beratung', 'werbeleistung',
                        'fernsehempfangsrecht', 'edv', 'datenverarbeitung', 'software',
                        'managementleistung', 'rechtsanwalt', 'vertretung'],
            'signal_words': ['leistungsort', 'schuldet', 'erstattung', 'b2b'],
            'require_context': False,
        },
        '3a.9': {
            'keywords': ['grundstück', 'gebäude', 'fassade', 'klimaanlage', 'wartung',
                        'montage', 'serverstellplatz', 'liegenschaft', 'bauwerk',
                        'demontage', 'sanierung', 'planung', 'planungsleistung'],
            'signal_words': ['belegenheitsort', 'gelegen'],
            'require_context': False,
        },
        '3a.11': {
            'keywords': ['seminar', 'schulung', 'fortbildung', 'unterricht', 'wissenschaftlich',
                        'fachseminar', 'lehrgang'],
            'signal_words': ['tätigkeitsort'],
            'require_context': False,
        },
        '3a.11a': {
            'keywords': ['konzert', 'veranstaltung', 'kulturell', 'eintrittsberechtigung',
                        'eintritt', 'festival', 'messe'],
            'signal_words': ['veranstaltungsort'],
            'require_context': False,
        },
        '3a.12': {
            'keywords': ['dienstwagen', 'firmenwagen', 'beförderungsmittel', 'fahrzeug',
                        'vermietung', 'überlassung', 'kfz', 'bus', 'tonstudio'],
            'signal_words': ['kurzfristig', 'nicht kurzfristig', 'übergabeort'],
            'require_context': False,
        },
        '3a.13': {
            'keywords': ['telekommunikation', 'mobilfunk', 'elektronisch', 'digital',
                        'download', 'streaming'],
            'signal_words': ['nichtunternehmer', 'privatperson', 'wohnsitz'],
            'require_context': True,
        },
        '3a.1a': {
            'keywords': ['privathaushalt', 'dienstnehmer', 'unentgeltlich', 'eigenverbrauch',
                        'personal', 'privatnutzung', 'aufmerksamkeit'],
            'signal_words': ['gleichgestellt'],
            'require_context': False,
        },
        '3a.4': {
            'keywords': ['besorgung', 'eigenen namen', 'reiseleistung', 'weiterverrechnung'],
            'signal_words': ['besorgungsleistung'],
            'require_context': False,
        },
        '3a.2': {
            'keywords': ['tausch', 'gegenleistung', 'tauschähnlich'],
            'signal_words': ['tauschähnlicher umsatz'],
            'require_context': False,
        },
    }

    def _apply_keyword_boost_3a(
        self, query: str, candidates: List[RetrievalResult]
    ) -> List[RetrievalResult]:
        """Boost § 3a chunks based on keyword matching."""
        query_lower = query.lower()

        matched_abs = []
        for abs_key, config in self.KEYWORD_3A_MAP.items():
            kw_hits = sum(1 for kw in config['keywords'] if kw in query_lower)
            sig_hits = sum(1 for sw in config['signal_words'] if sw in query_lower)
            if kw_hits >= 1 or sig_hits >= 1:
                if config.get('require_context'):
                    context_words = ['privatperson', 'nichtunternehmer', 'privat',
                                    'wohnsitz', 'konsument', 'endverbraucher']
                    if not any(cw in query_lower for cw in context_words):
                        continue
                matched_abs.append(abs_key)

        if not matched_abs:
            return candidates

        print(f"  § 3a keyword-Boost: {matched_abs}")

        # Boost also existing § 3a chunks
        candidate_ids = {r.chunk_id for r in candidates}
        for r in candidates:
            if r.chunk.ref.paragraph != '3a':
                continue
            chunk_abs = r.chunk.ref.absatz
            if not chunk_abs:
                continue
            for abs_key in matched_abs:
                target_abs = abs_key.split('.')[1]
                if chunk_abs == target_abs:
                    r.combined_score *= self.KEYWORD_3A_BOOST
                    r.boost_reason = (r.boost_reason + ", " if r.boost_reason else "") + f"kw_3a_{abs_key}"

        # Inject missing § 3a chunks
        injected = 0
        for abs_key in matched_abs:
            target_abs = abs_key.split('.')[1]
            has_match = any(
                r.chunk.ref.paragraph == '3a' and r.chunk.ref.absatz == target_abs
                for r in candidates
            )
            if has_match:
                continue
            for chunk in self.store.get_by_paragraph('3a'):
                if chunk.ref.level == ChunkLevel.PARAGRAPH:
                    continue
                if chunk.ref.absatz == target_abs and chunk.chunk_id not in candidate_ids:
                    if chunk.source_type in (SourceType.USTG, SourceType.ANHANG):
                        candidates.append(RetrievalResult(
                            chunk=chunk,
                            dense_score=0.0, sparse_score=0.0,
                            combined_score=0.75,
                            boost_reason=f"kw_inject_3a_{abs_key}",
                        ))
                        candidate_ids.add(chunk.chunk_id)
                        injected += 1

        if injected:
            print(f"   Injected {injected} § 3a chunks via keyword boost")

        return candidates

    def _rerank(
        self,
        query: str,
        candidates: List[RetrievalResult]
    ) -> List[RetrievalResult]:
        """Rerank with cross-encoder."""
        import math
        boosted = [r for r in candidates if r.boost_reason]
        unboosted = [r for r in candidates if not r.boost_reason]
        unboosted.sort(key=lambda r: r.combined_score, reverse=True)
        to_rerank_set = set()
        to_rerank = []
        for r in boosted:
            to_rerank.append(r)
            to_rerank_set.add(r.chunk_id)
        for r in unboosted:
            if len(to_rerank) >= self.RERANK_TOP_K:
                break
            if r.chunk_id not in to_rerank_set:
                to_rerank.append(r)
                to_rerank_set.add(r.chunk_id)

        rest = [r for r in candidates if r.chunk_id not in to_rerank_set]

        self._load_reranker()

        if self.reranker and to_rerank:
            pairs = [
                (query, (r.chunk.text_with_context or r.chunk.search_text)[:self.RERANK_CONTEXT_CHARS])
                for r in to_rerank
            ]

            try:
                raw_scores = self.reranker.predict(pairs)

                for i, result in enumerate(to_rerank):
                    norm_score = 1.0 / (1.0 + math.exp(-float(raw_scores[i])))
                    result.rerank_score = norm_score

                    base_score = (
                        self.DENSE_WEIGHT_RERANKED * result.dense_score +
                        self.SPARSE_WEIGHT_RERANKED * result.sparse_score +
                        self.RERANK_WEIGHT * result.rerank_score
                    )

                    br = result.boost_reason or ""
                    if "exact_ref" in br:
                        base_score *= self.EXACT_REF_BOOST
                    elif "explicit" in br or "partial_ref" in br:
                        base_score *= self.EXPLICIT_REF_BOOST
                    if "ustr_linked" in br:
                        base_score *= self.USTR_LINKED_BOOST
                    if "rz_match" in br:
                        base_score *= self.EXACT_REF_BOOST

                    result.combined_score = base_score

                print(f"   Reranked {len(to_rerank)} candidates (Sigmoid-Norm)")

            except Exception as e:
                print(f"   Reranking failed: {e}")

        for r in to_rerank:
            if r.rerank_score is not None and r.rerank_score < self.MIN_RERANK_SCORE:
                if not r.boost_reason:
                    r.combined_score *= 0.1

        all_candidates = to_rerank + rest
        all_candidates.sort(key=lambda r: r.combined_score, reverse=True)

        return all_candidates

    # Reduced UStR guaranteed slots from 4 to 2.
    MIN_USTR_LINKED = 2
    MIN_USTG_EXPLICIT = 8

    def _balance_sources(
        self,
        candidates: List[RetrievalResult],
        eu_relevant: bool,
        top_k: int,
        has_explicit_arts: bool = False,
    ) -> List[RetrievalResult]:
        """Balance sources in final output."""

        ustg_explicit = []
        ustg_other = []
        ustr_linked = []
        ustr_other = []
        anhang_all = []

        for r in candidates:
            boost = r.boost_reason or ""
            if r.chunk.source_type == SourceType.USTG:
                is_injected = (
                    r.combined_score <= 0.11 and
                    r.rerank_score is None and
                    not boost
                )
                if not is_injected and ("explicit" in boost or "exact_ref" in boost or "partial_ref" in boost):
                    ustg_explicit.append(r)
                else:
                    ustg_other.append(r)
            elif r.chunk.source_type == SourceType.USTR:
                if "ustr_linked" in boost:
                    ustr_linked.append(r)
                else:
                    ustr_other.append(r)
            elif r.chunk.source_type == SourceType.ANHANG:
                anhang_all.append(r)

        final = []
        seen = set()

        def _add(results: List[RetrievalResult], max_n: int):
            added = 0
            for r in results:
                if added >= max_n or len(final) >= top_k:
                    break
                if r.chunk_id not in seen:
                    final.append(r)
                    seen.add(r.chunk_id)
                    added += 1
            return added

        n_ustg_explicit = _add(ustg_explicit, self.MIN_USTG_EXPLICIT)

        ustr_min = min(self.MIN_USTR_LINKED, len(ustr_linked))
        n_ustr_linked = _add(ustr_linked, ustr_min)

        ustg_remaining = self.MAX_USTG - n_ustg_explicit
        _add(ustg_explicit, ustg_remaining)
        _add(ustg_other, ustg_remaining)

        n_ustg_total = sum(1 for r in final if r.chunk.source_type == SourceType.USTG)

        ustr_remaining = self.MAX_USTR - n_ustr_linked
        _add(ustr_linked, ustr_remaining)
        _add(ustr_other, ustr_remaining)

        n_ustr_total = sum(1 for r in final if r.chunk.source_type == SourceType.USTR)

        # Also include Anhang chunks when explicit Art. refs are present, not just when eu_relevant keywords.
        n_anhang = 0
        if eu_relevant or has_explicit_arts:
            n_anhang = _add(anhang_all, self.MAX_ANHANG)

        remaining_budget = top_k - len(final)
        if remaining_budget > 0:
            all_remaining = ustg_explicit + ustg_other + ustr_linked + ustr_other + anhang_all
            all_remaining.sort(key=lambda r: r.combined_score, reverse=True)
            _add(all_remaining, remaining_budget)

        print(f"   Balance: UStG={n_ustg_total} (explicit={n_ustg_explicit}), "
              f"UStR={n_ustr_total} (linked={n_ustr_linked}), "
              f"Anhang={n_anhang}")

        if len(final) < top_k:
            print(f"   Only {len(final)}/{top_k} results available, not enough candidates")

        return final[:top_k]

    def _expand_paragraph_chunks(
        self,
        final: List[RetrievalResult],
        all_candidates: List[RetrievalResult],
        top_k: int,
    ) -> List[RetrievalResult]:
        """Replace §-level UStG/Anhang chunks with their Abs-level children.
        §-level chunks ("§ 3a", "§ 12") appear in top-K but have no Absatz, so depth=1, never match expected refs with Abs (depth >= 2). 62% of top-20 slots were wasted on these unmatchable refs.
        So for each §-level chunk in the list, find its best Abs-level children from the candidate pool and substitute them in."""
        expanded = []
        seen_ids = set()
        candidate_map = {r.chunk_id: r for r in all_candidates}
        n_expanded = 0
        
        for r in final:
            chunk = r.chunk
            
            # Skip UStR §-level chunks entirely bc no probabilty of matching
            if (chunk.source_type == SourceType.USTR and 
                chunk.ref.level == ChunkLevel.PARAGRAPH):
                continue
            # Expand UStG/Anhang §-level chunks to Abs children
            if (chunk.ref.level == ChunkLevel.PARAGRAPH and 
                chunk.source_type in (SourceType.USTG, SourceType.ANHANG)): 
            # Find Abs-level children for this paragraph
                children = self.store.get_children(chunk.chunk_id)
                abs_children = [c for c in children if c.ref.level == ChunkLevel.ABSATZ]
                if abs_children:
            # Score children: use candidate scores if available, else use parent score
                    scored_children = []
                    for child in abs_children:
                        if child.chunk_id in candidate_map:
                            scored_children.append((child, candidate_map[child.chunk_id].combined_score))
                        else:
            # Give it parent's score slightly reduced
                            scored_children.append((child, r.combined_score * 0.8))
            # Sort by score, take top 3
                    scored_children.sort(key=lambda x: x[1], reverse=True)
                    
                    added = 0
                    for child, score in scored_children[:3]:
                        if child.chunk_id not in seen_ids:
                            child_result = candidate_map.get(child.chunk_id)
                            if child_result is None:
                                child_result = RetrievalResult(
                                    chunk=child,
                                    dense_score=r.dense_score * 0.8,
                                    sparse_score=r.sparse_score * 0.8,
                                    combined_score=score,
                                    boost_reason="expanded_from_para",
                                )
                            expanded.append(child_result)
                            seen_ids.add(child.chunk_id)
                            added += 1
                    
                    if added > 0:
                        n_expanded += 1
                        continue 
                
            # If no children found, keep the §-level chunk
                if chunk.chunk_id not in seen_ids:
                    expanded.append(r)
                    seen_ids.add(chunk.chunk_id)
            else:
            # Non-§-level chunk: keep as-is
                if chunk.chunk_id not in seen_ids:
                    expanded.append(r)
                    seen_ids.add(chunk.chunk_id)
        
        if n_expanded:
            print(f"   Expanded {n_expanded} §-level chunks to Abs-level children")
        
        return expanded[:top_k]

    def _deduplicate_by_paragraph(
        self,
        results: List[RetrievalResult],
        top_k: int,
        max_per_para: int = 4,
    ) -> List[RetrievalResult]:
        """Change paragraph diversity to improve Recall@K. Reason for this: without this, § 12 can fill 15+ of 20 slots with different Abs/Z/lit chunks and leaving no room for § 3a, § 19 etc."""
        primary = []
        tail = []
        # Separate counters per source type
        ustg_para_count: dict = {}
        ustr_para_count: dict = {}

        for r in results:
            key = r.chunk.ref.paragraph
            if r.chunk.source_type == SourceType.USTR:
                count = ustr_para_count.get(key, 0)
                if count < 2:  # Max 2 UStR per paragraph
                    primary.append(r)
                    ustr_para_count[key] = count + 1
                else:
                    tail.append(r)
            else:
        # UStG and Anhang
                count = ustg_para_count.get(key, 0)
                if count < max_per_para:
                    primary.append(r)
                    ustg_para_count[key] = count + 1
                else:
                    tail.append(r)

        return (primary + tail)[:top_k]

    # Paragraphs that are "general background" and almost never the main norm, LLM sees them in context and cites them even when they not the answer, so only include if explicitly in query.
    NOISE_PARAGRAPHS = {'1', '23', '27', '28'}

    def get_context_for_llm(self, results: List[RetrievalResult], explicit_paras: set = None) -> tuple:
        """Format retrieval results as structured context for the LLM. Filter out paragraphs like §1, §23, §27, §28 unless they
        were explicitly referenced in the query."""
        if explicit_paras is None:
            explicit_paras = set()
        
        # Remove noise paragraphs from UStG results
        noise_to_remove = self.NOISE_PARAGRAPHS - explicit_paras
        
        ustg_results = [r for r in results 
                       if r.chunk.source_type == SourceType.USTG
                       and r.chunk.ref.paragraph not in noise_to_remove]
        anhang_results = [r for r in results if r.chunk.source_type == SourceType.ANHANG]
        ustr_results = [r for r in results if r.chunk.source_type == SourceType.USTR]

        parts = []
        source_num = 1
        source_map = {}

        if ustg_results:
            parts.append("LEGAL BASIS (UStG 1994)🏛️")
            for r in ustg_results:
                parts.append(f"[{source_num}] {r.chunk.citation}")
                if r.chunk.title:
                    parts.append(f"    ({r.chunk.title})")
                parts.append(f"    {r.chunk.text_with_context or r.chunk.text}")
                if r.boost_reason:
                    parts.append(f"    [Match: {r.boost_reason}]")
                parts.append("")
                source_map[source_num] = r.chunk.citation
                source_num += 1

        if anhang_results:
            parts.append("EU-BINNENMARKTREGELUNG (Anhang UStG)🏛️")
            for r in anhang_results:
                parts.append(f"[{source_num}] {r.chunk.citation}")
                parts.append(f"    {r.chunk.text_with_context or r.chunk.text}")
                parts.append("")
                source_map[source_num] = r.chunk.citation
                source_num += 1

        if ustr_results:
            parts.append("RICHTLINIEN & JUDIKATUR (UStR 2000)🏛️")
            for r in ustr_results:
                parts.append(f"[{source_num}] {r.chunk.citation}")
                if r.chunk.judikatur:
                    parts.append(f"    Judikatur: {', '.join(r.chunk.judikatur[:3])}")
                if r.chunk.linked_ustg_refs:
                    parts.append(f"    Bezug: §§ {', '.join(r.chunk.linked_ustg_refs)}")
                parts.append(f"    {r.chunk.text_with_context or r.chunk.text}")
                parts.append("")
                source_map[source_num] = r.chunk.citation
                source_num += 1

        return "\n".join(parts), source_map
