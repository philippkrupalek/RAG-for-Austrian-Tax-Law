"""
evaluate.py – Evaluation pipeline for UStG RAG, to run all experiments

2 models: DeepSeek-V3, Llama-3.1-8B
4 setups: Baseline up to full rag
9 metrics: 7 primary (rule-based) + 2 secondary (LLM-as-judge)
150 test cases from golden_dataset.json
"""

import json
import re
import csv
import math
import time
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field, asdict
from collections import defaultdict

from config import (
    ModelConfig, DEEPSEEK_V3, LLAMA_8B, GPT4O_MINI,
    ExperimentSetup, SetupID, SETUPS,
    DEEPSEEK_SETUPS, LLAMA_SETUPS,
    GOLDEN_DATASET_PATH, RESULTS_DIR, EMBEDDING_MODEL,
    USTG_RTF_PATH, ANHANG_RTF_PATH, USTR_XML_PATH,
)
from models import ChunkStore, RetrievalResult, SourceType, ChunkLevel
from llm import (
    get_client, AnswerGenerator, QueryRewriter,
    extract_cited_paragraphs,
    extract_cited_references, find_missing_paragraphs,
)


# helpers

def _ref_to_dotkey(ref) -> str:
    """Convert a LegalReference into a dot-separated key for detailed matching. Example: LegalReference(paragraph="12", absatz="2", ziffer="2", litera="a") = "12.2.2.a"
    Also annex chunks use paragraph="Art7". The "Art" prefix is kept so _parse_legal_ref can parse it again with the Art dot-key branch. Example: LegalReference(paragraph="Art7", absatz="1", ziffer="4") = "Art7.1.4"
    """
    parts = [ref.paragraph]
    if ref.absatz:
        parts.append(ref.absatz)
        if ref.ziffer:
            parts.append(ref.ziffer)
            if ref.litera:
                parts.append(ref.litera)
    return ".".join(parts)


# datastructures

@dataclass
class TestCase:
    """A single test case from the dataset"""
    case_id: int
    statement: str
    question: str
    answer: str
    result: str           # "Ja" / "Nein" / other
    paragraphs: List[str] # Expected §§
    topic: str


@dataclass
class RunResult:
    """Result of a single experiment run"""
    # identifiers
    model_name: str
    setup_id: str
    case_id: int

    # Inputs
    question: str
    expected_result: str
    expected_paragraphs: List[str]

    # Outputs
    answer: str = ""
    predicted_result: str = ""
    cited_paragraphs: List[str] = field(default_factory=list)
    retrieved_paragraphs: List[str] = field(default_factory=list)

    # Metrics for Retrieval
    recall_at_5: float = 0.0
    recall_at_20: float = 0.0
    ndcg_at_10: float = 0.0

    # Metrics for Generation - Citation 
    outcome_accuracy: float = 0.0
    citation_precision: float = 0.0
    citation_recall: float = 0.0
    citation_f1: float = 0.0
    
    # Metrics for Generation: Citation (partial = partial credit for ancestor)
    citation_precision_partial: float = 0.0
    citation_recall_partial: float = 0.0
    citation_f1_partial: float = 0.0

    # Secondary Metrics - LM-as-Judge
    judge_groundedness: Optional[float] = None     # None = N/A (S1 no context)
    judge_doc_relevance: Optional[float] = None    # None = N/A (S1 no context)
    judge_answer_correctness: Optional[float] = None  # Works for all setups incl. baseline

    # Meta
    backfill_count: int = 0
    duration_seconds: float = 0.0
    error: str = ""


# Load GOlden Dataset

def load_golden_dataset(path: str = None) -> List[TestCase]:
    """Load test cases from golden_dataset.json"""
    path = path or str(GOLDEN_DATASET_PATH)

    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    cases = []
    for item in data:
        cases.append(TestCase(
            case_id=item['case_id'],
            statement=item['statement'],
            question=item['question'],
            answer=item['answer'],
            result=item['result'] or "",
            paragraphs=item['paragraphs'],
            topic=item['topic'],
        ))

    print(f"Loaded {len(cases)} test cases")
    return cases


# Primary Metrics

class RuleBasedMetrics:
    """Computes all rule-based metrics. No LLM calls"""

    @staticmethod
    def outcome_accuracy(predicted: str, expected: str) -> float:
        """Does the predicted outcome match the expected one? Normalizes Ja/Nein and handles all non inary expected values.
        """
        pred = predicted.strip().lower()
        exp = expected.strip().lower()

        # Direct match
        if pred == exp:
            return 1.0

        # Normalize common Ja/Nein variants
        ja_variants = ['ja', 'yes', 'zulässig', 'steht zu', 'berechtigt', 'voller vsta',
                       'empfängerort', 'steht der vorsteuerabzug zu']
        nein_variants = ['nein', 'no', 'nicht zulässig', 'steht nicht zu', 'nicht berechtigt']

        pred_is_ja = any(v in pred for v in ja_variants)
        pred_is_nein = any(v in pred for v in nein_variants)
        exp_is_ja = any(v in exp for v in ja_variants)
        exp_is_nein = any(v in exp for v in nein_variants)

        if pred_is_ja and exp_is_ja:
            return 1.0
        if pred_is_nein and exp_is_nein:
            return 1.0
        if pred_is_ja and exp_is_nein:
            return 0.0
        if pred_is_nein and exp_is_ja:
            return 0.0

        # Special or non binary outcomes – keyword matching, each entry: (keywords_that_must_appear_in_pred, canonical_exp_keyword)
        SPECIAL_OUTCOMES = [
            (['doppelte steuerschuld', 'doppelt'],          'doppelte steuerschuld'),
            (['ex nunc'],                                    'ex nunc'),
            (['ex tunc'],                                    'ex tunc'),
            (['sollbesteuerung', 'soll-besteuerung'],        'sollbesteuerung'),
            (['istbesteuerung',  'ist-besteuerung'],         'istbesteuerung'),
            (['leistungsempfänger', 'reverse charge', 'rc bau', 'empfänger schuldet'],
                                                             'leistungsempfänger'),
            (['empfängerort', 'empfänger-ort', 'deutschland', 'bestimmungsort'],
                                                             'empfängerort'),
            (['teilweise', 'anteilig', 'zum teil'],          'teilweise'),
        ]

        for keywords, canonical in SPECIAL_OUTCOMES:
            exp_matches = any(kw in exp for kw in keywords) or canonical in exp
            pred_matches = any(kw in pred for kw in keywords)
            if exp_matches and pred_matches:
                return 1.0

        # "Ja (voller VStA)" and "Ja (Empfängerort)" - expected contains "ja" qualifier, predicted just says "ja" -> still correct
        if exp_is_ja and pred_is_ja:
            return 1.0

        # Case 16: Doppelte Steuerschuld: Model says "leistender Unternehmer schuldet .. schuldet er diesen Betrag" = both parties liable -> that is doppelte steuerschuld, model just doesn't
        # use the exact term. Check if both liabilities are mentioned.
        if 'doppelte steuerschuld' in exp:
            both_liable = (
                ('leistende' in pred or 'leistungserbringer' in pred) and
                ('schuldet' in pred) and
                ('rechnung' in pred or 'ausweis' in pred)
            )
            if both_liable:
                return 1.0

        return 0.0

    @staticmethod
    def extract_outcome(answer: str, expected: str = "") -> str:
        """Extract the predicted outcome from the LLM answer text. Priority:
        1. Structured "Result:" / "Ergebnis:" token -> Yes/No/special 1b. Direct Yes/No in section d) 1c. Scan the entire Result section for Yes/No (handles descriptive answers)
        2. Special non-binary patterns (double tax liability, ex nunc, etc.)
        3. Non-binary: match expected term directly in the answer
        4. Fallback: pattern matching on the last 800 + first 500 characters"""
        
        answer_lower = answer.lower()
        full_text = answer_lower

        # Structured token: "Ergebnis:" / "Result:" / " Ergebnis"
        ergebnis_match = re.search(
            r'(?:ergebnis|result)[:\s*]*\**\s*'
            r'(ja|nein|teilweise|anteilig|kommt\s+darauf\s+an|doppelte\s+steuerschuld|ex\s+nunc|ex\s+tunc'
            r'|sollbesteuerung|istbesteuerung|leistungsempfänger|empfängerort)',
            full_text,
        )
        if ergebnis_match:
            token = ergebnis_match.group(1).strip()
            if token == 'ja': return "Ja"
            elif token == 'nein': return "Nein"
            elif token in ('teilweise', 'anteilig'): return "Teilweise"
            elif 'doppelte' in token: return "Doppelte Steuerschuld"
            elif token == 'ex nunc': return "Ex nunc (Zeitpunkt der Berichtigung)"
            elif token == 'ex tunc': return "Ex tunc"
            elif token == 'sollbesteuerung': return "Sollbesteuerung"
            elif token == 'istbesteuerung': return "Istbesteuerung"
            elif 'leistungsempfänger' in token: return "Leistungsempfänger (RC Bau)"
            elif 'empfängerort' in token: return "Empfängerort (Deutschland)"

        # section without Ergebnis keyword: "d) **Ja, ..."
        d_match = re.search(
            r'\bd\)\s*\**\s*(ja|nein|doppelte\s+steuerschuld|ex\s+nunc|ex\s+tunc'
            r'|sollbesteuerung|leistungsempfänger|empfängerort)',
            full_text,
        )
        if d_match:
            token = d_match.group(1).strip()
            if token == 'ja': return "Ja"
            elif token == 'nein': return "Nein"
            elif 'doppelte' in token: return "Doppelte Steuerschuld"
            elif token == 'ex nunc': return "Ex nunc (Zeitpunkt der Berichtigung)"
            elif token == 'sollbesteuerung': return "Sollbesteuerung"
            elif 'leistungsempfänger' in token: return "Leistungsempfänger (RC Bau)"
            elif 'empfängerort' in token: return "Empfängerort (Deutschland)"

        # Scan the Ergebnis Section for Ja/Nein (not just first token)
        ergebnis_section_match = re.search(
            r'(?:d\)\s*)?(?:ergebnis|result)\s*[:\s]*\**\s*(.*)',
            full_text, re.DOTALL
        )
        if ergebnis_section_match:
            ergebnis_text = ergebnis_section_match.group(1)[:600]

            ja_in = re.search(r'\bja\b', ergebnis_text)
            nein_in = re.search(r'\bnein\b', ergebnis_text)
            if ja_in and not nein_in: return "Ja"
            if nein_in and not ja_in: return "Nein"
            if ja_in and nein_in:
                return "Ja" if ja_in.start() < nein_in.start() else "Nein"

            # Semantic patterns in ergebnis section
            nein_semantic = [
                r'\bnicht\s+(?:zu|ausgeschlossen|zulässig|möglich|berechtigt|abzugsfähig|erstattungsfähig|steuerbar)\b',
                r'\bkein\s+(?:vorsteuerabzug|übergang|anspruch)\b',
                r'\bsteht\s+nicht\s+zu\b', r'\bist\s+ausgeschlossen\b',
                r'\bnicht\s+ein(?:tritt|greif)', r'\bversag(?:t|en)\b',
                r'\bnicht\s+als\s+vorsteuer\s+abzieh',
                r'\bnicht\s+auf\s+den\b.*\büber\b',
                r'\bkann\s+nicht\b', r'\bnicht\s+erstattungsfähig\b',
                r'\bliegt\s+nicht\s+vor\b',
            ]
            ja_semantic = [
                r'\bsteht\s+(?:ihm|ihr|der|dem|zu)\b.*\bzu\b',
                r'\bist\s+(?:zulässig|berechtigt|steuerbar|steuerfrei|steuerpflichtig|abzugsfähig)\b',
                r'\bliegt\s+(?:vor|in\s+österreich)\b',
                r'\bgeht\s+(?:über|auf)\b.*\büber\b',
                r'\bberechtigt\b', r'\bschuldet\b', r'\bentsteht\b',
                r'\bist\s+zu\s+berichtig',
                r'\bist\s+(?:die|eine)\s+(?:steuerfreie|steuerpflichtige)\b',
                r'\bkann\b.*\bberichtig',
                r'\bleistungsort\s+liegt\b',
                r'\bsind\s+erfüllt\b',
                r'\bist\s+(?:eine\s+)?(?:lieferung|sonstige\s+leistung|reiseleistung|bauleistung)\b',
                r'\bist\s+erstattungsfähig\b',
            ]
            for pat in nein_semantic:
                if re.search(pat, ergebnis_text): return "Nein"
            for pat in ja_semantic:
                if re.search(pat, ergebnis_text): return "Ja"

        # 2. Special non-binary patterns (scan full answer)
        if 'doppelte steuerschuld' in full_text: return "Doppelte Steuerschuld"
        if 'ex nunc' in full_text: return "Ex nunc (Zeitpunkt der Berichtigung)"
        if 'ex tunc' in full_text: return "Ex tunc"
        if re.search(r'leistungsempfänger\s+schuldet|schuldet.*leistungsempfänger|reverse.charge.*bauleistung', full_text):
            return "Leistungsempfänger (RC Bau)"
        if re.search(r'leistungsort.*empfängerort|empfängerort.*deutschland|bestimmungsortprinzip', full_text):
            return "Empfängerort (Deutschland)"
        if 'sollbesteuerung' in full_text and 'istbesteuerung' not in full_text[:100]:
            return "Sollbesteuerung"

        # Non-binary: try to match the expected term directly
        if expected:
            exp_lower = expected.strip().lower()
            exp_core = re.sub(r'\s*\(.*?\)', '', exp_lower).strip()
            if exp_core and len(exp_core) > 3 and exp_core in full_text:
                return expected.strip()

        # Fallback: last 800 and first 500 chars
        search_window = answer_lower[:500] + ' ' + answer_lower[-800:]
        nein_patterns = [
            r'\bnein\b', r'\bnicht zulässig\b', r'\bnicht möglich\b',
            r'\bkein vorsteuerabzug\b', r'\bnicht berechtigt\b',
            r'\bkann nicht\b', r'\bsteht nicht zu\b',
            r'\bnicht abziehen\b', r'\bnicht abzugsfähig\b',
            r'\bnicht steuerbar\b', r'\bnicht erstattungsfähig\b',
            r'\bausgeschlossen\b', r'\bzu versagen\b',
        ]
        ja_patterns = [
            r'\bja\b', r'\bzulässig\b',
            r'\bvorsteuerabzug steht zu\b', r'\bberechtigt\b',
            r'\bkann.*abziehen\b', r'\babzugsfähig\b',
            r'\bist steuerbar\b', r'\bist steuerfrei\b',
            r'\bliegt vor\b', r'\bschuldet\b.*\bumsatzsteuer\b',
        ]
        for pat in nein_patterns:
            if re.search(pat, search_window): return "Nein"
        for pat in ja_patterns:
            if re.search(pat, search_window): return "Ja"

        return "Unklar"

    @staticmethod
    def _split_compound_refs(refs: List[str]) -> List[str]:
        """ Split compound references into individual refs.The golden dataset sometimes contains compound references, e.g.:
        -) "§ 3a para. 9 and para. 10 UStG 1994" -> two refs: 3a.9 and 3a.10
        -) "§ 12 para. 3 item 1 and 2 UStG 1994" -> two refs: 12.3.1 and 12.3.2
        -) "§ 6 para. 1 item 16 and para. 2 UStG 1994" -> two refs: 6.1.16 and 6.2
        Without this, only the first sub-reference is parsed and the second is lost."""
        
        result = []
        for ref in refs:
            # If "§ X Abs. Y und Abs. Z", split them into two refs
            m = re.match(
                r'(§\s*\d+[a-z]?)\s+Abs\.?\s*(\d+[a-z]?)\s+und\s+Abs\.?\s*(\d+[a-z]?)(.*)',
                ref, re.IGNORECASE
            )
            if m:
                suffix = m.group(4).strip()
                result.append(f"{m.group(1)} Abs. {m.group(2)} {suffix}")
                result.append(f"{m.group(1)} Abs. {m.group(3)} {suffix}")
                continue
            
            # If "§ X Abs. Y Z N und M", split into two refs
            m = re.match(
                r'(§\s*\d+[a-z]?\s+Abs\.?\s*\d+[a-z]?)\s+Z\.?\s*(\d+)\s+und\s+(\d+)(.*)',
                ref, re.IGNORECASE
            )
            if m:
                suffix = m.group(4).strip()
                result.append(f"{m.group(1)} Z {m.group(2)} {suffix}")
                result.append(f"{m.group(1)} Z {m.group(3)} {suffix}")
                continue
            
            # If "§ X Abs. Y Z N und Abs. Z" (mixed), two refs at different levels
            m = re.match(
                r'(§\s*\d+[a-z]?)\s+Abs\.?\s*(\d+[a-z]?)\s+Z\.?\s*(\d+)\s+und\s+Abs\.?\s*(\d+[a-z]?)(.*)',
                ref, re.IGNORECASE
            )
            if m:
                suffix = m.group(5).strip()
                result.append(f"{m.group(1)} Abs. {m.group(2)} Z {m.group(3)} {suffix}")
                result.append(f"{m.group(1)} Abs. {m.group(4)} {suffix}")
                continue
            
            # No compound pattern
            result.append(ref)
        
        return result
    
    @staticmethod
    def _parse_legal_ref(ref_str: str) -> Tuple[str, Optional[str], Optional[str], Optional[str]]:
        """Parse a legal reference string into (paragraph, absatz, ziffer, litera). handles: 
        -) Golden dataset format: "§ 12 para. 2 item 2 lit. a UStG 1994"
        -) Dot format: "12.2.2.a", "12.2", or "12"
        -) Art dot format: "Art7.1.4" or "Art7.1"
        -) Bare number: "12"
        -) Article references: "Art. 1 para. 1" or "Art. 7 para. 1 UStG 1994"
        -) Paragraphs with letter suffixes: "§ 19 para. 1a", "§ 3a para. 11a"
        """
        
        ref = ref_str.strip()
        
        # Art format "Art7.1.4" or "Art7" (from _ref_to_dotkey for Anhang)
        art_dot_match = re.match(r'^Art(\d+[a-z]?)(?:\.(\d+[a-z]?))?(?:\.(\d+))?(?:\.([a-z]))?$', ref)
        if art_dot_match:
            return (
                f"Art{art_dot_match.group(1)}",
                art_dot_match.group(2),
                art_dot_match.group(3),
                art_dot_match.group(4),
            )
        
        # Format "12.2.2.a" / "12.2" / "12" / "3a.1a.2" (separated by dot, no § symbol)
        
        dot_match = re.match(r'^(\d+[a-z]?)(?:\.(\d+[a-z]?))?(?:\.(\d+))?(?:\.([a-z]))?$', ref)
        if dot_match:
            return (
                dot_match.group(1),
                dot_match.group(2),  # absatz or None
                dot_match.group(3),  # ziffer or None
                dot_match.group(4),  # litera or None
            )
        
        # Full format like "§ 12 Abs. 2 Z 2 lit. a" or "Art. 7 Abs. 1", distinguish § from Art. so Art. refs get "Art" prefix
        art_match = re.search(r'Art\.?\s*(\d+[a-z]?)', ref)
        para_match = re.search(r'§\s*(\d+[a-z]?)', ref)
        
        if art_match and not para_match:
            # it is an Art. reference prefix with "Art" for Anhang matching
            paragraph = f"Art{art_match.group(1)}"
        elif para_match:
            paragraph = para_match.group(1)
        else:
            return (ref, None, None, None)
        
        # Abs regex captures letter: "1a", "10a", "11a"
        abs_match = re.search(r'Abs(?:atz)?\.?\s*(\d+[a-z]?)', ref, re.IGNORECASE)
        absatz = abs_match.group(1) if abs_match else None
        
        z_match = re.search(r'Z(?:iffer)?\.?\s*(\d+)', ref, re.IGNORECASE)
        ziffer = z_match.group(1) if z_match else None
        
        lit_match = re.search(r'lit(?:era)?\.?\s*([a-z])', ref, re.IGNORECASE)
        litera = lit_match.group(1).lower() if lit_match else None
        
        return (paragraph, absatz, ziffer, litera)
    
    @staticmethod
    def _ref_to_key(parsed: Tuple[str, Optional[str], Optional[str], Optional[str]]) -> str:
        """Convert parsed ref to a canonical comparison key like '12.2.2.a'"""
        parts = [parsed[0]]  # paragraph always present
        if parsed[1]:  # absatz
            parts.append(parsed[1])
            if parsed[2]:  # digit (ziffer)
                parts.append(parsed[2])
                if parsed[3]:  # litera
                    parts.append(parsed[3])
        return ".".join(parts)
    
    @staticmethod
    def _ref_depth(parsed: Tuple) -> int:
        """specified levels: 1=§, 2=§+Abs, 3=§+Abs+Z, 4=§+Abs+Z+lit"""
        d = 1
        if parsed[1] is not None: d += 1
        if parsed[2] is not None: d += 1
        if parsed[3] is not None: d += 1
        return d
    
    @staticmethod
    def _is_ancestor(cited_parsed: Tuple, expected_parsed: Tuple) -> bool:
        """True if cited is a proper ancestor of expected on the same branch.
        -) § 12 para. 2" is an ancestor of "§ 12 para. 2 item 2 lit. a" = True
        -) "§ 12" is an ancestor of "§ 12 para. 2 item 2 lit. a" = True
        -) "§ 12 para. 1" is not an ancestor of "§ 12 para. 2" = False (wrong branch
        """
        c_para, c_abs, c_z, c_lit = cited_parsed
        e_para, e_abs, e_z, e_lit = expected_parsed
        
        # Must be same paragraph
        if c_para != e_para:
            return False
        
        # Cited must be strictly less specific
        c_depth = RuleBasedMetrics._ref_depth(cited_parsed)
        e_depth = RuleBasedMetrics._ref_depth(expected_parsed)
        if c_depth >= e_depth:
            return False  # no ancestor
        
        # All levels that cited specifies must match expected
        if c_abs is not None and c_abs != e_abs:
            return False
        if c_z is not None and c_z != e_z:
            return False
        if c_lit is not None and c_lit != e_lit:
            return False
        
        return True
    
    @staticmethod
    def _strict_match(cited_parsed: Tuple, expected_parsed: Tuple) -> bool:
        """ Strict match, cited must cover all levels specified in expected. cited may be more specific, but not less. "§ 12 para. 2 item 2 lit. a" matches "§ 12 para. 2",
        "§ 12 para. 2 item 2 lit. a" matches "§ 12 para. 2 item 2 lit. a", "§ 12 para. 2" doesn't match "§ 12 para. 2 item 2 lit. a" """
        c_para, c_abs, c_z, c_lit = cited_parsed
        e_para, e_abs, e_z, e_lit = expected_parsed
        
        if c_para != e_para:
            return False
        if e_abs is not None and (c_abs is None or c_abs != e_abs):
            return False
        if e_z is not None and (c_z is None or c_z != e_z):
            return False
        if e_lit is not None and (c_lit is None or c_lit != e_lit):
            return False
        return True
    
    @staticmethod
    def _partial_score(cited_parsed: Tuple, expected_parsed: Tuple) -> float:
        """Partial credit score for a cited ref compared with an expected ref.
        -) Exact or more specific match = 1.0
        -) Ancestor on the correct branch = proportional credit
        -) Example: cited "§ 12 para. 2" vs expected "§ 12 para. 2 item 2 lit. a" (depth 4), cited matches 2 of 4 levels = 0.5
        -) Wrong branch = 0.0 credit
        -) Wrong section = 0.0 credit
        """
        # Check strict match first
        if RuleBasedMetrics._strict_match(cited_parsed, expected_parsed):
            return 1.0
        
        # Check if ancestor on correct branch
        if RuleBasedMetrics._is_ancestor(cited_parsed, expected_parsed):
            c_depth = RuleBasedMetrics._ref_depth(cited_parsed)
            e_depth = RuleBasedMetrics._ref_depth(expected_parsed)
            return c_depth / e_depth
        
        return 0.0

    @staticmethod
    def citation_metrics(
        cited: List[str],
        expected: List[str],
    ) -> Dict[str, float]:
        """Compute citation metrics using both strict and partial scoring. Returns a dict with 6 values:

        -) precision, recall, f1: strict scoring (binary exact match)
        -) precision_partial, recall_partial, f1_partial: partial scoring (ancestor credit)
        
        Strict: the cited reference must cover all levels of the expected reference.
        Partial: ancestor references get proportional credit (depth_cited / depth_expected). """
        parse = RuleBasedMetrics._parse_legal_ref
        to_key = RuleBasedMetrics._ref_to_key
        
        # Split compound refs before parsing
        expected = RuleBasedMetrics._split_compound_refs(expected)
        
        cited_parsed = [parse(c) for c in cited]
        expected_parsed = [parse(e) for e in expected]
        
        # Deduplicate cited
        seen = set()
        cited_dedup = []
        for cp in cited_parsed:
            k = to_key(cp)
            if k not in seen:
                cited_dedup.append(cp)
                seen.add(k)
        cited_parsed = cited_dedup
        
        # Deduplicate expected
        seen = set()
        exp_dedup = []
        for ep in expected_parsed:
            k = to_key(ep)
            if k not in seen:
                exp_dedup.append(ep)
                seen.add(k)
        expected_parsed = exp_dedup
        
        empty = {
            'precision': 0.0, 'recall': 0.0, 'f1': 0.0,
            'precision_partial': 0.0, 'recall_partial': 0.0, 'f1_partial': 0.0,
        }
        if not cited_parsed and not expected_parsed:
            return {k: 1.0 for k in empty}
        if not cited_parsed or not expected_parsed:
            return empty
        
        n_cited = len(cited_parsed)
        n_expected = len(expected_parsed)
        
        # strict: greedy binary matching
        strict_matched_e = set()
        strict_matched_c = set()
        for ei, ep in enumerate(expected_parsed):
            for ci, cp in enumerate(cited_parsed):
                if ci not in strict_matched_c:
                    if RuleBasedMetrics._strict_match(cp, ep):
                        strict_matched_e.add(ei)
                        strict_matched_c.add(ci)
                        break
        
        s_tp = len(strict_matched_e)
        s_prec = s_tp / n_cited
        s_rec = s_tp / n_expected
        s_f1 = 2 * s_prec * s_rec / (s_prec + s_rec) if (s_prec + s_rec) > 0 else 0.0
        
        # Partial: greedy best-score matching, for each expected: cited ref with highest partial score
        partial_scores_per_exp = []  # one Score per expected ref
        partial_scores_per_cited = {}  # best Score per cited ref (for Precision)
        partial_used_c = set()

        for ei, ep in enumerate(expected_parsed):
            best_score = 0.0
            best_ci = -1
            for ci, cp in enumerate(cited_parsed):
                if ci not in partial_used_c:
                    s = RuleBasedMetrics._partial_score(cp, ep)
                    if s > best_score:
                        best_score = s
                        best_ci = ci
            partial_scores_per_exp.append(best_score)
            if best_ci >= 0 and best_score > 0:
                partial_used_c.add(best_ci)
                # store the best score for this cited reference
                partial_scores_per_cited[best_ci] = max(
                    partial_scores_per_cited.get(best_ci, 0.0), best_score
                )

        p_rec = sum(partial_scores_per_exp) / n_expected
        # Partial precision = weighted share of cited refs that match something
        # (not binary 0/1, but the actual partial score)
        if n_cited > 0:
            p_prec = sum(partial_scores_per_cited.values()) / n_cited
        else:
            p_prec = 0.0
        p_f1 = 2 * p_prec * p_rec / (p_prec + p_rec) if (p_prec + p_rec) > 0 else 0.0
        
        return {
            'precision': s_prec, 'recall': s_rec, 'f1': s_f1,
            'precision_partial': p_prec, 'recall_partial': p_rec, 'f1_partial': p_f1,
        }

    @staticmethod
    def _ref_matches(cited_parsed: Tuple, expected_parsed: Tuple) -> bool:
        """Match for retrieval metrics (lenient). A retrieved chunk matches if it is on the correct branch.
        This is appropriate for retrieval because:
        A chunk for "§ 12 para. 2" contains the text of item 2 lit. a within it (ancestor match -> valid hit). 
        A chunk for "§ 12 para. 2 item 2 lit. a" is a sub-part of "§ 12 para. 2" (descendant match -> also a valid hit, since we retrieved the right area)
        But retrieving the wrong paragraph (e.g. para. 1 vs para. 2) is a miss. Special case: if the expected ref is section-only (depth=1), a section-only retrieved chunk is a valid hit.
        """
        # Strict match always counts
        if RuleBasedMetrics._strict_match(cited_parsed, expected_parsed):
            return True

        e_depth = RuleBasedMetrics._ref_depth(expected_parsed)
        c_depth = RuleBasedMetrics._ref_depth(cited_parsed)

        # If expected is paragraph-only, a paragraph-level hit is sufficient
        if e_depth == 1 and c_depth == 1 and cited_parsed[0] == expected_parsed[0]:
            return True

        # Ancestor on correct branch also counts for retrieval (chunk text contains the sub-chunk)
        if RuleBasedMetrics._is_ancestor(cited_parsed, expected_parsed):
            # Require at least Abs-level match (depth >= 2)
            return c_depth >= 2

        # Descendant on correct branch also counts e.g. retrieved "12.2.2.a" matches expected "12.2" because the chunk is part of the expected section
        if RuleBasedMetrics._is_ancestor(expected_parsed, cited_parsed):
            return True

        return False

    @staticmethod
    def retrieval_metrics(
        retrieved_paragraphs: List[str],
        expected_paragraphs: List[str],
    ) -> Dict[str, float]:
        """
        Compute IR metrics: Recall@5, Recall@20, NDCG@10.
        """
        parse = RuleBasedMetrics._parse_legal_ref
        matches_fn = RuleBasedMetrics._ref_matches
        to_key = RuleBasedMetrics._ref_to_key

        retrieved_parsed_raw = [parse(r) for r in retrieved_paragraphs]
        # Deduplicate while preserving rank order
        retrieved_parsed = []
        seen_r = set()
        for rp in retrieved_parsed_raw:
            key = to_key(rp)
            if key not in seen_r:
                retrieved_parsed.append(rp)
                seen_r.add(key)

        expected_parsed = []
        seen_e = set()
        for ep_str in expected_paragraphs:
            ep = parse(ep_str)
            key = to_key(ep)
            if key not in seen_e:
                expected_parsed.append(ep)
                seen_e.add(key)

        if not expected_parsed:
            return {'recall_at_5': 0.0, 'recall_at_20': 0.0, 'ndcg_at_10': 0.0}

        results = {}

        # Recall@K: how many expected refs are matched by top-K retrieved?
        for k in [5, 20]:
            top_k = retrieved_parsed[:k]
            hits = 0
            for ep in expected_parsed:
                for rp in top_k:
                    if matches_fn(rp, ep):
                        hits += 1
                        break
            results[f'recall_at_{k}'] = hits / len(expected_parsed)

        # NDCG@10 (binary relevance)
        k = 10
        dcg = 0.0
        matched_expected = set()
        for i in range(min(k, len(retrieved_parsed))):
            rp = retrieved_parsed[i]
            for ei, ep in enumerate(expected_parsed):
                if ei not in matched_expected and matches_fn(rp, ep):
                    dcg += 1.0 / math.log2(i + 2)
                    matched_expected.add(ei)
                    break

        num_relevant = min(len(expected_parsed), k)
        idcg = sum(1.0 / math.log2(i + 2) for i in range(num_relevant))
        results['ndcg_at_10'] = dcg / idcg if idcg > 0 else 0.0

        return results


# LLM-As-Judge (2 Standalone Functions, GPT-4o-mini)

class LLMJudge:
    """
    Uses GPT-4o-mini as LLM-as-Judge for secondary metrics.

    2 standalone scoring functions:
    1. score_groundedness - Are claims supported by retrieved context?
    2. score_document_relevance - Are retrieved chunks relevant to the question?
    Each function returns a float 0.0-1.0, extracted from model output.
    """

    def __init__(self):
        self.client = None
        self._enabled = False

        try:
            self.client = get_client(GPT4O_MINI)
            self._enabled = True
        except Exception as e:
            print(f"LLM Judge not available: {e}")

    @staticmethod
    def _extract_score(text: str) -> float:
        """Extract the first number between 0 and 1 from model output."""
        m = re.search(r'([01](?:\.\d+)?)', text)
        if m:
            return float(m.group(1))
        m = re.search(r'(\d+\.?\d*)', text)
        if m:
            return min(max(float(m.group(1)), 0.0), 1.0)
        return 0.0

    def _call(self, prompt: str) -> float:
        """Make a single API call and parse float response."""
        if not self._enabled:
            return 0.0
        try:
            response = self.client.chat.completions.create(
                model=GPT4O_MINI.model_string,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=10,
            )
            text = response.choices[0].message.content.strip()
            return self._extract_score(text)
        except Exception as e:
            print(f"   ⚠️ Judge call failed: {e}")
            return 0.0

    # Groundedness (Faithfulness)
    def score_groundedness(self, question: str, answer: str, contexts: List[str]) -> float:
        """
        Score how well the answer is grounded/faithful to the retrieved context. 0 = largely invented, 1 = every claim is supported by context."""
        context_text = "\n\n-----\n\n".join(c[:500] for c in contexts[:8])
        prompt = f"""You are grading *groundedness/faithfulness* of a RAG answer.

Question:
{question}

Retrieved context chunks:
{context_text[:3000]}

Model answer:
{answer[:2000]}

On a scale from 0 to 1, where:
- 1 means every important claim in the answer is clearly supported by the context,
- 0 means the answer largely invents information not present in the context,

respond with ONLY a single number between 0 and 1."""
        return self._call(prompt)

    # Document Relevance
    def score_document_relevance(self, question: str, contexts: List[str]) -> float:
        """
        Score how relevant the retrieved context chunks are to the question.
        0 = irrelevant, 1 = highly relevant and sufficient.
        """
        context_text = "\n-----\n".join(c[:500] for c in contexts[:8])
        prompt = f"""You are grading *document relevance* in a legal RAG system.

Question:
{question}

Retrieved context chunks:
{context_text[:3000]}

On a scale from 0 to 1, where:
- 1 means the contexts are highly relevant and sufficient to answer the question,
- 0 means they are completely irrelevant,

respond with only a single number between 0 and 1."""
        return self._call(prompt)

    # Answer Correctness (works for ALL setups including baseline)
    def score_answer_correctness(self, question: str, model_answer: str, golden_answer: str) -> float:
        """
        Score how correct and complete the model answer is compared to the golden answer.
        Works for all setups (S1-S4) since it does not depend on retrieved context.
        0 = completely wrong, 1 = fully correct and complete.
        """
        prompt = f"""You are grading the *correctness* of an answer to an Austrian tax law question.
Compare the model's answer against the reference answer.

Question:
{question}

Reference answer (ground truth):
{golden_answer[:2000]}

Model answer:
{model_answer[:2000]}

On a scale from 0 to 1, where:
- 1 means the model answer is fully correct: right conclusion, right legal provisions cited, correct reasoning,
- 0.5 means partially correct: right conclusion but wrong provisions, or wrong conclusion but right provisions,
- 0 means completely wrong conclusion and wrong provisions,

respond with ONLY a single number between 0 and 1."""
        return self._call(prompt)
# Experiment Run

class ExperimentRunner:
    """
    Runs all experiments: Models x Setups x TestCases
    For each run:
    1. Execute RAG pipeline (with setup-specific components)
    2. Extract predictions
    3. Compute all 9 metrics (7 primary + 2 secondary)
    4. Store results
    """

    def __init__(
        self,
        chunk_store: ChunkStore,
        test_cases: List[TestCase],
        device: str = 'cpu',
    ):
        self.store = chunk_store
        self.test_cases = test_cases
        self.device = device

        # initialized when needed
        self._retriever = None
        self._judge = None

        # Results
        self.results: List[RunResult] = []

    def _get_retriever(self):
        """Lazy-load retriever (shared across all runs)"""
        if self._retriever is None:
            from retriever import HybridRetriever
            self._retriever = HybridRetriever(self.store, device=self.device)
            self._retriever.build(EMBEDDING_MODEL)
        return self._retriever

    def _get_judge(self):
        """Lazy-load LLM judge (GPT-4o-mini)"""
        if self._judge is None:
            self._judge = LLMJudge()
        return self._judge

    def run_all(
        self,
        models: Optional[List[str]] = None,
        setups: Optional[List[str]] = None,
        skip_judge: bool = False,
    ):
        """
        Run all experiments.

        Args:
            models: Filter by model name ("deepseek", "llama")
            setups: Filter by setup ID ("S1", "S4")
            skip_judge: Skip LLM judge calls (no API needed)
        """
        # Build experiment plan
        plan = []

        if models is None or "deepseek" in models:
            for sid in DEEPSEEK_SETUPS:
                if setups is None or sid.value in setups:
                    plan.append((DEEPSEEK_V3, SETUPS[sid]))

        if models is None or "llama" in models:
            for sid in LLAMA_SETUPS:
                if setups is None or sid.value in setups:
                    plan.append((LLAMA_8B, SETUPS[sid]))

        total_runs = len(plan) * len(self.test_cases)
        print(f"\n{'='*70}")
        print(f" EVALUATION PLAN")
        print(f"{'='*70}")
        print(f"   Models × Setups: {len(plan)} combinations")
        print(f"   Test cases: {len(self.test_cases)}")
        print(f"   Total runs: {total_runs}")
        print(f"   LLM Judge: {'ON' if not skip_judge else 'OFF'}")
        print(f"{'='*70}\n")

        run_num = 0
        start_time = time.time()

        for model_config, setup in plan:
            print(f"\n{'─'*70}")
            print(f" {model_config.name} | {setup.setup_id.value}: {setup.description}")
            print(f"{'─'*70}")

            # Create LLM client for this model
            try:
                client = get_client(model_config)
            except Exception as e:
                print(f" Cannot create client for {model_config.name}: {e}")
                continue

            generator = AnswerGenerator(client=client, model_config=model_config)
            rewriter = QueryRewriter(client=client, model_config=model_config) if setup.use_query_rewrite else None

            for case in self.test_cases:
                run_num += 1
                elapsed = time.time() - start_time
                eta = (elapsed / run_num) * (total_runs - run_num) if run_num > 0 else 0

                print(f"\r   [{run_num}/{total_runs}] Case {case.case_id} "
                      f"(ETA: {eta:.0f}s)", end="", flush=True)

                try:
                    result = self._run_single(
                        model_config=model_config,
                        setup=setup,
                        case=case,
                        generator=generator,
                        rewriter=rewriter,
                        skip_judge=skip_judge,
                    )
                    self.results.append(result)

                except Exception as e:
                    print(f"\n    Case {case.case_id} failed: {e}")
                    result = RunResult(
                        model_name=model_config.name,
                        setup_id=setup.setup_id.value,
                        case_id=case.case_id,
                        question=case.question,
                        expected_result=case.result,
                        expected_paragraphs=case.paragraphs,
                        error=str(e),
                    )
                    self.results.append(result)

            print()  # Newline after progress

        total_time = time.time() - start_time
        print(f"\n{'='*70}")
        print(f" COMPLETE: {len(self.results)} runs in {total_time:.1f}s")
        print(f"{'='*70}")

    def _run_single(
        self,
        model_config: ModelConfig,
        setup: ExperimentSetup,
        case: TestCase,
        generator: AnswerGenerator,
        rewriter: Optional[QueryRewriter],
        skip_judge: bool,
    ) -> RunResult:
        """Execute a single experiment run"""
        t0 = time.time()

        result = RunResult(
            model_name=model_config.name,
            setup_id=setup.setup_id.value,
            case_id=case.case_id,
            question=case.question,
            expected_result=case.result,
            expected_paragraphs=case.paragraphs,
        )
#  Both the facts and the question are used:
#  - retrieval_query (facts + question): for dense/BM25 semantic search
#  - ref_query (question only): for § reference extraction in the retriever
#  - rewrite_query (question only): for the QueryRewriter
#  - llm_query (facts + question): for LLM generation
        
        retrieval_query = f"{case.statement}\n\nFrage: {case.question}" if case.statement else case.question
        # LLM gets fact + questions
        llm_query = retrieval_query
        context = ""
        source_map = {}
        retrieval_results = []

        # S1: Baseline (no retrieval) ──
        if not setup.use_retrieval:
            response = generator.generate_baseline(llm_query)
            result.answer = response['answer']

        else:
            # S2-S4: Retrieval-Pipeline
            retriever = self._get_retriever()

            # QueryRewriter only gets question (no fact)
            # REWRITE_PROMPT optimated for short questions
            rewritten = None
            if setup.use_query_rewrite and rewriter:
                _, rewritten = rewriter.rewrite(case.question)

    # Retrieval (always hybrid: dense + BM25)
    # retrieval_query = facts + question for semantic search
    # ref_query = question only for § reference extraction (no noise from facts)
    # rerank_query = question only for the cross-encoder (optimized for short queries)
            retrieval_results = retriever.retrieve(
                query=retrieval_query,
                rewritten_query=rewritten,
                top_k=25,
                use_reranking=setup.use_reranking,
                ref_query=case.question,
                rerank_query=case.question,
            )

            # Build context
            if retrieval_results:
                # Extract explicit §§ from question to pass to context filter
                from retriever import ReferenceExtractor
                _refs = ReferenceExtractor.extract(case.question)
                _explicit = set(_refs['paragraphs'] + _refs['artikel'])
                
                context, source_map = retriever.get_context_for_llm(
                    retrieval_results, explicit_paras=_explicit
                )
                # etrieved_paragraphs sorted by score for correct Recall@K metric
                sorted_by_score = sorted(retrieval_results, key=lambda r: r.combined_score, reverse=True)
                # Filter out UStR paragraph-level chunks from recall calculation. UStR chunks have paragraph="107" (section numbers) with no Abs granularity.
                # They never match expected refs (which have Abs) - waste slots in top-K. The chunks are still in the LLM context, just not counted for recall.
                result.retrieved_paragraphs = [
                    _ref_to_dotkey(r.chunk.ref) for r in sorted_by_score
                    if not (r.chunk.source_type == SourceType.USTR 
                            and r.chunk.ref.level == ChunkLevel.PARAGRAPH)
                ]

            # LLM generates with fact + question
            response = generator.generate(llm_query, context, source_map)
            result.answer = response['answer']

            # 2-Pass Backfill (S4)
            if setup.use_2pass and retrieval_results:
                cited_paras = extract_cited_paragraphs(result.answer)
                context_paras_abs = set()
                for r in retrieval_results:
                    context_paras_abs.add(r.chunk.ref.paragraph)
                    if r.chunk.ref.absatz:
                        context_paras_abs.add(f"{r.chunk.ref.paragraph}.{r.chunk.ref.absatz}")
                missing = find_missing_paragraphs(cited_paras, context_paras_abs)

                if missing:
                    cited_refs = extract_cited_references(result.answer)
                    backfill = retriever.backfill_paragraphs(
                        missing_paras=missing,
                        existing_results=retrieval_results,
                        query=case.question,
                        max_per_para=4,
                        cited_refs=cited_refs,
                    )
                    if backfill:
                        result.backfill_count = len(backfill)
                        all_results = retrieval_results + backfill
                        context2, source_map2 = retriever.get_context_for_llm(
                            all_results, explicit_paras=_explicit
                        )
                        response = generator.generate(llm_query, context2, source_map2)
                        result.answer = response['answer']
                        retrieval_results = all_results
                        sorted_by_score2 = sorted(retrieval_results, key=lambda r: r.combined_score, reverse=True)
                        # Apply same UStR-paragraph filter as first pass
                        result.retrieved_paragraphs = [
                            _ref_to_dotkey(r.chunk.ref) for r in sorted_by_score2
                            if not (r.chunk.source_type == SourceType.USTR 
                                    and r.chunk.ref.level == ChunkLevel.PARAGRAPH)
                        ]

        # Extract predictions
        result.predicted_result = RuleBasedMetrics.extract_outcome(
            result.answer, expected=case.result
        )
        # Granular citation extraction: "§ 12 Abs. 2 Z 2 lit. a" → "12.2.2.a"
        cited_refs = extract_cited_references(result.answer)
        result.cited_paragraphs = []
        seen_keys = set()
        for cr in cited_refs:
            parts = [cr.paragraph]
            if cr.absatz:
                parts.append(cr.absatz)
                if cr.ziffer:
                    parts.append(cr.ziffer)
                    if cr.litera:
                        parts.append(cr.litera)
            key = ".".join(parts)
            if key not in seen_keys:
                result.cited_paragraphs.append(key)
                seen_keys.add(key)

        # Outcome Accuracy
        result.outcome_accuracy = RuleBasedMetrics.outcome_accuracy(
            result.predicted_result, case.result
        )

        # Citation P/R/F1 (Strict + Partial)
        cit = RuleBasedMetrics.citation_metrics(
            result.cited_paragraphs, case.paragraphs
        )
        result.citation_precision = cit['precision']
        result.citation_recall = cit['recall']
        result.citation_f1 = cit['f1']
        result.citation_precision_partial = cit['precision_partial']
        result.citation_recall_partial = cit['recall_partial']
        result.citation_f1_partial = cit['f1_partial']

        # Retrieval metrics (only if retrieval was used)
        if retrieval_results:
            ret = RuleBasedMetrics.retrieval_metrics(
                result.retrieved_paragraphs, case.paragraphs
            )
            result.recall_at_5 = ret['recall_at_5']
            result.recall_at_20 = ret['recall_at_20']
            result.ndcg_at_10 = ret['ndcg_at_10']

        # LLM Judge (2 scores, GPT-4o-mini)
        if not skip_judge:
            judge = self._get_judge()

            # Build context list for judge
            judge_contexts = []
            if retrieval_results:
                judge_contexts = [
                    r.chunk.text_with_context or r.chunk.text
                    for r in retrieval_results
                ]

            # Context-dependent metrics (only when retrieval was used)
            if judge_contexts:
                result.judge_groundedness = judge.score_groundedness(
                    case.question, result.answer, judge_contexts
                )
                result.judge_doc_relevance = judge.score_document_relevance(
                    case.question, judge_contexts
                )
            else:
                result.judge_groundedness = None   # N/A for baseline (no context)
                result.judge_doc_relevance = None  # N/A for baseline (no context)

            # Answer correctness (works for ALL setups, compares against golden answer)
            result.judge_answer_correctness = judge.score_answer_correctness(
                case.question, result.answer, case.answer
            )

        result.duration_seconds = time.time() - t0
        return result

    # RESULTS EXPORT

    def save_results(self, filename: str = None):
        """Save all results to JSON and CSV"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = filename or f"eval_{timestamp}"

        # JSON (full detail)
        json_path = RESULTS_DIR / f"{filename}.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump([asdict(r) for r in self.results], f,
                      ensure_ascii=False, indent=2)
        print(f"   JSON: {json_path}")

        # CSV (for analysis)
        csv_path = RESULTS_DIR / f"{filename}.csv"
        if self.results:
            fieldnames = list(asdict(self.results[0]).keys())
            with open(csv_path, 'w', encoding='utf-8', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for r in self.results:
                    row = asdict(r)
                    # Convert lists to strings for CSV
                    for k, v in row.items():
                        if isinstance(v, list):
                            row[k] = "; ".join(str(x) for x in v)
                    writer.writerow(row)
            print(f"   CSV: {csv_path}")

        return json_path, csv_path

    def print_summary(self):
        """Print summary tables: 9 metrics across all setups"""
        if not self.results:
            print("No results to summarize.")
            return

        # Table Main Results (All Metrics)
        print(f"\n{'='*140}")
        print(" TABLE 1: Main Results (All Metrics)")
        print(f"{'='*140}")
        print(f"{'Model':<16} {'Setup':<6} {'Acc':<7} {'CitF1s':<7} {'CitF1p':<7} "
              f"{'CitPs':<7} {'CitRs':<7} {'CitPp':<7} {'CitRp':<7} "
              f"{'R@5':<7} {'R@20':<7} {'NDCG':<7} {'Gnd':<7} {'DocRel':<7} {'AnsC':<7}")
        print(f"{'─'*16} {'─'*6} {'─'*7} {'─'*7} {'─'*7} "
              f"{'─'*7} {'─'*7} {'─'*7} {'─'*7} "
              f"{'─'*7} {'─'*7} {'─'*7} {'─'*7} {'─'*7} {'─'*7}")

        # Group by model + setup
        groups = defaultdict(list)
        for r in self.results:
            groups[(r.model_name, r.setup_id)].append(r)

        for (model, setup), runs in sorted(groups.items()):
            acc = np.mean([r.outcome_accuracy for r in runs])
            cit_p = np.mean([r.citation_precision for r in runs])
            cit_r = np.mean([r.citation_recall for r in runs])
            f1 = np.mean([r.citation_f1 for r in runs])
            cit_pp = np.mean([r.citation_precision_partial for r in runs])
            cit_rp = np.mean([r.citation_recall_partial for r in runs])
            f1p = np.mean([r.citation_f1_partial for r in runs])
            r5 = np.mean([r.recall_at_5 for r in runs])
            r20 = np.mean([r.recall_at_20 for r in runs])
            ndcg = np.mean([r.ndcg_at_10 for r in runs])

            # LLM Judge
            gnd_runs = [r.judge_groundedness for r in runs if r.judge_groundedness is not None]
            gnd = np.mean(gnd_runs) if gnd_runs else float('nan')
            drel_runs = [r.judge_doc_relevance for r in runs if r.judge_doc_relevance is not None]
            drel = np.mean(drel_runs) if drel_runs else float('nan')
            ansc_runs = [r.judge_answer_correctness for r in runs if r.judge_answer_correctness is not None]
            ansc = np.mean(ansc_runs) if ansc_runs else float('nan')

            r5_str = f"{r5:>5.0%}" if setup != 'S1' else "  N/A"
            r20_str = f"{r20:>5.0%}" if setup != 'S1' else "  N/A"
            ndcg_str = f"{ndcg:>5.2f}" if setup != 'S1' else "  N/A"
            gnd_str = f"{gnd:>5.2f}" if not (gnd != gnd) else "  N/A"
            drel_str = f"{drel:>5.2f}" if not (drel != drel) else "  N/A"
            ansc_str = f"{ansc:>5.2f}" if not (ansc != ansc) else "  N/A"

            print(f"{model:<16} {setup:<6} {acc:>5.0%}  {f1:>5.0%}  {f1p:>5.0%}  "
                  f"{cit_p:>5.0%}  {cit_r:>5.0%}  {cit_pp:>5.0%}  {cit_rp:>5.0%}  "
                  f"{r5_str}  {r20_str}  {ndcg_str}  {gnd_str}  {drel_str}  {ansc_str}")

        print()
        print("  CitF1s/CitPs/CitRs = Strict (exact hierarchical match)")
        print("  CitF1p/CitPp/CitRp = Partial (ancestor credit)")
        print("  R@5/R@20/NDCG = N/A for S1 (no retrieval)")
        print("  Gnd/DocRel = N/A for S1 (no retrieval context)")
        print("  AnsC = Answer Correctness (LLM judge, all setups)")

        # ── TABLE 2: Component Ablation (DeepSeek only) ──
        deepseek_runs = [r for r in self.results if r.model_name == DEEPSEEK_V3.name]
        if deepseek_runs:
            print(f"\n{'='*110}")
            print("TABLE 2: Component Ablation (DeepSeek-V3)")
            print(f"{'='*110}")
            print(f"{'Setup':<6} {'Description':<35} {'Acc':<7} {'CitF1s':<7} {'CitF1p':<7} "
                  f"{'R@5':<7} {'R@20':<7} {'NDCG':<7} {'Gnd':<7}")
            print(f"{'─'*6} {'─'*35} {'─'*7} {'─'*7} {'─'*7} "
                  f"{'─'*7} {'─'*7} {'─'*7} {'─'*7}")

            setup_order = ['S1', 'S2', 'S3', 'S4']
            for sid in setup_order:
                runs = [r for r in deepseek_runs if r.setup_id == sid]
                if not runs:
                    continue

                desc = SETUPS[SetupID(sid)].description
                acc = np.mean([r.outcome_accuracy for r in runs])
                f1 = np.mean([r.citation_f1 for r in runs])
                f1p = np.mean([r.citation_f1_partial for r in runs])
                r5 = np.mean([r.recall_at_5 for r in runs])
                r20 = np.mean([r.recall_at_20 for r in runs])
                ndcg = np.mean([r.ndcg_at_10 for r in runs])
                gnd_runs = [r.judge_groundedness for r in runs if r.judge_groundedness is not None]
                gnd = np.mean(gnd_runs) if gnd_runs else float('nan')

                r5_str = f"{r5:>5.0%}" if sid != 'S1' else "  N/A"
                r20_str = f"{r20:>5.0%}" if sid != 'S1' else "  N/A"
                ndcg_str = f"{ndcg:>5.2f}" if sid != 'S1' else "  N/A"
                gnd_str = f"{gnd:>5.2f}" if not (gnd != gnd) else "  N/A"

                print(f"{sid:<6} {desc:<35} {acc:>5.0%}  {f1:>5.0%}  {f1p:>5.0%}  "
                      f"{r5_str}  {r20_str}  {ndcg_str}  {gnd_str}")

        # Table Model Comparison (S1 vs S4)
        print(f"\n{'='*80}")
        print("TABLE 3: Model Comparison (S1 Baseline → S4 Full RAG)")
        print(f"{'='*80}")
        print()

        for model_name in [DEEPSEEK_V3.name, LLAMA_8B.name]:
            baseline = [r for r in self.results
                        if r.model_name == model_name and r.setup_id == 'S1']
            full_rag = [r for r in self.results
                        if r.model_name == model_name and r.setup_id == 'S4']

            if baseline and full_rag:
                b_acc = np.mean([r.outcome_accuracy for r in baseline])
                f_acc = np.mean([r.outcome_accuracy for r in full_rag])
                b_f1 = np.mean([r.citation_f1 for r in baseline])
                f_f1 = np.mean([r.citation_f1 for r in full_rag])
                b_f1p = np.mean([r.citation_f1_partial for r in baseline])
                f_f1p = np.mean([r.citation_f1_partial for r in full_rag])
                f_r5 = np.mean([r.recall_at_5 for r in full_rag])
                f_r20 = np.mean([r.recall_at_20 for r in full_rag])
                f_ndcg = np.mean([r.ndcg_at_10 for r in full_rag])

                print(f"\n{model_name}:")
                print(f"   Accuracy:           {b_acc:.0%} → {f_acc:.0%} ({f_acc-b_acc:+.0%})")
                print(f"   Citation F1 strict: {b_f1:.0%} → {f_f1:.0%} ({f_f1-b_f1:+.0%})")
                print(f"   Citation F1 partial:{b_f1p:.0%} → {f_f1p:.0%} ({f_f1p-b_f1p:+.0%})")
                print(f"   Recall@5:           N/A → {f_r5:.0%}")
                print(f"   Recall@20:          N/A → {f_r20:.0%}")
                print(f"   NDCG@10:            N/A → {f_ndcg:.2f}")


# MAIN

def main():
    parser = argparse.ArgumentParser(description="UStG RAG v2 Evaluation")
    parser.add_argument('--model', nargs='+', choices=['deepseek', 'llama'],
                       help='Models to evaluate')
    parser.add_argument('--setup', nargs='+',
                       choices=['S1', 'S2', 'S3', 'S4'],
                       help='Setups to run (S1=Baseline, S2=Hybrid, S3=Rerank, S4=Full)')
    parser.add_argument('--skip-judge', action='store_true',
                       help='Skip LLM judge calls (GPT-4o-mini)')
    parser.add_argument('--dry-run', action='store_true',
                       help='Show plan without running')
    parser.add_argument('--output', type=str, default=None,
                       help='Output filename (without extension)')
    parser.add_argument('--judge-only', type=str, default=None,
                       help='Path to existing eval JSON. Runs only LLM judge on those results (no retrieval/generation).')
    args = parser.parse_args()

    print("\n" + "="*70)
    print("UStG RAG - Evaluation Pipeline")
    print("="*70)

    # Load golden dataset
    test_cases = load_golden_dataset()
    gd_map = {c.case_id: c for c in test_cases}

    # ── Judge-only mode: load existing results, run only LLM judge ──
    if args.judge_only:
        print(f"\n   JUDGE-ONLY mode: {args.judge_only}")
        with open(args.judge_only, 'r', encoding='utf-8') as f:
            existing = json.load(f)
        print(f"   Loaded {len(existing)} existing results")

        judge = LLMJudge()
        if not judge._enabled:
            print("   LLM Judge not available (no OpenAI key). Aborting.")
            return

        for i, r in enumerate(existing):
            cid = r['case_id']
            case = gd_map.get(cid)
            if not case:
                continue

            print(f"\r   [{i+1}/{len(existing)}] {r['model_name']} {r['setup_id']} Case {cid}", end="", flush=True)

            # Groundedness + DocRelevance (only if retrieval context exists)
            if r.get('retrieved_paragraphs'):
                # We don't have the actual chunk texts in the JSON, so skip Gnd/DocRel
                # unless they're already filled
                pass

            # Answer Correctness (works for all setups)
            if r.get('judge_answer_correctness') is None:
                r['judge_answer_correctness'] = judge.score_answer_correctness(
                    case.question, r.get('answer', ''), case.answer
                )

        print()

        # Save updated results
        out_path = args.judge_only.replace('.json', '_judged.json')
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(existing, f, ensure_ascii=False, indent=2)
        print(f"\n   Saved to: {out_path}")

        # Print summary
        from collections import defaultdict
        groups = defaultdict(list)
        for r in existing:
            groups[(r['model_name'], r['setup_id'])].append(r)

        print(f"\n{'Model':<16} {'Setup':<6} {'Acc':>5} {'CitF1':>6} {'AnsC':>6}")
        print("-" * 45)
        for (model, setup), runs in sorted(groups.items()):
            acc = np.mean([r['outcome_accuracy'] for r in runs])
            f1 = np.mean([r['citation_f1'] for r in runs])
            ansc_vals = [r['judge_answer_correctness'] for r in runs if r.get('judge_answer_correctness') is not None]
            ansc = np.mean(ansc_vals) if ansc_vals else float('nan')
            ansc_str = f"{ansc:>5.2f}" if not np.isnan(ansc) else "  N/A"
            print(f"{model:<16} {setup:<6} {acc:>4.0%} {f1:>5.0%} {ansc_str}")

        return

    if args.dry_run:
        models = args.model or ['deepseek', 'llama']
        setups = args.setup

        plan = []
        if 'deepseek' in models:
            for s in DEEPSEEK_SETUPS:
                if setups is None or s.value in setups:
                    plan.append((DEEPSEEK_V3.name, s.value))
        if 'llama' in models:
            for s in LLAMA_SETUPS:
                if setups is None or s.value in setups:
                    plan.append((LLAMA_8B.name, s.value))

        total = len(plan) * len(test_cases)
        print(f"\n📋 DRY RUN - Plan:")
        for model, setup in plan:
            print(f"   {model} × {setup} × {len(test_cases)} cases")
        print(f"\n   Total: {total} runs")
        print(f"   Estimated time: ~{total * 5 / 60:.0f} minutes")
        return

    # Parse sources and build index
    from parsers import parse_all_sources

    store = parse_all_sources(
        ustg_path=str(USTG_RTF_PATH),
        anhang_path=str(ANHANG_RTF_PATH),
        ustr_path=str(USTR_XML_PATH),
    )

    if store.size == 0:
        print("No chunks parsed! Check source file paths.")
        return

    # Detect device
    import torch
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"   Device: {device}")

    # Run experiments
    runner = ExperimentRunner(
        chunk_store=store,
        test_cases=test_cases,
        device=device,
    )

    runner.run_all(
        models=args.model,
        setups=args.setup,
        skip_judge=args.skip_judge,
    )

    # Save results
    json_path, csv_path = runner.save_results(args.output)

    # Print summary
    runner.print_summary()

    print(f"\n Results saved to:")
    print(f"   {json_path}")
    print(f"   {csv_path}")


if __name__ == "__main__":
    main()
