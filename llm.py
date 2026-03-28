"""
llm.py - LLM Integration for UStG RAG

Multi-Model Support:
- DeepSeek-V3 (deepseek-chat) - high-performance
- Llama-3.1-8B (llama-3.1-8b-instant via Groq) - weaker model
- Gemini 1.5 Flash - for Evaluation (LLM-as-Judge)
"""

import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Set
from openai import OpenAI

from models import RetrievalResult
from config import ModelConfig, DEEPSEEK_V3


# client factory

def get_client(model_config: ModelConfig) -> OpenAI:
    """Create OpenAI-compatible client for any model"""
    return OpenAI(
        api_key=model_config.api_key,
        base_url=model_config.base_url,
    )


def get_llm_client() -> OpenAI:
    """Create DeepSeek client"""
    return get_client(DEEPSEEK_V3)


# citation extractor (for 2-pass retrieval)

def extract_cited_paragraphs(answer: str) -> Set[str]:
    """
    Extract all § references cited in an LLM answer. Returns set of paragraph numbers: {"12", "19", "11"}
    """
    result: Set[str] = set()

    # "§§ 12 und 19" or "§§ 6, 10, 12" (comma/und separated list)
    plural_pattern = re.compile(
        r'§{1,2}\s*(\d+[a-z]?)(?:\s*(?:,|und|sowie|bzw\.?)\s*(\d+[a-z]?))*',
        re.IGNORECASE
    )
    for m in plural_pattern.finditer(answer):
        # Extract all numbers from the full match text
        for num in re.findall(r'\d+[a-z]?', m.group(0)):
            result.add(num)

    # Single "§ 12" references (catches anything the plural pattern missed)
    for m in re.finditer(r'§\s*(\d+[a-z]?)', answer, re.IGNORECASE):
        result.add(m.group(1))

    return result


def extract_cited_paragraphs_full(answer: str) -> List[str]:
    """
    Extract § references at Abs granularity for finer citation metrics. Returns list of strings like ["12.2", "3a.6", "19"] where the number
    after the dot is the Absatz number. Falls back to paragraph only if
    no Abs is present. Example: "§ 12 Abs. 2 Z 2 lit. a" -> "12.2"
    """
    refs = extract_cited_references(answer)
    result = []
    for r in refs:
        if r.absatz:
            result.append(f"{r.paragraph}.{r.absatz}")
        else:
            result.append(r.paragraph)
    return result


@dataclass
class CitedReference:
    """A structured reference extracted from LLM answer text"""
    paragraph: str
    absatz: Optional[str] = None
    ziffer: Optional[str] = None
    litera: Optional[str] = None
    
    @property
    def specificity(self) -> int:
        score = 1
        if self.absatz: score += 1
        if self.ziffer: score += 1
        if self.litera: score += 1
        return score


def extract_cited_references(answer: str) -> List[CitedReference]:
    """ Extract hierarchical references from LLM answer. 
    Matches: § 19 Abs. 2 Z 1 lit. b AND Art. 7 Abs. 1
    """
    # § references
    para_pattern = re.compile(
        r'(?:§|Paragraph)\s*(\d+[a-z]?)'
        r'(?:\s+Abs(?:atz)?\.?\s*(\d+[a-z]?))?'
        r'(?:\s+Z(?:iffer)?\.?\s*(\d+))?'
        r'(?:\s+lit(?:era)?\.?\s*([a-z]))?',
        re.IGNORECASE
    )
    
    # Art. references (Anhang)
    art_pattern = re.compile(
        r'Art(?:ikel)?\.?\s*(\d+[a-z]?)'
        r'(?:\s+Abs(?:atz)?\.?\s*(\d+[a-z]?))?'
        r'(?:\s+Z(?:iffer)?\.?\s*(\d+))?'
        r'(?:\s+lit(?:era)?\.?\s*([a-z]))?',
        re.IGNORECASE
    )
    
    refs = []
    seen = set()
    
    # Extract § references
    for m in para_pattern.finditer(answer):
        key = ('para', m.group(1), m.group(2), m.group(3), m.group(4))
        if key not in seen:
            refs.append(CitedReference(
                paragraph=m.group(1),
                absatz=m.group(2),
                ziffer=m.group(3),
                litera=m.group(4),
            ))
            seen.add(key)
    
    # Extract Art. references
    for m in art_pattern.finditer(answer):
    # skip "Art" that's part of other words
        start = m.start()
        if start > 0 and answer[start-1].isalpha():
            continue
        key = ('art', m.group(1), m.group(2), m.group(3), m.group(4))
        if key not in seen:
            # Store with "Art" prefix so paragraph matches Anhang chunk IDs
            art_para = f"Art{m.group(1)}"
            refs.append(CitedReference(
                paragraph=art_para,
                absatz=m.group(2),
                ziffer=m.group(3),
                litera=m.group(4),
            ))
            seen.add(key)
    
    refs.sort(key=lambda r: r.specificity, reverse=True)
    return refs


def find_missing_paragraphs(
    cited: Set[str], 
    context_paragraphs: Set[str],
) -> Set[str]:
    """Find §§ that the LLM cited but that were not in the retrieval context."""
    return cited - context_paragraphs


# Query Rewriter

class QueryRewriter:
    """Rewrites user queries into precise legal formulations."""
    
    REWRITE_PROMPT = """Du bist ein sehr erfahrener Experte für österreichisches Umsatzsteuerrecht. Extrahiere aus der folgenden Frage eine präzise Suchanfrage für ein Retrieval-System.
    REGELN: Formuliere die Rechtsfrage in einem Satz mit juristischen Fachbegriffen des UStG 1994
    Behalte explizit genannte §§ und Artikel bei
    Füge KEINE §§ hinzu, die nicht in der Frage stehen
    Maximal 25 Wörter
    Keine näheren Erklärungen, nur die Suchanfrage

    FRAGE: {query}

    SUCHANFRAGE:"""
    
    def __init__(self, client: Optional[OpenAI] = None, model_config: Optional[ModelConfig] = None):
        self.client = client
        self.model_config = model_config or DEEPSEEK_V3
        self._enabled = client is not None
    
    def rewrite(self, query: str) -> Tuple[str, Optional[str]]:
        if not self._enabled:
            return query, None
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_config.model_string,
                messages=[
                    {"role": "user", "content": self.REWRITE_PROMPT.format(query=query)}
                ],
                temperature=0,
                max_tokens=100,
            )
            
            rewritten = response.choices[0].message.content.strip()
            if not rewritten or len(rewritten) < 5:
                return query, None
            return query, rewritten
            
        except Exception as e:
            print(f"   ⚠️ Query rewrite failed: {e}")
            return query, None


# Answer Generator

class AnswerGenerator:
    """ Generates answers from retrieved context, also works with any OpenAI compatible API.
    """
    
    SYSTEM_PROMPT = """Du beantwortest Umsatzsteuer-Fragen nach oesterreichischem Recht (UStG 1994, UStR 2000, UStG_Anhang) auf Basis des bereitgestellten Kontexts. 
    Regeln fürs Zitieren (Strikt, ohne Ausnahmen befolgen):
    1. Zitiere nur die Normen, die den Sachverhalt DIREKT und UNMITTELBAR regeln.
    Zitiere NICHT:
        -) § 1 (Steuerbarkeit), ist fast nie die Hauptnorm
        -) § 2 (Unternehmerbegriff) — nur wenn der Unternehmerstatus selbst fraglich ist
        -) § 23 (Kleinunternehmer) — nur wenn Kleinunternehmerregelung direkt gefragt
        -) § 27 (Haftung/Fiskalvertreter) — nur wenn Haftung direkt gefragt
        -) Allgemeine Hintergrundnormen die nicht die konkrete Frage beantworten
    2. Zitiere immer, ohne Ausnahme, mit voller Granularitaet: '§ X Abs. Y Z n lit. a'. Niemals nur "§ 12" zitieren, immer den Absatz angeben sowie Zeile (Z) und litera (zum Beispiel "a"), falls juristisch korrekt.
    Falls es juristisch nicht richtig waere, muss natürlich keine zusaetzliche Hierarchiestufe wie Zeile oder lit. angegeben werden.!
    3. Bei Artikeln des Anhangs: "Art. X Abs. Y" verwenden (NICHT §).
    4. Nur so viele Normzitate wie notwendig pro Antwort. Weniger ist besser.

    ERGEBNIS-BESTIMMUNG (KRITISCH):
    - Pruefe ob die Voraussetzungen der Norm im konkreten Sachverhalt ERFUELLT sind.
    - Wenn eine Norm einen Anspruch gewaehrt und der Sachverhalt keinen konkreten Ausschlussgrund nennt: Ergebnis ist JA (Anspruch besteht), nicht "Nein weil theoretische Ausschlussgruende im Gesetz stehen"
    - Ausschlussgruende nur anwenden wenn sie im Sachverhalt tatsaechlich vorliegen. Im Zweifel: Wenn der Sachverhalt die Grundvoraussetzung erfuellt, dann JA.

    LEISTUNGSORT-FRAGEN:
    - § 3a Abs. 6: B2B-Grundregel Empfaengerort
    - § 3a Abs. 7: B2C-Grundregel → Unternehmerort  
    - § 3a Abs. 9: Grundstueck → Belegenheitsort
    - § 3a Abs. 11: Veranstaltungen/Seminare → Taetigkeitsort
    - § 3a Abs. 12: Befoerderungsmittel → Uebergabeort
    - § 3a Abs. 13: Elektronische Leistungen → Empfaengerort
    - Immer die speziellere Regel vor der allgemeinen pruefen, falls diese nicht zutrifft, kann natuerlich die allgemeine genommen werden..

    Format:
    a) Sachverhalt (1 Satz)
    b) Anwendbare Norm: Nur die notwendigen Normen mit voller Granularitaet (§ X Abs. Y Z n lit. a)
    c) Subsumtion: Tatbestandsmerkmale im Sachverhalt pruefen
    d) Ergebnis: Ja / Nein / [konkretes Ergebnis] mit praezisem Normzitat"""

    ANSWER_PROMPT = """Sachverhalt und Frage:
{query}

Kontext:
{context}

Beantworte praezise (a/b/c/d).
WICHTIG: 
-) Zitiere nicht mehr und nicht weniger als die notwendigen Normen mit voller Granularitaet (§ X Abs. Y Z n lit. a).
-) Bei Artikeln: "Art. X Abs. Y" verwenden.  
-) KEINE allgemeinen Hintergrundnormen zitieren (§ 1, § 2, § 23, § 27).
-) Wenn der Sachverhalt die Grundvoraussetzung erfuellt und kein konkreter Ausschlussgrund vorliegt, dann ist das Ergebnis JA."""

    BASELINE_PROMPT = """Du beantwortest Umsatzsteuer-Fragen nach österreichischem Recht (UStG 1994, UStR 2000)
auf Basis deines Wissens.

Aufgabe:
-) Beantworte die Frage klar und sachverhaltsbezogen.
-) Zitiere die relevanten Normen (§/Abs/Z/lit).
-) Erkläre kurz, warum diese Normen anwendbar sind.

Wichtig - Ausschlusstatbestände:
Wenn eine Norm grundsätzlich einen Anspruch gewaehrt (z.B. Vorsteuerabzug nach § 12 Abs. 1),
prüfe immer, ob die Ausschlusstatbestände im konkreten Sachverhalt tatsaechlich erfuellt sind.
-) Wenn kein Ausschlussgrund vorliegt, dann antworte "Ja, Vorsteuerabzug steht zu."
-) Nicht: "Nein, weil Ausschlussgründe theoretisch existieren."

Wichtig - Mehrwertige Ergebnisse:
-) "Wer schuldet die Steuer?" -> Prüfe ob DOPPELTE STEUERSCHULD besteht (§ 19 Abs. 1 + § 11 Abs. 12).
-) "Ex nunc oder ex tunc?" -> Antworte direkt mit dem Zeitpunkt, nicht "Unklar".
-) "Wer schuldet?" bei Bauleistungen / § 19 Abs. 1a → Leistungsempfänger (Reverse Charge).

Wichtig:
-) Wenn die Frage mehrere Teilprobleme hat, beantworte jeden Teil explizit.
-) Wenn du bei einer Norm unsicher bist, sage das offen.
-) Nur was für den konkreten Fall relevant ist, keine Lehrbuch-Generalien.

Format:
a) Sachverhalt (nicht mehr als das notwendigste)
b) Anwendbare Norm
c) Ausschlussprüfung (Ja/Nein + Begründung)
d) Ergebnis: Ja / Nein / [konkretes Ergebnis] mit kurzer Begründung und Normzitaten"""
    
    def __init__(self, client: Optional[OpenAI] = None, model_config: Optional[ModelConfig] = None):
        self.client = client or get_llm_client()
        self.model_config = model_config or DEEPSEEK_V3
    
    def generate(
        self, 
        query: str, 
        context: str, 
        source_map: Dict[int, str],
    ) -> Dict:
        """Generate answer with retrieval context (RAG mode)"""
        prompt = self.ANSWER_PROMPT.format(context=context, query=query)
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_config.model_string,
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=self.model_config.temperature,
                max_tokens=self.model_config.max_tokens,
            )
            
            return {
                'answer': response.choices[0].message.content,
                'source_map': source_map,
            }
        
        except Exception as e:
            return {
                'answer': f" LLM Error: {e}",
                'source_map': source_map,
            }
    
    def generate_baseline(self, query: str) -> Dict:
        """Generate answer WITHOUT context (Baseline S1)"""
        try:
            response = self.client.chat.completions.create(
                model=self.model_config.model_string,
                messages=[
                    {"role": "system", "content": self.BASELINE_PROMPT},
                    {"role": "user", "content": query},
                ],
                temperature=self.model_config.temperature,
                max_tokens=self.model_config.max_tokens,
            )
            
            return {
                'answer': response.choices[0].message.content,
                'source_map': {},
            }
        
        except Exception as e:
            return {
                'answer': f" LLM Error: {e}",
                'source_map': {},
            }
