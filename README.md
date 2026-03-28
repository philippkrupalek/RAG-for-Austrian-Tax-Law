# UStG RAG — Retrieval-Augmented Generation for Austrian VAT Law

NOTICE: You can download the UStR2000_html.xml file (the guidelines) from https://findok.bmf.gv.at/findok/volltext(suche:Standardsuche)?dokumentId=c47353b7-028f-487c-9b78-2f1b53354e08

A RAG system for answering questions about Austrian Value Added Tax law (Umsatzsteuergesetz 1994). Built as part of my bachelor's thesis.

The system retrieves relevant legal provisions from the UStG, its Annex (Anhang/Binnenmarktregelung), and the Austrian VAT Guidelines (UStR 2000), then uses an LLM to generate grounded answers with precise legal citations.

## Why

LLMs know a lot about tax law, but they hallucinate the wrong provision numbers. In my evaluation, DeepSeek-V3 without RAG gets the *answer* right 79% of the time, but only cites the correct norm in 19% of cases (Citation F1). It'll say "yes, input VAT deduction applies" and then cite § 10 instead of § 4, or § 3a Abs. 3 instead of § 3a Abs. 11.

With RAG, citation quality goes from 19% to 58% F1. The system retrieves the actual legal text so the LLM can cite what it actually read, not what it vaguely remembers from training.

## Architecture

```
Question -> Reference Extraction -> Hybrid Retrieval (Dense BGE-M3, BM25 lexical matching)-> Boosting -> LLM -> Answer                                            
                      
```

**Sources** (parsed from raw files into hierarchical chunks: § -> Abs -> Z -> lit):
- UStG 1994 (RTF) — primary law, 978 chunks
- Anhang/Binnenmarkt (RTF) — EU single market provisions, 266 chunks
- UStR 2000 (XML) — guidelines with case law references, 1819 chunks

**Retrieval pipeline** (retriever.py):
1. Extract explicit § references from the question
2. Dense retrieval with BGE-M3 embeddings (FAISS)
3. BM25 sparse retrieval
4. Merge candidates, filter paragraph-level chunks
5. Boost chunks matching explicit references + UStR norm coupling
6. Keyword-based boost for § 3a (Leistungsort), maps ~70 keywords to 9 specific subsections
7. Optional cross-encoder reranking (bge-reranker-v2-m3)
8. Source balancing + paragraph-diversity dedup

**LLM** (llm.py): DeepSeek-V3 or Llama-3.1-8B via OpenAI-compatible API. Prompt instructs the model to cite with full granularity (§ X Abs. Y Z n lit. a) and avoid background norms.

## Evaluation

150 test cases derived from real rulings of the Austrian Federal Tax Court (BFG). Each case has a question, expected answer (Ja/Nein), and expected legal citations at Abs/Z/lit granularity.

**Ablation setups:**

| Setup | Components | Accuracy | Citation F1 | Recall@20 |
|-------|-----------|----------|-------------|-----------|
| S1 | Baseline (no RAG) | 79% | 19% | — |
| S2 | Dense + BM25 | 77% | 58% | 76% |
| S3 | + Cross-Encoder Rerank | — | — | — |
| S4 | + Query Rewrite + 2-Pass Backfill | — | — | — |

S3/S4 numbers not final yet. S2 is the current best setup — reranking hasn't improved results so far because the base retrieval is already quite targeted through explicit reference extraction and keyword boosting.

The accuracy drop from S1→S2 is expected and well-documented in RAG literature: sometimes the retrieved context misleads the LLM. The value of RAG here is the citation quality (+39pp F1).

## Files

| File | What it does |
|------|-------------|
| `main.py` | Interactive CLI, runs the full pipeline |
| `config.py` | Model configs, API keys, experiment setups, paths |
| `models.py` | Data classes: LegalReference, LegalChunk, ChunkStore, RetrievalResult |
| `parsers.py` | Parses UStG (RTF), Anhang (RTF), UStR (XML) into hierarchical chunks |
| `retriever.py` | Hybrid retrieval: dense + BM25 + boosting + reranking + dedup |
| `llm.py` | LLM integration: query rewrite, answer generation, citation extraction |
| `evaluate.py` | Evaluation framework: runs all setups × models, computes metrics |
| `evaluate_nq_v2.py` | NQ benchmark script (separate from the main system, validates BGE-M3 on standard benchmark) |

## Data Files

The legal source files are not included in the repo. You need:

- `UStG1994_rtf.rtf` — UStG 1994 full text (RTF export from RIS)
- `anhang_ustg.rtf` — Anhang/Binnenmarktregelung (RTF export from RIS)
- `UStR2000_html.xml` — UStR 2000 guidelines. Download as XML from [findok.bmf.gv.at](https://findok.bmf.gv.at/findok/volltext(suche:Standardsuche)?dokumentId=c47353b7-028f-487c-9b78-2f1b53354e08) and rename to `UStR2000_html.xml`. Direct link https://findok.bmf.gv.at/findok/volltext(suche:Standardsuche)?dokumentId=c47353b7-028f-487c-9b78-2f1b53354e08
- `golden_dataset.json` — 150 evaluation cases with BFG references

Place all files in the same directory as the Python scripts.

## Setup

```bash
pip install striprtf rank-bm25 langchain-community faiss-cpu \
            sentence-transformers openai beautifulsoup4 torch numpy

# API keys are in config.py, or can be set:
export DEEPSEEK_API_KEY="..."
export GROQ_API_KEY="..."       # for Llama
export OPENAI_API_KEY="..."     # for LLM-as-Judge evaluation

# run
python main.py                  # interactive mode, DeepSeek
python main.py --model llama    # use Llama instead
```

First run downloads BGE-M3 (~2.3GB) and builds the FAISS index + BM25 index (cached to `index_cache/`).

## Evaluation

```bash
python evaluate.py --model deepseek --setup S2 --skip-judge
python evaluate.py --model deepseek --setup S1          # baseline only
python evaluate.py --model deepseek llama --setup S1 S2 S3 S4  # full run
```

Results are saved as JSON in `results/` with per-case metrics.
