"""
UStG RAG System

Setup:
1. Chunking:   § -> Abs -> Z -> lit with parent/child structure + preservation of the introductory sentence
2. Retrieval:  Dense (BGE-M3) + BM25 + Cross-Encoder Rerank
3. Boosting:   Explicit §-refs + UStR norm linking
4. LLM:        Query Rewrite + answer generation

Sources:
-) UStG 1994 (RTF) → PRIMARY
-) Anhang Binnenmarkt (RTF) → CONDITIONAL (EU only)
-) UStR 2000 (XML) → BRIDGE (explanations, case law)

Different models:
python main.py                    # DeepSeek (default)
python main.py --model llama      # Llama-3.1-8B via Groq
python main.py --model deepseek   # DeepSeek-V3 (explizit)

Requirements:
  pip install striprtf rank-bm25 langchain-community faiss-cpu \\
              sentence-transformers openai beautifulsoup4 torch numpy
"""

import os
import sys
import torch
import argparse
from pathlib import Path
from typing import Optional
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent))

# cuda

def get_device() -> str:
    if torch.cuda.is_available():
        try:
            gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
            if gpu_mem >= 4:
                print(f"GPU: {torch.cuda.get_device_name(0)} ({gpu_mem:.1f}GB)")
                return 'cuda'
        except:
            pass
    print("CPU")
    return 'cpu'


# config

DEVICE = get_device()

from config import (
    USTG_RTF_PATH, ANHANG_RTF_PATH, USTR_XML_PATH,
    EMBEDDING_MODEL,
)


# main

def main():
# Argument parsing
    parser = argparse.ArgumentParser(description="UStG RAG v2.0")
    parser.add_argument('--model', choices=['deepseek', 'llama'], default='deepseek',
                       help='LLM für Antwortgenerierung (default: deepseek)')
    args = parser.parse_args()
    
# Select model config
    from config import DEEPSEEK_V3, LLAMA_8B
    
    if args.model == 'llama':
        model_config = LLAMA_8B
        api_key_env = "GROQ_API_KEY"
    else:
        model_config = DEEPSEEK_V3
        api_key_env = "DEEPSEEK_API_KEY"
    
    print("\n" + "=" * 70)
    print(f"UStG RAG System - {model_config.name}")
    print("=" * 70)
    
# Check API key
    if not os.getenv(api_key_env):
        print(f"\n  {api_key_env} nicht gesetzt!")
        print(f"   export {api_key_env}='...'")
        print("   query rewrite deactivated")
    
# Parse all sources
    print("\n" + "=" * 70)
    print("parse sources")
    print("=" * 70)
    
    from parsers import parse_all_sources
    
# Report file status
    print(f"   UStG:   {USTG_RTF_PATH} {'✅' if USTG_RTF_PATH.exists() else '❌ NICHT GEFUNDEN'}")
    print(f"   Anhang: {ANHANG_RTF_PATH} {'✅' if ANHANG_RTF_PATH.exists() else '❌ NICHT GEFUNDEN'}")
    print(f"   UStR:   {USTR_XML_PATH} {'✅' if USTR_XML_PATH.exists() else '❌ NICHT GEFUNDEN'}")
    
    if not USTR_XML_PATH.exists():
        print("\n   UStR not found!")
        print("   → UStR-Chunks missing, no guidelines/judication")
    
    store = parse_all_sources(
        ustg_path=str(USTG_RTF_PATH) if USTG_RTF_PATH.exists() else None,
        anhang_path=str(ANHANG_RTF_PATH) if ANHANG_RTF_PATH.exists() else None,
        ustr_path=str(USTR_XML_PATH) if USTR_XML_PATH.exists() else None,
    )
    
    if store.size == 0:
        print("\n no chunks found!")
        return
    
# Build hybrid retriever
    print("\n" + "=" * 70)
    print("build hybrid retriever")
    print("=" * 70)
    
    from retriever import HybridRetriever
    
    retriever = HybridRetriever(store, device=DEVICE)
    retriever.build(embedding_model=EMBEDDING_MODEL)
    
# Initialize LLM components
    print("\n" + "=" * 70)
    print("initialize LLM components")
    print("=" * 70)
    
    from llm import (
        QueryRewriter, AnswerGenerator, get_client,
        extract_cited_paragraphs, extract_cited_references,
        find_missing_paragraphs,
    )
    
    llm_client = None
    rewriter = None
    generator = None
    
    try:
        llm_client = get_client(model_config)
        rewriter = QueryRewriter(client=llm_client, model_config=model_config)
        generator = AnswerGenerator(client=llm_client, model_config=model_config)
        print(f"   ✅ LLM ready ({model_config.name})")
    except ValueError as e:
        print(f"   {e}")
        print("    retrieval only (w/o answer generation)")
        rewriter = QueryRewriter(client=None)
    
# Ready
    print("\n" + "=" * 70)
    print("SYSTEM READY")
    print("=" * 70)
    print("\n Architecture:")
    print("  1. Chunking:  § → Abs → Z → lit (Parent/Child + Einleitungssatz)")
    print("  2. Retrieval: Dense (BGE-M3) + BM25 → Merge + Boost → Rerank")
    print("  3. Boosting:  Explizite §-Refs + UStR Norm-Kopplung")
    print("  4. LLM:       Query Rewrite (safe) + answer")
    
    stats = store.stats()
    print(f"\n {stats['total']} Chunks:")
    for source, count in stats['by_source'].items():
        print(f"   {source}: {count}")
    for level, count in stats['by_level'].items():
        print(f"   {level}: {count}")
    
# Interactive loop
    print("\n Your questions ('exit' to leave):\n")
    
    while True:
        try:
            query = input("Question: ").strip()
            
            if not query or query.lower() in ('exit', 'quit', 'q'):
                print("\n Bye!")
                break
            
        # Special commands
            if query.startswith('/debug '):
                _debug_retrieval(query[7:], retriever, rewriter)
                continue
            
            if query == '/stats':
                print(f"\n {store.stats()}")
                continue
            
        # Pipeline
            print("\n Processing...")
            
        # 1. Query rewrite (original, rewritten)
            rewritten = None
            if rewriter:
                original, rewritten = rewriter.rewrite(query)
                if rewritten:
                    print(f"   Rewritten: '{rewritten[:80]}...'")
            
        # 2. Hybrid retrieval:
            results = retriever.retrieve(
                query=query,
                rewritten_query=rewritten,
                top_k=22,
                use_reranking=True,
            )
            
            if not results:
                print("\n No relevant sources found.")
                continue
            
        # 3. Generate answer with 2-pass backfill
            if generator:
                context, source_map = retriever.get_context_for_llm(results)
                response = generator.generate(query, context, source_map)
                answer = response['answer']
                
                # 2-Pass: Check for missing §§
                cited_paras = extract_cited_paragraphs(answer)
                cited_refs = extract_cited_references(answer)
                # Which §§ were actually in the retrieval context
                context_paras = set()
                for r in results:
                    if r.chunk.ref.paragraph:
                        context_paras.add(r.chunk.ref.paragraph)
                
                missing = find_missing_paragraphs(cited_paras, context_paras)
                
                if missing:
                    print(f"\n   2-Pass: LLM cites §§ {missing}")
                    
                # Backfill: inject missing § chunks with hierarchical ref matching
                    backfill_results = retriever.backfill_paragraphs(
                        missing_paras=missing,
                        existing_results=results,
                        query=query,
                        max_per_para=4,
                        cited_refs=cited_refs,
                    )
                    
                    if backfill_results:
                        all_results = results + backfill_results
                        print(f"   Backfilled {len(backfill_results)} chunks → {len(all_results)} total")
                        
                        context2, source_map2 = retriever.get_context_for_llm(all_results)
                        response = generator.generate(query, context2, source_map2)
                        answer = response['answer']
                        results = all_results
                    else:
                        print(f"   No chunks found for missing §§ ")
                
                print("\n" + "─" * 70)
                print(" Answer:")
                print("─" * 70)
                print(answer)
                
                # show ALL sources
                print("\n" + "─" * 70)
                dist = defaultdict(int)
                for r in results:
                    dist[r.chunk.source_type.value] += 1
                print(f"SOURCES ({len(results)}):")
                print(f"   UStG: {dist.get('UStG', 0)} | "
                      f"Anhang: {dist.get('Anhang', 0)} | "
                      f"UStR: {dist.get('UStR', 0)}")
                
                for r in results:
                    boost = f" [{r.boost_reason}]" if r.boost_reason else ""
                    print(f"   • {r.chunk.citation}{boost}")
                print("─" * 70)
            
            else:
                # Retrieval-only mode
                print("\n" + "─" * 70)
                print("RETRIEVAL RESULTS:")
                print("─" * 70)
                for i, r in enumerate(results):
                    boost = f" [{r.boost_reason}]" if r.boost_reason else ""
                    score = f"D:{r.dense_score:.3f} B:{r.sparse_score:.3f} R:{r.rerank_score:.3f}"
                    print(f"\n[{i+1}] {r.chunk.citation}{boost}")
                    print(f"    Score: {score} → {r.combined_score:.3f}")
                    print(f"    {r.chunk.text[:150]}...")
                print("─" * 70)
            
            print()
        
        except KeyboardInterrupt:
            print("\n\n Bye!")
            break
        except Exception as e:
            print(f"\n Error: {e}")
            import traceback
            traceback.print_exc()


def _debug_retrieval(query: str, retriever, rewriter):
    """Debug mode: show detailed retrieval scores"""
    print(f"\n debeug retrieval: '{query}'")
    
    rewritten = None
    if rewriter:
        original, rewritten = rewriter.rewrite(query)
        if rewritten:
            print(f"   Rewritten: '{rewritten}'")
    
    results = retriever.retrieve(query, rewritten_query=rewritten, top_k=25)
    
    print(f"\n{'Rank':<5} {'Score':<8} {'Dense':<8} {'BM25':<8} {'Rerank':<8} {'Boost':<15} {'Citation'}")
    print("─" * 100)
    
    for i, r in enumerate(results):
        print(f"{i+1:<5} {r.combined_score:<8.3f} {r.dense_score:<8.3f} "
              f"{r.sparse_score:<8.3f} {r.rerank_score:<8.3f} "
              f"{r.boost_reason or '-':<15} {r.chunk.citation}")
    
    print()


if __name__ == "__main__":
    main()
