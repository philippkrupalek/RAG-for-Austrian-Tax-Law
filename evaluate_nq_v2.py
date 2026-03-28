"""
evaluate_nq_v4.py – NQ Retrieval Benchmark, Voller Wikipedia-Korpus, NUR DENSE (BGE-M3)
                     Direkt vergleichbar mit DPR (Karpukhin et al. 2020).

Kein BM25 – der BM25 Index für 21M Passages braucht 200-400GB RAM (Python Dict Overhead).
Dense-only ist methodisch sauber: DPR ist auch pure dense retrieval.

Hardware-Anforderungen:
    - RAM: ~80GB für Embeddings (21M × 1024-dim × float32)
    - VRAM: 24GB reicht für Encoding (Batches à 512)
    - Encoding-Zeit: ~2-4 Stunden

Download:
    wget https://dl.fbaipublicfiles.com/dpr/wikipedia_split/psgs_w100.tsv.gz
    gunzip psgs_w100.tsv.gz

Verwendung:
    python evaluate_nq_v4.py \\
        --wiki_file psgs_w100.tsv \\
        --nq_file biencoder-nq-dev.json \\
        --output nq_v4_results.json

Benchmark-Zielzahlen (MINDER Table 1, voller Korpus):
    BM25                   43.6%  @5
    DPR (Karpukhin 2020)   68.3%  @5   ← unser Hauptvergleich
    MINDER                 65.8%  @5
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np


# ============================================================================
# ARGUMENT PARSING
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="NQ Retrieval Benchmark – Dense only (BGE-M3), voller Wikipedia-Korpus"
    )
    parser.add_argument("--wiki_file", type=str, default="psgs_w100.tsv")
    parser.add_argument("--nq_file", type=str, default="biencoder-nq-dev.json")
    parser.add_argument(
        "--max_passages", type=int, default=None,
        help="Max Passages (None=alle 21M). Für Test: 1000000"
    )
    parser.add_argument("--max_questions", type=int, default=None)
    parser.add_argument("--output", type=str, default="nq_v4_results.json")
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--top_k", type=int, default=100)
    parser.add_argument(
        "--embedding_cache", type=str, default="nq_v4_embeddings.npy",
        help="Cache-Datei für Corpus-Embeddings (überspringt Encoding beim 2. Lauf)"
    )
    parser.add_argument(
        "--passage_ids_cache", type=str, default="nq_v4_passage_ids.json"
    )
    return parser.parse_args()


# ============================================================================
# LOAD WIKIPEDIA CORPUS
# ============================================================================

def load_wiki_corpus(tsv_file: str, max_passages: Optional[int] = None) -> Dict[str, str]:
    if not Path(tsv_file).exists():
        print(f"❌ Nicht gefunden: {tsv_file}")
        print("   wget https://dl.fbaipublicfiles.com/dpr/wikipedia_split/psgs_w100.tsv.gz")
        print("   gunzip psgs_w100.tsv.gz")
        sys.exit(1)

    print(f"\n📂 Lade Wikipedia-Korpus: {tsv_file}")
    corpus: Dict[str, str] = {}
    start = time.time()

    with open(tsv_file, "r", encoding="utf-8") as f:
        f.readline()  # skip header
        for line_num, line in enumerate(f, start=1):
            if max_passages and line_num > max_passages:
                break
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                continue
            pid = parts[0].strip()
            text = parts[1].strip() if len(parts) > 1 else ""
            title = parts[2].strip() if len(parts) > 2 else ""
            corpus[pid] = f"{title}. {text}" if title else text

            if line_num % 1_000_000 == 0:
                elapsed = time.time() - start
                print(f"   {line_num:,} Passages ({elapsed:.0f}s)")

    elapsed = time.time() - start
    print(f"   ✅ {len(corpus):,} Passages geladen in {elapsed:.1f}s")
    return corpus


# ============================================================================
# LOAD NQ QUESTIONS
# ============================================================================

def load_nq_questions(nq_file: str, max_questions: Optional[int] = None):
    import gzip
    if not Path(nq_file).exists():
        print(f"❌ Nicht gefunden: {nq_file}")
        sys.exit(1)

    print(f"\n📂 Lade NQ Fragen: {nq_file}")
    if nq_file.endswith(".gz"):
        with gzip.open(nq_file, "rt", encoding="utf-8") as f:
            data = json.load(f)
    else:
        with open(nq_file, "r", encoding="utf-8") as f:
            data = json.load(f)

    print(f"   Gesamt: {len(data)} Fragen")
    if max_questions:
        data = data[:max_questions]

    questions, positive_ids = [], []
    for item in data:
        questions.append(item["question"])
        pos_ids = {str(ctx["passage_id"]) for ctx in item.get("positive_ctxs", [])}
        positive_ids.append(pos_ids)

    print(f"   Evaluations-Fragen: {len(questions)}")
    return questions, positive_ids


# ============================================================================
# DENSE RETRIEVER (BGE-M3) mit Embedding-Cache
# ============================================================================

class DenseRetriever:

    def __init__(self, model_name="BAAI/bge-m3", batch_size=512,
                 embedding_cache=None, passage_ids_cache=None):
        self.model_name = model_name
        self.batch_size = batch_size
        self.embedding_cache = embedding_cache
        self.passage_ids_cache = passage_ids_cache
        self.model = None
        self._use_flag = False
        self.corpus_embeddings: Optional[np.ndarray] = None
        self.passage_ids: List[str] = []

    def _load_model(self):
        if self.model is not None:
            return
        print(f"   Dense: Lade {self.model_name}...")
        try:
            from FlagEmbedding import BGEM3FlagModel
            self.model = BGEM3FlagModel(self.model_name, use_fp16=True)
            self._use_flag = True
            print("   Dense: FlagEmbedding geladen (fp16)")
        except ImportError:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(self.model_name)
            self._use_flag = False
            print("   Dense: SentenceTransformers geladen")

    def _encode(self, texts: List[str]) -> np.ndarray:
        if not self._use_flag:
            embs = self.model.encode(
                texts, batch_size=self.batch_size,
                show_progress_bar=False, normalize_embeddings=True
            )
            return embs.astype(np.float32)
        else:
            result = self.model.encode(texts, batch_size=self.batch_size, max_length=512)
            embs = result['dense_vecs'].astype(np.float32)
            norms = np.linalg.norm(embs, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1, norms)
            return embs / norms

    def build_index(self, corpus: Dict[str, str]):
        """
        Encoding direkt in np.memmap (memory-mapped file auf Disk).
        Kein RAM-Aufbau während des Encodings — jeder Batch wird sofort auf Disk geschrieben.
        RAM-Bedarf während Encoding: nur ein Batch (~batch_size × 1024 × 4 Byte).
        """
        EMB_DIM = 1024
        cache_file = self.embedding_cache or "nq_v4_embeddings.npy"
        ids_file = self.passage_ids_cache or "nq_v4_passage_ids.json"
        # memmap wird als .dat gespeichert (npy-Format nicht kompatibel mit memmap write-mode)
        mmap_file = cache_file.replace(".npy", ".dat")

        # Try loading from cache
        if (Path(mmap_file).exists() and Path(ids_file).exists()):
            print(f"   Dense: Lade Cache: {mmap_file}")
            with open(ids_file, "r") as f:
                self.passage_ids = json.load(f)
            n = len(self.passage_ids)
            if n == len(corpus):
                self.corpus_embeddings = np.memmap(
                    mmap_file, dtype="float32", mode="r", shape=(n, EMB_DIM)
                )
                print(f"   Dense: Cache OK ({n:,} Passages, Shape: {self.corpus_embeddings.shape})")
                return
            else:
                print(f"   Dense: Cache-Größe passt nicht ({n:,} vs {len(corpus):,}), neu encodieren...")

        self._load_model()
        self.passage_ids = list(corpus.keys())
        texts = [corpus[pid] for pid in self.passage_ids]
        n = len(texts)

        disk_gb = n * EMB_DIM * 4 / 1e9
        eta_min = n / self.batch_size * 0.05 / 60
        print(f"   Dense: Encode {n:,} Passages")
        print(f"   Dense: Disk-Bedarf: ~{disk_gb:.0f}GB ({mmap_file})")
        print(f"   Dense: Geschätzte Zeit: ~{eta_min:.0f} Minuten")
        print(f"   Dense: RAM während Encoding: minimal (memmap, batch-weise auf Disk)")
        start = time.time()

        # Erstelle memmap direkt auf Disk — kein RAM-Aufbau
        self.corpus_embeddings = np.memmap(
            mmap_file, dtype="float32", mode="w+", shape=(n, EMB_DIM)
        )

        for i in range(0, n, self.batch_size):
            batch = texts[i:i + self.batch_size]
            embs = self._encode(batch)
            self.corpus_embeddings[i:i + len(embs)] = embs

            if (i // self.batch_size) % 200 == 0 and i > 0:
                elapsed = time.time() - start
                pct = i / n * 100
                eta = (elapsed / i) * (n - i)
                print(f"   Dense: {pct:.1f}% ({i:,}/{n:,}) | "
                      f"Elapsed: {elapsed/60:.1f}min | ETA: {eta/60:.1f}min")

        # Flush to disk
        self.corpus_embeddings.flush()
        elapsed = time.time() - start
        print(f"   Dense: ✅ Fertig in {elapsed/60:.1f}min, Shape: {self.corpus_embeddings.shape}")

        # Save passage IDs
        with open(ids_file, "w") as f:
            json.dump(self.passage_ids, f)
        print(f"   Dense: Passage-IDs gespeichert: {ids_file} ✅")

    def search(self, query: str, top_k: int = 100) -> List[Tuple[str, float]]:
        q_emb = self._encode([query])[0]
        scores = self.corpus_embeddings @ q_emb
        top_indices = np.argpartition(scores, -min(top_k, len(scores)))[-top_k:]
        top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]
        return [(self.passage_ids[i], float(scores[i])) for i in top_indices]


# ============================================================================
# HITS@K
# ============================================================================

def compute_hits_at_k(retrieved: List[str], relevant: Set[str], k_values: List[int]):
    return {k: 1 if set(retrieved[:k]) & relevant else 0 for k in k_values}


# ============================================================================
# EVALUATION LOOP
# ============================================================================

def evaluate_dense(
    questions, positive_ids, dense: DenseRetriever,
    top_k=100, k_values=[5, 20, 100]
) -> Dict:
    print(f"\n{'='*65}")
    print("📊 Evaluiere: Dense-only (BGE-M3)")
    print(f"{'='*65}")

    hits = {k: 0 for k in k_values}
    total = len(questions)
    start = time.time()

    for i, (question, pos_ids) in enumerate(zip(questions, positive_ids)):
        results = dense.search(question, top_k=top_k)
        retrieved_ids = [pid for pid, _ in results]
        h = compute_hits_at_k(retrieved_ids, pos_ids, k_values)
        for k in k_values:
            hits[k] += h[k]

        if (i + 1) % 200 == 0 or (i + 1) == total:
            elapsed = time.time() - start
            eta = (elapsed / (i + 1)) * (total - i - 1)
            h20 = hits[20] / (i + 1) * 100
            print(f"   [{i+1:4d}/{total}] Hits@20={h20:.1f}% | "
                  f"Elapsed: {elapsed:.0f}s | ETA: {eta:.0f}s")

    metrics = {f"hits@{k}": hits[k] / total * 100 for k in k_values}
    metrics["total_questions"] = total
    metrics["setup"] = "Dense-only (BGE-M3)"

    print(f"\n   📈 Ergebnisse:")
    for k in k_values:
        print(f"      Hits@{k:3d}: {metrics[f'hits@{k}']:.1f}%")

    return metrics


# ============================================================================
# RESULTS TABLE
# ============================================================================

BENCHMARKS = {
    "BM25 (MINDER Table 1)":       {"hits@5": 43.6, "hits@20": 62.9, "hits@100": 78.1},
    "DPR (Karpukhin et al. 2020)": {"hits@5": 68.3, "hits@20": 80.1, "hits@100": 86.1},
    "GAR (Mao et al. 2021)":       {"hits@5": 59.3, "hits@20": 73.9, "hits@100": 85.0},
    "MINDER (Li et al. 2023)":     {"hits@5": 65.8, "hits@20": 78.3, "hits@100": 86.7},
}


def print_results_table(results: Dict, k_values=[5, 20, 100]):
    print("\n" + "=" * 70)
    print("ERGEBNISTABELLE – NQ Dev, Voller Wikipedia-Korpus (21M Passages)")
    print("=" * 70)
    header = f"{'Setup':<40}"
    for k in k_values:
        header += f"  Hits@{k:<4}"
    print(header)
    print("-" * 70)

    print("  ▶ Dieses System")
    row = f"    {'BGE-M3 Dense-only':<36}"
    for k in k_values:
        row += f"  {results[f'hits@{k}']:6.1f}%"
    print(row)

    print("-" * 70)
    print("  ▶ Publizierte Baselines (MINDER Table 1, voller Korpus)")
    for name, bench in BENCHMARKS.items():
        row = f"    {name:<36}"
        for k in k_values:
            v = bench.get(f"hits@{k}")
            row += f"  {v:6.1f}%" if v else f"  {'–':>7}"
        print(row)

    print("=" * 70)
    print("\nHinweis: Voller Wikipedia-Korpus (psgs_w100.tsv, ~21M Passages)")
    print("         NQ Dev Split, 6515 Fragen (DPR-Format)")


# ============================================================================
# MAIN
# ============================================================================

def main():
    args = parse_args()

    print("=" * 65)
    print("NQ Dev Benchmark – Dense-only (BGE-M3), Voller Korpus")
    print(f"Wikipedia:  {args.wiki_file}")
    print(f"NQ-Fragen:  {args.nq_file}")
    print(f"Passages:   {args.max_passages or 'alle (~21M)'}")
    print(f"Batch-Size: {args.batch_size}")
    print(f"Cache:      {args.embedding_cache}")
    print("=" * 65)

    # 1) Load corpus
    corpus = load_wiki_corpus(args.wiki_file, max_passages=args.max_passages)

    # 2) Load questions
    questions, positive_ids = load_nq_questions(args.nq_file, max_questions=args.max_questions)

    # 3) Build Dense index
    print(f"\n🔧 Baue Dense Index (BGE-M3)...")
    dense = DenseRetriever(
        model_name="BAAI/bge-m3",
        batch_size=args.batch_size,
        embedding_cache=args.embedding_cache,
        passage_ids_cache=args.passage_ids_cache,
    )
    dense.build_index(corpus)

    # 4) Evaluate
    results = evaluate_dense(
        questions=questions,
        positive_ids=positive_ids,
        dense=dense,
        top_k=args.top_k,
        k_values=[5, 20, 100],
    )

    # 5) Print table
    print_results_table(results)

    # 6) Save
    output = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "corpus": "psgs_w100.tsv (voller DPR Wikipedia-Dump, ~21M Passages)",
        "embedding_model": "BAAI/bge-m3",
        "setup": "Dense-only (kein BM25)",
        "results": results,
        "published_benchmarks": BENCHMARKS,
    }
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n💾 Ergebnisse gespeichert: {args.output}")
    print("✅ Fertig!")


if __name__ == "__main__":
    main()