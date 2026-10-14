import sys
from pathlib import Path

from maintagent.retrieval import HybridRetriever


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHUNKS_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "chunks.jsonl"
)
EMBEDDINGS_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "embeddings.npy"
)


if len(sys.argv) < 2:
    raise SystemExit(
        '用法：python scripts/search_hybrid.py "用户问题"'
    )

query = " ".join(sys.argv[1:]).strip()

retriever = HybridRetriever(
    chunks_path=CHUNKS_PATH,
    embeddings_path=EMBEDDINGS_PATH,
)

results = retriever.search(
    query=query,
    top_k=5,
)

print(f"查询：{query}")
print("=" * 70)

for rank, result in enumerate(results, start=1):
    print(
        f"Top {rank} | "
        f"RRF：{result.rrf_score:.6f} | "
        f"BM25排名：{result.bm25_rank} | "
        f"Dense排名：{result.dense_rank} | "
        f"PDF第{result.page_number}页"
    )
    print(result.text[:600])
    print("-" * 70)