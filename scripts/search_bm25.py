import json
import sys
from pathlib import Path

import jieba
from rank_bm25 import BM25Okapi


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHUNKS_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "chunks.jsonl"
)
TOP_K = 5


def tokenize(text: str) -> list[str]:
    return [
        token.strip().lower()
        for token in jieba.lcut(text)
        if token.strip()
    ]


if len(sys.argv) < 2:
    raise SystemExit(
        '用法：python scripts/search_bm25.py "用户问题"'
    )

query = " ".join(sys.argv[1:]).strip()

chunks = []

with CHUNKS_PATH.open("r", encoding="utf-8") as chunks_file:
    for line in chunks_file:
        chunks.append(json.loads(line))

tokenized_corpus = [
    tokenize(chunk["text"])
    for chunk in chunks
]

bm25 = BM25Okapi(tokenized_corpus)
query_tokens = tokenize(query)
scores = bm25.get_scores(query_tokens)

top_indices = sorted(
    range(len(scores)),
    key=lambda index: scores[index],
    reverse=True,
)[:TOP_K]

print(f"查询：{query}")
print(f"分词结果：{query_tokens}")
print("=" * 70)

for rank, chunk_index in enumerate(top_indices, start=1):
    chunk = chunks[chunk_index]
    score = scores[chunk_index]

    print(
        f"Top {rank} | 分数：{score:.4f} | "
        f"PDF第{chunk['page_number']}页 | "
        f"文本块{chunk['chunk_index']}"
    )
    print(chunk["text"][:600])
    print("-" * 70)