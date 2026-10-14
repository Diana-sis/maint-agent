import json
import sys
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer


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

MODEL_NAME = "BAAI/bge-small-zh-v1.5"
QUERY_INSTRUCTION = "为这个句子生成表示以用于检索相关文章："
TOP_K = 5


if len(sys.argv) < 2:
    raise SystemExit(
        '用法：python scripts/search_dense.py "用户问题"'
    )

query = " ".join(sys.argv[1:]).strip()

chunks = []

with CHUNKS_PATH.open("r", encoding="utf-8") as chunks_file:
    for line in chunks_file:
        if line.strip():
            chunks.append(json.loads(line))

embeddings = np.load(EMBEDDINGS_PATH)

if len(chunks) != len(embeddings):
    raise RuntimeError(
        "文本块数量与向量数量不一致，请重新生成向量"
    )

model = SentenceTransformer(
    MODEL_NAME,
    device="cpu",
)

query_embedding = model.encode(
    QUERY_INSTRUCTION + query,
    normalize_embeddings=True,
    show_progress_bar=False,
    convert_to_numpy=True,
)

scores = embeddings @ query_embedding
top_indices = np.argsort(scores)[::-1][:TOP_K]

print(f"查询：{query}")
print("=" * 70)

for rank, chunk_index in enumerate(top_indices, start=1):
    chunk = chunks[int(chunk_index)]
    score = float(scores[chunk_index])

    print(
        f"Top {rank} | 相似度：{score:.4f} | "
        f"PDF第{chunk['page_number']}页 | "
        f"文本块{chunk['chunk_index']}"
    )
    print(chunk["text"][:600])
    print("-" * 70)