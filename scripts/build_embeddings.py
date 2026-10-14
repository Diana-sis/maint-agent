import json
import time
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
METADATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "embedding_metadata.json"
)

MODEL_NAME = "BAAI/bge-small-zh-v1.5"
BATCH_SIZE = 16


chunks = []

with CHUNKS_PATH.open("r", encoding="utf-8") as chunks_file:
    for line in chunks_file:
        if line.strip():
            chunks.append(json.loads(line))

texts = [chunk["text"] for chunk in chunks]

print(f"加载文本块：{len(texts)}")
print(f"嵌入模型：{MODEL_NAME}")
print(f"批次大小：{BATCH_SIZE}")

model = SentenceTransformer(
    MODEL_NAME,
    device="cpu",
)

start_time = time.perf_counter()

embeddings = model.encode(
    texts,
    batch_size=BATCH_SIZE,
    normalize_embeddings=True,
    show_progress_bar=True,
    convert_to_numpy=True,
)

elapsed_seconds = time.perf_counter() - start_time
embeddings = embeddings.astype(np.float32)

np.save(EMBEDDINGS_PATH, embeddings)

metadata = {
    "model_name": MODEL_NAME,
    "chunk_count": len(chunks),
    "embedding_dimension": int(embeddings.shape[1]),
    "normalized": True,
    "batch_size": BATCH_SIZE,
    "elapsed_seconds": round(elapsed_seconds, 2),
}

METADATA_PATH.write_text(
    json.dumps(metadata, ensure_ascii=False, indent=2),
    encoding="utf-8",
)

print(f"向量形状：{embeddings.shape}")
print(f"耗时：{elapsed_seconds:.2f}秒")
print(f"向量文件：{EMBEDDINGS_PATH}")
print(f"元数据文件：{METADATA_PATH}")