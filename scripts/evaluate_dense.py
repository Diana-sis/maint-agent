import json
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
QUESTIONS_PATH = (
    PROJECT_ROOT
    / "data"
    / "evaluation"
    / "retrieval_questions.jsonl"
)

MODEL_NAME = "BAAI/bge-small-zh-v1.5"
QUERY_INSTRUCTION = "为这个句子生成表示以用于检索相关文章："
TOP_K = 5


def load_jsonl(path: Path) -> list[dict]:
    records = []

    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                records.append(json.loads(line))

    return records


chunks = load_jsonl(CHUNKS_PATH)
questions = load_jsonl(QUESTIONS_PATH)
chunk_embeddings = np.load(EMBEDDINGS_PATH)

if len(chunks) != len(chunk_embeddings):
    raise RuntimeError(
        "文本块数量与向量数量不一致"
    )

model = SentenceTransformer(
    MODEL_NAME,
    device="cpu",
)

query_texts = [
    QUERY_INSTRUCTION + question["question"]
    for question in questions
]

query_embeddings = model.encode(
    query_texts,
    normalize_embeddings=True,
    show_progress_bar=False,
    convert_to_numpy=True,
)

hit_at_1 = 0
hit_at_3 = 0
reciprocal_rank_sum = 0.0

for question, query_embedding in zip(
    questions,
    query_embeddings,
):
    scores = chunk_embeddings @ query_embedding
    top_indices = np.argsort(scores)[::-1][:TOP_K]

    retrieved_pages = [
        chunks[int(index)]["page_number"]
        for index in top_indices
    ]
    relevant_pages = set(question["relevant_pages"])

    first_relevant_rank = None

    for rank, page_number in enumerate(
        retrieved_pages,
        start=1,
    ):
        if page_number in relevant_pages:
            first_relevant_rank = rank
            break

    if first_relevant_rank == 1:
        hit_at_1 += 1

    if (
        first_relevant_rank is not None
        and first_relevant_rank <= 3
    ):
        hit_at_3 += 1

    if first_relevant_rank is not None:
        reciprocal_rank_sum += 1 / first_relevant_rank

    print(
        f"{question['id']} | "
        f"正确页：{sorted(relevant_pages)} | "
        f"Top 5：{retrieved_pages} | "
        f"首个正确排名：{first_relevant_rank}"
    )

question_count = len(questions)

print("\n纯向量检索结果")
print(f"Recall@1：{hit_at_1 / question_count:.3f}")
print(f"Recall@3：{hit_at_3 / question_count:.3f}")
print(
    f"MRR：{reciprocal_rank_sum / question_count:.3f}"
)