import json
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
QUESTIONS_PATH = (
    PROJECT_ROOT
    / "data"
    / "evaluation"
    / "retrieval_questions.jsonl"
)
TOP_K = 5


def load_jsonl(path: Path) -> list[dict]:
    records = []

    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                records.append(json.loads(line))

    return records


def tokenize(text: str) -> list[str]:
    return [
        token.strip().lower()
        for token in jieba.lcut(text)
        if token.strip()
    ]


chunks = load_jsonl(CHUNKS_PATH)
questions = load_jsonl(QUESTIONS_PATH)

tokenized_corpus = [
    tokenize(chunk["text"])
    for chunk in chunks
]
bm25 = BM25Okapi(tokenized_corpus)

hit_at_1 = 0
hit_at_3 = 0
reciprocal_rank_sum = 0.0

for question in questions:
    scores = bm25.get_scores(
        tokenize(question["question"])
    )
    top_indices = sorted(
        range(len(scores)),
        key=lambda index: scores[index],
        reverse=True,
    )[:TOP_K]

    retrieved_pages = [
        chunks[index]["page_number"]
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

print("\n评测结果")
print(f"Recall@1：{hit_at_1 / question_count:.3f}")
print(f"Recall@3：{hit_at_3 / question_count:.3f}")
print(
    f"MRR：{reciprocal_rank_sum / question_count:.3f}"
)