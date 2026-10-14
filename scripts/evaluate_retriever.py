import json
from pathlib import Path

from maintagent.retrieval import HybridRetriever


PROJECT_ROOT = Path(__file__).resolve().parents[1]

CHUNKS_PATH = PROJECT_ROOT / "data" / "processed" / "chunks.jsonl"
EMBEDDINGS_PATH = PROJECT_ROOT / "data" / "processed" / "embeddings.npy"
QUESTIONS_PATH = (
    PROJECT_ROOT
    / "data"
    / "evaluation"
    / "retrieval_questions.jsonl"
)


def load_questions(path: Path) -> list[dict]:
    questions = []

    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                questions.append(json.loads(line))

    return questions


retriever = HybridRetriever(
    chunks_path=CHUNKS_PATH,
    embeddings_path=EMBEDDINGS_PATH,
)

questions = load_questions(QUESTIONS_PATH)

recall_at_1 = 0
recall_at_3 = 0
reciprocal_rank_sum = 0.0

for item in questions:
    results = retriever.search(
        query=item["question"],
        top_k=5,
    )

    retrieved_pages = [
        result.page_number
        for result in results
    ]
    relevant_pages = set(item["relevant_pages"])

    first_relevant_rank = None

    for rank, page_number in enumerate(retrieved_pages, start=1):
        if page_number in relevant_pages:
            first_relevant_rank = rank
            break

    if retrieved_pages and retrieved_pages[0] in relevant_pages:
        recall_at_1 += 1

    if any(
        page_number in relevant_pages
        for page_number in retrieved_pages[:3]
    ):
        recall_at_3 += 1

    if first_relevant_rank is not None:
        reciprocal_rank_sum += 1 / first_relevant_rank

    print(
        f"{item['id']} | "
        f"正确页：{sorted(relevant_pages)} | "
        f"Top 5：{retrieved_pages} | "
        f"首个正确排名：{first_relevant_rank}"
    )

question_count = len(questions)

print()
print("封装后的混合检索结果")
print(f"评测问题数：{question_count}")
print(f"Recall@1：{recall_at_1 / question_count:.3f}")
print(f"Recall@3：{recall_at_3 / question_count:.3f}")
print(f"MRR：{reciprocal_rank_sum / question_count:.3f}")