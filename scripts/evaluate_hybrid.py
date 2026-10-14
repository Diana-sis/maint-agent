import json
from pathlib import Path

import jieba
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHUNKS_PATH = PROJECT_ROOT / "data" / "processed" / "chunks.jsonl"
EMBEDDINGS_PATH = PROJECT_ROOT / "data" / "processed" / "embeddings.npy"
QUESTIONS_PATH = (
    PROJECT_ROOT / "data" / "evaluation" / "retrieval_questions.jsonl"
)

MODEL_NAME = "BAAI/bge-small-zh-v1.5"
QUERY_INSTRUCTION = "为这个句子生成表示以用于检索相关文章："

TOP_K = 5
CANDIDATE_K = 50
RRF_K = 60
BM25_WEIGHT = 1.2
DENSE_WEIGHT = 1.0

ANSWER_SECTION_BOOST = 3.0
NAVIGATION_DOT_THRESHOLD = 20
NAVIGATION_PENALTY = 5.0


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as file:
        return [
            json.loads(line)
            for line in file
            if line.strip()
        ]


def tokenize(text: str) -> list[str]:
    return [
        token.strip().lower()
        for token in jieba.lcut(text)
        if token.strip()
    ]


chunks = load_jsonl(CHUNKS_PATH)
questions = load_jsonl(QUESTIONS_PATH)
chunk_embeddings = np.load(EMBEDDINGS_PATH)

tokenized_corpus = [
    tokenize(chunk["text"])
    for chunk in chunks
]
bm25 = BM25Okapi(tokenized_corpus)

model = SentenceTransformer(
    MODEL_NAME,
    device="cpu",
)

query_embeddings = model.encode(
    [
        QUERY_INSTRUCTION + question["question"]
        for question in questions
    ],
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
    bm25_scores = bm25.get_scores(
        tokenize(question["question"])
    )

    for index, chunk in enumerate(chunks):
        has_causes = "可能原因" in chunk["text"]
        has_solutions = "解决措施" in chunk["text"]

        if has_causes and has_solutions:
            bm25_scores[index] += ANSWER_SECTION_BOOST

        if (
            chunk["text"].count(".")
            >= NAVIGATION_DOT_THRESHOLD
        ):
            bm25_scores[index] -= NAVIGATION_PENALTY

    dense_scores = chunk_embeddings @ query_embedding

    bm25_indices = sorted(
        range(len(bm25_scores)),
        key=lambda index: bm25_scores[index],
        reverse=True,
    )[:CANDIDATE_K]

    dense_indices = np.argsort(
        dense_scores
    )[::-1][:CANDIDATE_K]

    rrf_scores = np.zeros(len(chunks), dtype=np.float32)

    for rank, index in enumerate(
        bm25_indices,
        start=1,
    ):
                rrf_scores[index] += (
            BM25_WEIGHT / (RRF_K + rank)
        )
    for rank, index in enumerate(
        dense_indices,
        start=1,
    ):
                rrf_scores[index] += (
            DENSE_WEIGHT / (RRF_K + rank)
        )

    top_indices = np.argsort(
        rrf_scores
    )[::-1][:TOP_K]

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

print("\nRRF混合检索结果")
print(f"Recall@1：{hit_at_1 / question_count:.3f}")
print(f"Recall@3：{hit_at_3 / question_count:.3f}")
print(f"MRR：{reciprocal_rank_sum / question_count:.3f}")