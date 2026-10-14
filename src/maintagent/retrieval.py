import json
from dataclasses import dataclass
from pathlib import Path

import jieba
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer


MODEL_NAME = "BAAI/bge-small-zh-v1.5"
QUERY_INSTRUCTION = "为这个句子生成表示以用于检索相关文章："

ANSWER_SECTION_BOOST = 3.0
NAVIGATION_DOT_THRESHOLD = 20
NAVIGATION_PENALTY = 5.0

RRF_K = 60
BM25_WEIGHT = 1.2
DENSE_WEIGHT = 1.0


@dataclass(frozen=True)
class SearchResult:
    chunk_id: str
    page_number: int
    chunk_index: int
    text: str
    rrf_score: float
    bm25_rank: int | None
    dense_rank: int | None


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


class HybridRetriever:
    def __init__(
        self,
        chunks_path: Path,
        embeddings_path: Path,
        device: str = "cpu",
    ) -> None:
        self.chunks = load_jsonl(chunks_path)
        self.embeddings = np.load(embeddings_path)

        if len(self.chunks) != len(self.embeddings):
            raise ValueError(
                "文本块数量与向量数量不一致，"
                "请重新构建语料和向量"
            )

        tokenized_corpus = [
            tokenize(chunk["text"])
            for chunk in self.chunks
        ]

        self.bm25 = BM25Okapi(tokenized_corpus)
        self.model = SentenceTransformer(
            MODEL_NAME,
            device=device,
        )

    def _get_bm25_scores(
        self,
        query: str,
    ) -> np.ndarray:
        scores = self.bm25.get_scores(
            tokenize(query)
        )

        for index, chunk in enumerate(self.chunks):
            text = chunk["text"]
            has_causes = "可能原因" in text
            has_solutions = "解决措施" in text

            if has_causes and has_solutions:
                scores[index] += ANSWER_SECTION_BOOST

            if (
                text.count(".")
                >= NAVIGATION_DOT_THRESHOLD
            ):
                scores[index] -= NAVIGATION_PENALTY

        return scores

    def _get_dense_scores(
        self,
        query: str,
    ) -> np.ndarray:
        query_embedding = self.model.encode(
            QUERY_INSTRUCTION + query,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )

        return self.embeddings @ query_embedding

    def search(
        self,
        query: str,
        top_k: int = 5,
        candidate_k: int = 50,
    ) -> list[SearchResult]:
        query = query.strip()

        if not query:
            raise ValueError("查询内容不能为空")

        bm25_scores = self._get_bm25_scores(query)
        dense_scores = self._get_dense_scores(query)

        bm25_indices = sorted(
            range(len(bm25_scores)),
            key=lambda index: bm25_scores[index],
            reverse=True,
        )[:candidate_k]

        dense_indices = [
            int(index)
            for index in np.argsort(
                dense_scores
            )[::-1][:candidate_k]
        ]

        bm25_ranks = {
            index: rank
            for rank, index in enumerate(
                bm25_indices,
                start=1,
            )
        }
        dense_ranks = {
            index: rank
            for rank, index in enumerate(
                dense_indices,
                start=1,
            )
        }

        rrf_scores: dict[int, float] = {}

        for index, rank in bm25_ranks.items():
            rrf_scores[index] = (
                rrf_scores.get(index, 0.0)
                + BM25_WEIGHT / (RRF_K + rank)
            )

        for index, rank in dense_ranks.items():
            rrf_scores[index] = (
                rrf_scores.get(index, 0.0)
                + DENSE_WEIGHT / (RRF_K + rank)
            )

        result_indices = sorted(
            rrf_scores,
            key=rrf_scores.get,
            reverse=True,
        )[:top_k]

        results = []

        for index in result_indices:
            chunk = self.chunks[index]

            results.append(
                SearchResult(
                    chunk_id=chunk["id"],
                    page_number=chunk["page_number"],
                    chunk_index=chunk["chunk_index"],
                    text=chunk["text"],
                    rrf_score=rrf_scores[index],
                    bm25_rank=bm25_ranks.get(index),
                    dense_rank=dense_ranks.get(index),
                )
            )

        return results