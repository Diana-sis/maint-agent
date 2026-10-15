from dataclasses import dataclass

from maintagent.generation import GroundedAnswerGenerator
from maintagent.retrieval import HybridRetriever, SearchResult
from maintagent.validation import (
    CitationValidation,
    validate_citations,
)


@dataclass(frozen=True)
class AgentResponse:
    query: str
    answer: str
    evidence: tuple[SearchResult, ...]
    citation_validation: CitationValidation


class MaintenanceAgent:
    def __init__(
        self,
        retriever: HybridRetriever,
        generator: GroundedAnswerGenerator,
        top_k: int = 3,
    ) -> None:
        if top_k < 1:
            raise ValueError("top_k必须大于等于1。")

        self.retriever = retriever
        self.generator = generator
        self.top_k = top_k

    def run(self, query: str) -> AgentResponse:
        query = query.strip()

        if not query:
            raise ValueError("用户问题不能为空。")

        evidence = tuple(
            self.retriever.search(
                query=query,
                top_k=self.top_k,
            )
        )

        answer = self.generator.generate(
            query=query,
            results=evidence,
        )

        citation_validation = validate_citations(
            answer=answer,
            results=evidence,
        )

        if not citation_validation.is_valid:
            if not citation_validation.has_citations:
                reason = "回答中没有检测到页码引用"
            else:
                pages = ", ".join(
                    str(page)
                    for page
                    in citation_validation.unsupported_pages
                )
                reason = f"存在非检索来源页码：{pages}"

            raise RuntimeError(
                f"回答引用校验失败，已阻止输出：{reason}。"
            )

        return AgentResponse(
            query=query,
            answer=answer,
            evidence=evidence,
            citation_validation=citation_validation,
        )