import unittest

from maintagent.agent import MaintenanceAgent
from maintagent.retrieval import SearchResult


def make_result(page_number: int) -> SearchResult:
    return SearchResult(
        chunk_id=f"page-{page_number}",
        page_number=page_number,
        chunk_index=1,
        text="测试证据",
        rrf_score=0.1,
        bm25_rank=1,
        dense_rank=1,
    )


class FakeRetriever:
    def __init__(self, results: list[SearchResult]) -> None:
        self.results = results
        self.last_query = None
        self.last_top_k = None

    def search(
        self,
        query: str,
        top_k: int,
    ) -> list[SearchResult]:
        self.last_query = query
        self.last_top_k = top_k
        return self.results[:top_k]


class FakeGenerator:
    def __init__(self, answer: str) -> None:
        self.answer = answer

    def generate(self, query, results) -> str:
        return self.answer


class MaintenanceAgentTests(unittest.TestCase):
    def test_returns_grounded_response(self) -> None:
        retriever = FakeRetriever(
            [
                make_result(568),
                make_result(573),
            ]
        )
        generator = FakeGenerator(
            "处理措施来自手册[PDF第568页]。"
        )
        agent = MaintenanceAgent(
            retriever=retriever,
            generator=generator,
            top_k=2,
        )

        response = agent.run("OCF如何处理？")

        self.assertEqual(response.query, "OCF如何处理？")
        self.assertEqual(len(response.evidence), 2)
        self.assertTrue(
            response.citation_validation.is_valid
        )
        self.assertEqual(retriever.last_top_k, 2)

    def test_blocks_unsupported_citation(self) -> None:
        agent = MaintenanceAgent(
            retriever=FakeRetriever(
                [make_result(568)]
            ),
            generator=FakeGenerator(
                "错误引用[PDF第999页]。"
            ),
        )

        with self.assertRaises(RuntimeError):
            agent.run("OCF如何处理？")

    def test_rejects_empty_query(self) -> None:
        agent = MaintenanceAgent(
            retriever=FakeRetriever(
                [make_result(568)]
            ),
            generator=FakeGenerator(
                "回答[PDF第568页]。"
            ),
        )

        with self.assertRaises(ValueError):
            agent.run("   ")

    def test_rejects_invalid_top_k(self) -> None:
        with self.assertRaises(ValueError):
            MaintenanceAgent(
                retriever=FakeRetriever([]),
                generator=FakeGenerator(""),
                top_k=0,
            )


if __name__ == "__main__":
    unittest.main()