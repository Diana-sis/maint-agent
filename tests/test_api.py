import unittest

from fastapi.testclient import TestClient

from maintagent.agent import AgentResponse
from maintagent.api import app
from maintagent.retrieval import SearchResult
from maintagent.validation import CitationValidation


def make_response() -> AgentResponse:
    evidence = (
        SearchResult(
            chunk_id="page-568",
            page_number=568,
            chunk_index=1,
            text="OCF过电流测试证据",
            rrf_score=0.1,
            bm25_rank=1,
            dense_rank=1,
        ),
    )

    validation = CitationValidation(
        cited_pages=(568,),
        unsupported_pages=(),
    )

    return AgentResponse(
        query="OCF如何处理？",
        answer="检查设备[PDF第568页]。",
        evidence=evidence,
        citation_validation=validation,
    )


class FakeAgent:
    def run(self, query: str) -> AgentResponse:
        response = make_response()

        return AgentResponse(
            query=query,
            answer=response.answer,
            evidence=response.evidence,
            citation_validation=(
                response.citation_validation
            ),
        )


class FailingAgent:
    def run(self, query: str) -> AgentResponse:
        raise RuntimeError("生成服务暂时不可用")


class ApiTests(unittest.TestCase):
    def setUp(self) -> None:
        app.state.agent = FakeAgent()
        self.client = TestClient(app)

    def tearDown(self) -> None:
        self.client.close()

    def test_health_endpoint(self) -> None:
        response = self.client.get("/health")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json(),
            {"status": "ok"},
        )

    def test_ask_endpoint(self) -> None:
        response = self.client.post(
            "/ask",
            json={
                "question": "OCF如何处理？",
            },
        )

        self.assertEqual(response.status_code, 200)

        body = response.json()

        self.assertEqual(
            body["question"],
            "OCF如何处理？",
        )
        self.assertEqual(
            body["cited_pages"],
            [568],
        )
        self.assertEqual(
            body["evidence"][0]["page_number"],
            568,
        )

    def test_rejects_empty_question(self) -> None:
        response = self.client.post(
            "/ask",
            json={
                "question": "",
            },
        )

        self.assertEqual(response.status_code, 422)

    def test_converts_agent_error_to_502(self) -> None:
        app.state.agent = FailingAgent()

        response = self.client.post(
            "/ask",
            json={
                "question": "OCF如何处理？",
            },
        )

        self.assertEqual(response.status_code, 502)
        self.assertEqual(
            response.json()["detail"],
            "生成服务暂时不可用",
        )


if __name__ == "__main__":
    unittest.main()