import unittest

from maintagent.retrieval import SearchResult
from maintagent.validation import validate_citations


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


class CitationValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.results = [
            make_result(568),
            make_result(573),
        ]

    def test_accepts_supported_citation(self) -> None:
        validation = validate_citations(
            "过电流故障[PDF第568页]。",
            self.results,
        )

        self.assertTrue(validation.is_valid)
        self.assertEqual(validation.cited_pages, (568,))
        self.assertEqual(validation.unsupported_pages, ())

    def test_rejects_unsupported_citation(self) -> None:
        validation = validate_citations(
            "错误内容[PDF第999页]。",
            self.results,
        )

        self.assertFalse(validation.is_valid)
        self.assertEqual(
            validation.unsupported_pages,
            (999,),
        )

    def test_rejects_answer_without_citation(self) -> None:
        validation = validate_citations(
            "这段回答没有页码。",
            self.results,
        )

        self.assertFalse(validation.has_citations)
        self.assertFalse(validation.is_valid)

    def test_removes_duplicate_citations(self) -> None:
        validation = validate_citations(
            "原因[PDF第568页]，措施[PDF第568页]。",
            self.results,
        )

        self.assertEqual(validation.cited_pages, (568,))


if __name__ == "__main__":
    unittest.main()