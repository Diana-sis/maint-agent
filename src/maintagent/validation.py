import re
from collections.abc import Sequence
from dataclasses import dataclass

from maintagent.retrieval import SearchResult


CITATION_PATTERN = re.compile(
    r"\[PDF第\s*(\d+)\s*页\]"
)


@dataclass(frozen=True)
class CitationValidation:
    cited_pages: tuple[int, ...]
    unsupported_pages: tuple[int, ...]

    @property
    def has_citations(self) -> bool:
        return bool(self.cited_pages)

    @property
    def is_valid(self) -> bool:
        return (
            self.has_citations
            and not self.unsupported_pages
        )


def validate_citations(
    answer: str,
    results: Sequence[SearchResult],
) -> CitationValidation:
    allowed_pages = {
        result.page_number
        for result in results
    }

    cited_pages = tuple(
        dict.fromkeys(
            int(page)
            for page in CITATION_PATTERN.findall(answer)
        )
    )

    unsupported_pages = tuple(
        page
        for page in cited_pages
        if page not in allowed_pages
    )

    return CitationValidation(
        cited_pages=cited_pages,
        unsupported_pages=unsupported_pages,
    )