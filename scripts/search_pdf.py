import sys
from pathlib import Path

import pymupdf


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PDF_PATH = (
    PROJECT_ROOT
    / "data"
    / "source_docs"
    / "ATV340_Programming_Manual_CN.pdf"
)
MAX_RESULTS = 10


if len(sys.argv) != 2:
    raise SystemExit(
        '用法：python scripts/search_pdf.py "搜索关键词"'
    )

query = sys.argv[1].strip()

if not query:
    raise SystemExit("搜索关键词不能为空")

with pymupdf.open(PDF_PATH) as document:
    result_count = 0

    for page_index, page in enumerate(document):
        page_text = " ".join(page.get_text("text").split())
        match_index = page_text.casefold().find(query.casefold())

        if match_index == -1:
            continue

        snippet_start = max(0, match_index - 80)
        snippet_end = min(len(page_text), match_index + len(query) + 120)
        snippet = page_text[snippet_start:snippet_end]

        print(f"[PDF第 {page_index + 1} 页]")
        print(snippet)
        print("-" * 60)

        result_count += 1

        if result_count >= MAX_RESULTS:
            break

print(f"共展示 {result_count} 条结果")