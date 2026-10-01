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


if len(sys.argv) != 2:
    raise SystemExit(
        "用法：python scripts/show_pdf_page.py 页码"
    )

try:
    page_number = int(sys.argv[1])
except ValueError:
    raise SystemExit("页码必须是整数")

with pymupdf.open(PDF_PATH) as document:
    if not 1 <= page_number <= document.page_count:
        raise SystemExit(
            f"页码范围应为1至{document.page_count}"
        )

    page = document.load_page(page_number - 1)
    page_text = page.get_text("text")

print(f"PDF第 {page_number} 页")
print("-" * 60)
print(page_text)