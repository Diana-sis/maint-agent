from pathlib import Path

import pymupdf


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PDF_PATH = (
    PROJECT_ROOT
    / "data"
    / "source_docs"
    / "ATV340_Programming_Manual_CN.pdf"
)


if not PDF_PATH.exists():
    raise FileNotFoundError(f"找不到PDF文件：{PDF_PATH}")

with pymupdf.open(PDF_PATH) as document:
    first_page = document.load_page(0)
    first_page_text = first_page.get_text("text")

    print(f"文件路径：{PDF_PATH}")
    print(f"PDF页数：{document.page_count}")
    print("第一页文本：")
    print(first_page_text[:500])
