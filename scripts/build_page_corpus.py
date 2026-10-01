import json
from pathlib import Path

import pymupdf


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PDF_PATH = (
    PROJECT_ROOT
    / "data"
    / "source_docs"
    / "ATV340_Programming_Manual_CN.pdf"
)
OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "pages.jsonl"
)
DOCUMENT_ID = "atv340-programming-manual-cn"
DOCUMENT_TITLE = "ATV340变频器中文编程手册"


def clean_text(text: str) -> str:
    cleaned_lines = []

    for line in text.splitlines():
        cleaned_line = " ".join(line.split())

        if cleaned_line:
            cleaned_lines.append(cleaned_line)

    return "\n".join(cleaned_lines)


OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

page_count = 0
total_characters = 0

with pymupdf.open(PDF_PATH) as document:
    with OUTPUT_PATH.open("w", encoding="utf-8") as output_file:
        for page_index, page in enumerate(document):
            text = clean_text(page.get_text("text"))

            if not text:
                continue

            record = {
                "id": f"{DOCUMENT_ID}-page-{page_index + 1:04d}",
                "document_id": DOCUMENT_ID,
                "document_title": DOCUMENT_TITLE,
                "page_number": page_index + 1,
                "text": text,
                "character_count": len(text),
            }

            output_file.write(
                json.dumps(record, ensure_ascii=False) + "\n"
            )

            page_count += 1
            total_characters += len(text)

print(f"输出文件：{OUTPUT_PATH}")
print(f"有效页数：{page_count}")
print(f"总字符数：{total_characters}")