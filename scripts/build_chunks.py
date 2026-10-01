import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PAGES_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "pages.jsonl"
)
CHUNKS_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "chunks.jsonl"
)

MAX_CHARACTERS = 800
OVERLAP_CHARACTERS = 100
MIN_BOUNDARY_POSITION = 400


def split_text(text: str) -> list[tuple[int, int, str]]:
    chunks = []
    start = 0

    while start < len(text):
        end = min(start + MAX_CHARACTERS, len(text))

        if end < len(text):
            boundary = text.rfind(
                "\n",
                start + MIN_BOUNDARY_POSITION,
                end,
            )

            if boundary != -1:
                end = boundary

        chunk_text = text[start:end].strip()

        if chunk_text:
            chunks.append((start, end, chunk_text))

        if end >= len(text):
            break

        next_start = end - OVERLAP_CHARACTERS

        if next_start <= start:
            next_start = end

        start = next_start

    return chunks


page_count = 0
chunk_count = 0
page_568_chunks = []

with PAGES_PATH.open("r", encoding="utf-8") as pages_file:
    with CHUNKS_PATH.open("w", encoding="utf-8") as chunks_file:
        for line in pages_file:
            page_record = json.loads(line)
            page_count += 1

            text_chunks = split_text(page_record["text"])

            for chunk_index, (
                char_start,
                char_end,
                chunk_text,
            ) in enumerate(text_chunks, start=1):
                chunk_record = {
                    "id": (
                        f"{page_record['id']}"
                        f"-chunk-{chunk_index:03d}"
                    ),
                    "document_id": page_record["document_id"],
                    "document_title": page_record["document_title"],
                    "page_number": page_record["page_number"],
                    "chunk_index": chunk_index,
                    "char_start": char_start,
                    "char_end": char_end,
                    "text": chunk_text,
                    "character_count": len(chunk_text),
                }

                chunks_file.write(
                    json.dumps(
                        chunk_record,
                        ensure_ascii=False,
                    )
                    + "\n"
                )

                chunk_count += 1

                if page_record["page_number"] == 568:
                    page_568_chunks.append(chunk_record)

print(f"处理页面：{page_count}")
print(f"生成文本块：{chunk_count}")
print(f"输出文件：{CHUNKS_PATH}")
print(f"PDF第568页文本块数量：{len(page_568_chunks)}")

for chunk in page_568_chunks:
    print(
        f"第{chunk['chunk_index']}块："
        f"{chunk['character_count']}字符"
    )