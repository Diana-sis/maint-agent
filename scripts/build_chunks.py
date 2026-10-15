import json
import re
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
FAULT_SECTION_START_PAGE = 544

FAULT_HEADER_PATTERN = re.compile(
    r"^\[[^\]\n]+\]\s+[A-Z][A-Z0-9]{1,7}\s*$",
    re.MULTILINE,
)


def split_text(
    text: str,
) -> list[tuple[int, int, str]]:
    chunks = []
    start = 0

    while start < len(text):
        end = min(
            start + MAX_CHARACTERS,
            len(text),
        )

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
            chunks.append(
                (start, end, chunk_text)
            )

        if end >= len(text):
            break

        next_start = end - OVERLAP_CHARACTERS

        if next_start <= start:
            next_start = end

        start = next_start

    return chunks


def find_fault_entry_starts(
    text: str,
) -> list[int]:
    starts = []

    for match in FAULT_HEADER_PATTERN.finditer(text):
        following_text = text[
            match.end():match.end() + 160
        ]

        if "可能原因" in following_text:
            starts.append(match.start())

    return starts


def split_fault_page(
    text: str,
) -> list[tuple[int, int, str]]:
    entry_starts = find_fault_entry_starts(text)

    if not entry_starts:
        return split_text(text)

    chunks = []

    first_entry_start = entry_starts[0]
    prefix = text[:first_entry_start]

    if prefix.strip():
        for start, end, chunk_text in split_text(
            prefix
        ):
            chunks.append(
                (start, end, chunk_text)
            )

    for index, entry_start in enumerate(
        entry_starts
    ):
        if index + 1 < len(entry_starts):
            entry_end = entry_starts[index + 1]
        else:
            entry_end = len(text)

        entry_text = text[
            entry_start:entry_end
        ]

        for (
            local_start,
            local_end,
            chunk_text,
        ) in split_text(entry_text):
            chunks.append(
                (
                    entry_start + local_start,
                    entry_start + local_end,
                    chunk_text,
                )
            )

    return chunks


def split_page_text(
    text: str,
    page_number: int,
) -> list[tuple[int, int, str]]:
    if page_number >= FAULT_SECTION_START_PAGE:
        return split_fault_page(text)

    return split_text(text)


page_count = 0
chunk_count = 0
page_576_chunks = []

with PAGES_PATH.open(
    "r",
    encoding="utf-8",
) as pages_file:
    with CHUNKS_PATH.open(
        "w",
        encoding="utf-8",
    ) as chunks_file:
        for line in pages_file:
            page_record = json.loads(line)
            page_count += 1

            text_chunks = split_page_text(
                text=page_record["text"],
                page_number=page_record[
                    "page_number"
                ],
            )

            for chunk_index, (
                char_start,
                char_end,
                chunk_text,
            ) in enumerate(
                text_chunks,
                start=1,
            ):
                chunk_record = {
                    "id": (
                        f"{page_record['id']}"
                        f"-chunk-{chunk_index:03d}"
                    ),
                    "document_id": page_record[
                        "document_id"
                    ],
                    "document_title": page_record[
                        "document_title"
                    ],
                    "page_number": page_record[
                        "page_number"
                    ],
                    "chunk_index": chunk_index,
                    "char_start": char_start,
                    "char_end": char_end,
                    "text": chunk_text,
                    "character_count": len(
                        chunk_text
                    ),
                }

                chunks_file.write(
                    json.dumps(
                        chunk_record,
                        ensure_ascii=False,
                    )
                    + "\n"
                )

                chunk_count += 1

                if (
                    page_record["page_number"]
                    == 576
                ):
                    page_576_chunks.append(
                        chunk_record
                    )

print(f"处理页面：{page_count}")
print(f"生成文本块：{chunk_count}")
print(f"输出文件：{CHUNKS_PATH}")
print(
    "PDF第576页文本块数量："
    f"{len(page_576_chunks)}"
)

for chunk in page_576_chunks:
    print(
        f"第{chunk['chunk_index']}块："
        f"{chunk['character_count']}字符 | "
        f"{chunk['text'].splitlines()[0]}"
    )