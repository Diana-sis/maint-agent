import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHUNKS_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "chunks.jsonl"
)

if len(sys.argv) != 2:
    raise SystemExit(
        "用法：python scripts/show_page_chunks.py 页码"
    )

page_number = int(sys.argv[1])

with CHUNKS_PATH.open("r", encoding="utf-8") as file:
    chunks = [
        json.loads(line)
        for line in file
        if line.strip()
    ]

matches = [
    chunk
    for chunk in chunks
    if chunk["page_number"] == page_number
]

print(f"PDF第{page_number}页文本块数量：{len(matches)}")

for chunk in matches:
    print()
    print(
        f"--- 文本块{chunk['chunk_index']}，"
        f"{len(chunk['text'])}字符 ---"
    )
    print(chunk["text"])