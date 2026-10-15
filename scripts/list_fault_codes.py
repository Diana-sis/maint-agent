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

FAULT_SECTION_START_PAGE = 544
HEADER_PATTERN = re.compile(
    r"^\[([^\]]+)\]\s+([A-Z][A-Z0-9]{1,7})\s*$",
    re.MULTILINE,
)

records = []

with PAGES_PATH.open("r", encoding="utf-8") as file:
    for line in file:
        if not line.strip():
            continue

        page = json.loads(line)
        page_number = page["page_number"]

        if page_number < FAULT_SECTION_START_PAGE:
            continue

        for title, code in HEADER_PATTERN.findall(
            page["text"]
        ):
            records.append(
                {
                    "page_number": page_number,
                    "title": title.strip(),
                    "code": code.strip(),
                }
            )

seen = set()

for record in records:
    key = (
        record["page_number"],
        record["code"],
    )

    if key in seen:
        continue

    seen.add(key)

    print(
        f"PDF第{record['page_number']}页 | "
        f"{record['code']:<8} | "
        f"{record['title']}"
    )

print()
print(f"共提取候选条目：{len(seen)}")