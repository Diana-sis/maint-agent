import json
import statistics
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CORPUS_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "pages.jsonl"
)


records = []

with CORPUS_PATH.open("r", encoding="utf-8") as corpus_file:
    for line in corpus_file:
        records.append(json.loads(line))

lengths = [record["character_count"] for record in records]
sorted_lengths = sorted(lengths)


def percentile(percent: float) -> int:
    index = int((len(sorted_lengths) - 1) * percent)
    return sorted_lengths[index]


print(f"页面数量：{len(records)}")
print(f"最短页面：{min(lengths)} 字符")
print(f"平均长度：{statistics.mean(lengths):.1f} 字符")
print(f"中位数：{statistics.median(lengths):.1f} 字符")
print(f"P90：{percentile(0.90)} 字符")
print(f"P95：{percentile(0.95)} 字符")
print(f"最长页面：{max(lengths)} 字符")
print(
    "超过1000字符的页面："
    f"{sum(length > 1000 for length in lengths)}"
)

print("\n最长的5个页面：")

for record in sorted(
    records,
    key=lambda item: item["character_count"],
    reverse=True,
)[:5]:
    print(
        f"PDF第{record['page_number']}页："
        f"{record['character_count']}字符"
    )

page_568 = next(
    record
    for record in records
    if record["page_number"] == 568
)

print(
    f"\nOCF所在的PDF第568页："
    f"{page_568['character_count']}字符"
)