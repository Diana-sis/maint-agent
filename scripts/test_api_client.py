import json
from urllib.request import Request, urlopen


API_URL = "http://127.0.0.1:8000/ask"

payload = {
    "question": (
        "ATV340出现OCF过电流故障，"
        "应该如何处理？"
    )
}

request_body = json.dumps(
    payload,
    ensure_ascii=False,
).encode("utf-8")

request = Request(
    API_URL,
    data=request_body,
    headers={
        "Content-Type": "application/json; charset=utf-8",
        "Accept": "application/json",
    },
    method="POST",
)

with urlopen(request, timeout=120) as response:
    response_body = response.read().decode("utf-8")

result = json.loads(response_body)

print(f"问题：{result['question']}")
print(
    "引用页码："
    + ", ".join(
        str(page)
        for page in result["cited_pages"]
    )
)
print(
    "证据页码："
    + ", ".join(
        str(item["page_number"])
        for item in result["evidence"]
    )
)
print("回答：")
print(result["answer"])