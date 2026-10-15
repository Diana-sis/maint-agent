import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI


PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")

api_key = os.getenv("DEEPSEEK_API_KEY")
base_url = os.getenv(
    "DEEPSEEK_BASE_URL",
    "https://api.deepseek.com",
)
model = os.getenv(
    "DEEPSEEK_MODEL",
    "deepseek-flash",
)

if not api_key:
    raise RuntimeError(
        "未找到 DEEPSEEK_API_KEY，请检查项目根目录下的 .env 文件。"
    )

client = OpenAI(
    api_key=api_key,
    base_url=base_url,
    timeout=30.0,
)

response = client.chat.completions.create(
    model=model,
    messages=[
        {
            "role": "system",
            "content": "你是一个API连通性测试助手。",
        },
        {
            "role": "user",
            "content": "只回复：API连接成功",
        },
    ],
    max_tokens=32,
    temperature=0,
)

answer = response.choices[0].message.content

print(f"模型：{model}")
print(f"响应：{answer}")