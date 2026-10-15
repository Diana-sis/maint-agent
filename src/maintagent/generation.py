import os
from collections.abc import Sequence

from openai import OpenAI

from maintagent.retrieval import SearchResult


SYSTEM_PROMPT = """
你是工业变频器维护辅助系统。

你必须遵守以下规则：
1. 只能依据用户提供的设备手册证据回答，不得补充证据中没有的信息。
2. 每项故障原因、处理措施和复位方法都必须标注来源页码，格式为[PDF第X页]。
3. 如果证据不足，必须明确说明“当前手册证据不足，无法确定”，不得猜测。
4. 将警告、错误代码和普通参数说明区分开。
5. 不得建议绕过保护机制、带电操作或忽略安全规范。
6. 回答不能替代具备资质的现场维护人员。

请按以下结构回答：
故障判断：
可能原因：
建议排查步骤：
复位或清除方式：
安全提示：
参考来源：
""".strip()


class GroundedAnswerGenerator:
    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str,
    ) -> None:
        if not api_key:
            raise ValueError("API Key不能为空。")

        self.model = model
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=60.0,
        )

    @classmethod
    def from_environment(cls) -> "GroundedAnswerGenerator":
        api_key = os.getenv("DEEPSEEK_API_KEY", "")
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
                "未找到DEEPSEEK_API_KEY，请检查环境变量。"
            )

        return cls(
            api_key=api_key,
            base_url=base_url,
            model=model,
        )

    @staticmethod
    def build_context(
        results: Sequence[SearchResult],
    ) -> str:
        context_parts = []

        for index, result in enumerate(results, start=1):
            context_parts.append(
                f"[证据{index} | PDF第{result.page_number}页]\n"
                f"{result.text}"
            )

        return "\n\n".join(context_parts)

    def generate(
        self,
        query: str,
        results: Sequence[SearchResult],
    ) -> str:
        query = query.strip()

        if not query:
            raise ValueError("用户问题不能为空。")

        if not results:
            return "当前手册证据不足，无法确定。"

        context = self.build_context(results)

        user_prompt = (
            f"用户问题：\n{query}\n\n"
            f"设备手册证据：\n{context}"
        )

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": user_prompt,
                },
            ],
                        temperature=0,
            max_tokens=1200,
            extra_body={
                "thinking": {
                    "type": "disabled",
                }
            },
        )

        answer = response.choices[0].message.content

        if not answer:
            raise RuntimeError("大模型返回了空答案。")

        return answer.strip()