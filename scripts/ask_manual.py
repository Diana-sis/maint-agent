import sys
from pathlib import Path

from dotenv import load_dotenv

from maintagent.agent import MaintenanceAgent
from maintagent.generation import GroundedAnswerGenerator
from maintagent.retrieval import HybridRetriever


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHUNKS_PATH = PROJECT_ROOT / "data" / "processed" / "chunks.jsonl"
EMBEDDINGS_PATH = PROJECT_ROOT / "data" / "processed" / "embeddings.npy"

load_dotenv(PROJECT_ROOT / ".env")

if len(sys.argv) < 2:
    raise SystemExit(
        '用法：python scripts/ask_manual.py "故障问题"'
    )

query = " ".join(sys.argv[1:]).strip()

retriever = HybridRetriever(
    chunks_path=CHUNKS_PATH,
    embeddings_path=EMBEDDINGS_PATH,
)

generator = GroundedAnswerGenerator.from_environment()

agent = MaintenanceAgent(
    retriever=retriever,
    generator=generator,
    top_k=3,
)

response = agent.run(query)

print("检索证据：")

for index, result in enumerate(
    response.evidence,
    start=1,
):
    print(
        f"{index}. PDF第{result.page_number}页 | "
        f"RRF={result.rrf_score:.6f}"
    )

print()
print("生成回答：")
print(response.answer)

pages = ", ".join(
    str(page)
    for page
    in response.citation_validation.cited_pages
)

print()
print(f"引用校验：通过，回答引用页码：{pages}")