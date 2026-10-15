import sys
from pathlib import Path

from dotenv import load_dotenv

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

results = retriever.search(
    query=query,
    top_k=3,
)

print("检索证据：")

for index, result in enumerate(results, start=1):
    print(
        f"{index}. PDF第{result.page_number}页 | "
        f"RRF={result.rrf_score:.6f}"
    )

print()
print("生成回答：")

generator = GroundedAnswerGenerator.from_environment()
answer = generator.generate(
    query=query,
    results=results,
)

print(answer)