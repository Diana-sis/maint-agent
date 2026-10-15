import os
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from maintagent.agent import MaintenanceAgent
from maintagent.generation import GroundedAnswerGenerator
from maintagent.retrieval import HybridRetriever

PROJECT_ROOT = Path(__file__).resolve().parents[2]
WEB_INDEX_PATH = PROJECT_ROOT / "web" / "index.html"

class AskRequest(BaseModel):
    question: str = Field(
        min_length=1,
        max_length=500,
    )

class UsageItem(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

class EvidenceItem(BaseModel):
    page_number: int
    rrf_score: float
    text: str


class AskResponse(BaseModel):
    question: str
    answer: str
    cited_pages: list[int]
    evidence: list[EvidenceItem]
    usage: UsageItem
    elapsed_ms: float


def build_agent() -> MaintenanceAgent:
    processed_dir = Path(
        os.getenv(
            "MAINT_AGENT_DATA_DIR",
            "data/processed",
        )
    )

    chunks_path = processed_dir / "chunks.jsonl"
    embeddings_path = processed_dir / "embeddings.npy"

    if not chunks_path.exists():
        raise FileNotFoundError(
            f"找不到文本块文件：{chunks_path.resolve()}"
        )

    if not embeddings_path.exists():
        raise FileNotFoundError(
            f"找不到向量文件：{embeddings_path.resolve()}"
        )

    retriever = HybridRetriever(
        chunks_path=chunks_path,
        embeddings_path=embeddings_path,
    )

    generator = GroundedAnswerGenerator.from_environment()

    return MaintenanceAgent(
        retriever=retriever,
        generator=generator,
        top_k=3,
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_dotenv()
    app.state.agent = build_agent()
    yield


app = FastAPI(
    title="MaintAgent API",
    description=(
        "基于设备手册检索、生成并校验引用的"
        "工业变频器维护辅助API。"
    ),
    version="0.1.0",
    lifespan=lifespan,
)

@app.get("/", include_in_schema=False)
def web_interface() -> FileResponse:
    if not WEB_INDEX_PATH.exists():
        raise HTTPException(
            status_code=404,
            detail="Web界面文件不存在。",
        )

    return FileResponse(WEB_INDEX_PATH)


@app.get("/health")
def health() -> dict[str, str]:
    return {
        "status": "ok",
    }


@app.post(
    "/ask",
    response_model=AskResponse,
)
async def ask(payload: AskRequest) -> AskResponse:
    try:
        response = await run_in_threadpool(
            app.state.agent.run,
            payload.question,
        )
    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        ) from error
    except RuntimeError as error:
        raise HTTPException(
            status_code=502,
            detail=str(error),
        ) from error

    evidence = [
        EvidenceItem(
            page_number=result.page_number,
            rrf_score=result.rrf_score,
            text=result.text,
        )
        for result in response.evidence
    ]

    return AskResponse(
        question=response.query,
        answer=response.answer,
        cited_pages=list(
            response.citation_validation.cited_pages
        ),
        evidence=evidence,
        usage=UsageItem(
            prompt_tokens=response.usage.prompt_tokens,
            completion_tokens=(
                response.usage.completion_tokens
            ),
            total_tokens=response.usage.total_tokens,
        ),
        elapsed_ms=round(response.elapsed_ms, 2),
    )