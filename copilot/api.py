"""HTTP-сервис: POST /triage по контракту, GET /health.

Запуск: uvicorn copilot.api:app --port 8000
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, Query
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse

from . import __version__
from . import dedup, llm
from .admin_api import router as admin_router
from .pipeline import triage
from .retrieval import get_history_retriever, get_rule_retriever
from .rules import get_index
from .schemas import TriageRequest, TriageResponse
from .taxonomy import get_taxonomy


@asynccontextmanager
async def lifespan(_: FastAPI):
    # прогрев: правила, таксономия и индексы грузятся один раз, а не на первом запросе
    get_index()
    get_taxonomy()
    get_rule_retriever()
    get_history_retriever()
    if dedup.enabled():
        dedup.get_store()
    yield


app = FastAPI(title="Grand Mobile Support Copilot", version=__version__, lifespan=lifespan)
app.include_router(admin_router)


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "version": __version__,
        "rule_points": len(get_index().points),
        "llm": llm.enabled(),
        "history_examples": get_history_retriever() is not None,
    }


@app.post("/triage", response_model=TriageResponse)
def triage_endpoint(req: TriageRequest, debug: bool = Query(False)):
    response, meta = triage(req, debug=debug)
    if debug:
        # вне контракта, только по ?debug=true; JSONResponse — чтобы response_model не отрезал поле
        return JSONResponse(jsonable_encoder({**response.model_dump(), "debug": meta}))
    return response
