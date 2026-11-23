from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from typing import Literal

from . import similarity

app = FastAPI(title="Text Similarity Checker")
ROOT = Path(__file__).resolve().parent.parent

Method = Literal["tfidf", "semantic"]


class CompareRequest(BaseModel):
    text1: str = Field(min_length=1)
    text2: str = Field(min_length=1)
    method: Method = "tfidf"


class RankRequest(BaseModel):
    query: str = Field(min_length=1)
    documents: list[str] = Field(min_length=1, max_length=100)
    method: Method = "tfidf"


def _run(query, docs, method):
    try:
        return similarity.compute(query, docs, method)
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/compare")
def compare(req: CompareRequest):
    score = _run(req.text1, [req.text2], req.method)[0]
    return {"score": score, "label": similarity.label(score), "method": req.method}


@app.post("/api/rank")
def rank(req: RankRequest):
    scores = _run(req.query, req.documents, req.method)
    results = [
        {"index": i, "text": d, "score": s, "label": similarity.label(s)}
        for i, (d, s) in enumerate(zip(req.documents, scores))
    ]
    results.sort(key=lambda r: r["score"], reverse=True)
    return {"method": req.method, "results": results}


@app.get("/")
def index():
    return FileResponse(ROOT / "index.html")
