"""FastAPI server exposing the patent OCR pipeline.

Run from ``patent-ocr-layer/src``:

    uvicorn api.api:app --reload --port 8000
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from ocr.patent_ocr_pipeline import PatentOCRPipeline
from llm.gemma_client import GemmaClient

logger = logging.getLogger("patent-ocr")
logging.basicConfig(level=logging.INFO)

app = FastAPI(title="Patent OCR Layer", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten in production
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Reuse one pipeline (EasyOCR model load is expensive).
_pipeline = PatentOCRPipeline()
_gemma = GemmaClient()


class AnalyzeClaimsRequest(BaseModel):
    claims: list[dict[str, Any]]
    ref_definitions: dict[str, str] = {}
    figure_pngs_b64: list[str] | None = None


@app.get("/api/health")
def health() -> dict[str, str]:
    return {
        "status": "ok",
        "gemma_provider": _gemma.provider,
        "gemma_model": _gemma.model,
    }


@app.post("/api/analyze-claims")
async def analyze_claims(req: AnalyzeClaimsRequest) -> dict[str, Any]:
    if not req.claims:
        raise HTTPException(status_code=400, detail="claims is required")
    try:
        analysis = await _gemma.analyze_claims(
            claims=req.claims,
            ref_definitions=req.ref_definitions,
            figure_pngs_b64=req.figure_pngs_b64,
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("Gemma analysis failed")
        raise HTTPException(status_code=502, detail=f"Gemma error: {exc}") from exc
    return {"analysis": analysis.model_dump()}


@app.post("/api/parse-patent")
async def parse_patent(file: UploadFile = File(...)) -> dict[str, Any]:
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only .pdf files are accepted")

    pdf_bytes = await file.read()
    if not pdf_bytes:
        raise HTTPException(status_code=400, detail="Empty file")
    if len(pdf_bytes) > 50 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="PDF exceeds 50 MB limit")

    logger.info("Parsing %s (%d bytes)", file.filename, len(pdf_bytes))
    try:
        result = _pipeline.run(pdf_bytes)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Pipeline failed")
        raise HTTPException(status_code=500, detail=f"Pipeline error: {exc}") from exc

    return {
        "filename": file.filename,
        "result": result.to_dict(),
        "patentData": result.to_patent_data_format(),
    }
