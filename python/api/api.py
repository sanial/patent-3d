"""FastAPI server exposing the patent OCR pipeline.

Run from ``patent-ocr-layer/src``:

    uvicorn api.api:app --reload --port 8000
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import os
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from ocr.patent_ocr_pipeline import PatentOCRPipeline
from ocr.figure_extractor import extract_figures_fast
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
        # Run the blocking pipeline in a worker thread so other endpoints
        # (e.g. /api/extract-figures) can be served concurrently.
        result = await asyncio.to_thread(_pipeline.run, pdf_bytes)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Pipeline failed")
        raise HTTPException(status_code=500, detail=f"Pipeline error: {exc}") from exc

    # If the regex-based claim parser came up empty (common on scanned
    # patents whose section headers are mangled by OCR), let Gemma 4 pull
    # the claims out of the assembled OCR text.
    structure = result.structure
    regex_claims = structure.claims if structure else []
    if structure is not None and not regex_claims and result.full_text:
        logger.info(
            "No claims from regex parser — falling back to Gemma extraction "
            "(%d chars of OCR text)",
            len(result.full_text),
        )
        try:
            gemma_claims = await _gemma.extract_claims_from_text(result.full_text)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Gemma claim extraction raised: %s", exc)
            gemma_claims = []
        if gemma_claims:
            from ocr.structure import ParsedClaim  # local import avoids cycle

            structure.claims = [
                ParsedClaim(
                    number=c["number"],
                    type=c["type"],
                    body=c["body"],
                    refs=c["refs"],
                    dependsOn=c["dependsOn"],
                )
                for c in gemma_claims
            ]
            # Mirror onto the flat list used by patentData claim attribution.
            result.claims = [c["body"] for c in gemma_claims]
            logger.info("Gemma extracted %d claims", len(gemma_claims))

    return {
        "filename": file.filename,
        "result": result.to_dict(),
        "patentData": result.to_patent_data_format(),
    }


@app.post("/api/extract-figures")
async def extract_figures_endpoint(
    file: UploadFile = File(...),
    ocr_captions: bool = True,
    use_gemma: bool = False,
) -> dict[str, Any]:
    """Fast figure-only extraction. Returns figures in seconds, even on scanned PDFs."""
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only .pdf files are accepted")
    pdf_bytes = await file.read()
    if not pdf_bytes:
        raise HTTPException(status_code=400, detail="Empty file")
    if len(pdf_bytes) > 50 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="PDF exceeds 50 MB limit")
    logger.info("Extracting figures from %s (%d bytes, gemma=%s)",
                file.filename, len(pdf_bytes), use_gemma)
    try:
        # extract_figures_fast is sync and (with use_gemma=True) calls
        # asyncio.run() internally — offload to a worker thread so it
        # doesn't collide with the running event loop and doesn't block
        # /api/parse-patent from being served in parallel.
        figs = await asyncio.to_thread(
            extract_figures_fast,
            pdf_bytes,
            None,    # langs
            False,   # gpu
            4,       # max_workers
            ocr_captions,
            use_gemma,
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("Fast figure extraction failed")
        raise HTTPException(status_code=500, detail=f"Extractor error: {exc}") from exc
    return {
        "filename": file.filename,
        "extractedFigures": [f.to_dict() for f in figs],
    }


def _render_last_pages_b64(pdf_bytes: bytes, n: int, scale: float = 1.5) -> list[str]:
    """Render the last ``n`` pages of a PDF to base64-encoded PNGs."""
    import base64
    import fitz  # PyMuPDF

    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    try:
        total = doc.page_count
        start = max(0, total - n)
        matrix = fitz.Matrix(scale, scale)
        out: list[str] = []
        for idx in range(start, total):
            pix = doc[idx].get_pixmap(matrix=matrix, alpha=False)
            out.append(base64.b64encode(pix.tobytes("png")).decode("ascii"))
        return out
    finally:
        doc.close()


@app.post("/api/extract-claims")
async def extract_claims_endpoint(
    file: UploadFile = File(...),
    last_pages: int = 4,
) -> dict[str, Any]:
    """Fast claim-only extraction using Gemma 4 vision on the last N pages.

    Skips OCR entirely — sends the rasterised claim pages directly to Gemma.
    Designed to be called in parallel with /api/parse-patent so the Claims
    tab populates quickly even on scanned PDFs.
    """
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only .pdf files are accepted")
    pdf_bytes = await file.read()
    if not pdf_bytes:
        raise HTTPException(status_code=400, detail="Empty file")
    if len(pdf_bytes) > 50 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="PDF exceeds 50 MB limit")

    logger.info(
        "Extracting claims from %s (%d bytes, last %d pages)",
        file.filename, len(pdf_bytes), last_pages,
    )
    try:
        page_pngs = await asyncio.to_thread(
            _render_last_pages_b64, pdf_bytes, last_pages, 1.5
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("Page render failed")
        raise HTTPException(status_code=500, detail=f"Render error: {exc}") from exc

    try:
        claims = await _gemma.extract_claims_from_pages(page_pngs)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Gemma claim extraction failed")
        raise HTTPException(status_code=502, detail=f"Gemma error: {exc}") from exc

    logger.info("Gemma extracted %d claims from %d pages", len(claims), len(page_pngs))
    return {
        "filename": file.filename,
        "claims": claims,
    }


# ── Static frontend (Cloud Run / production) ────────────────────────────────
# When STATIC_DIR is set (e.g. inside the Docker image), serve the built
# Vite SPA from FastAPI. API routes above take precedence.
_static_dir = Path(os.environ.get("STATIC_DIR", "/app/static"))
if _static_dir.is_dir():
    _assets_dir = _static_dir / "assets"
    if _assets_dir.is_dir():
        app.mount("/assets", StaticFiles(directory=_assets_dir), name="assets")

    _index = _static_dir / "index.html"

    @app.get("/{full_path:path}", include_in_schema=False)
    async def spa_fallback(full_path: str):
        if full_path.startswith("api/"):
            raise HTTPException(status_code=404)
        candidate = (_static_dir / full_path).resolve()
        try:
            candidate.relative_to(_static_dir.resolve())
        except ValueError:
            raise HTTPException(status_code=404)
        if candidate.is_file():
            return FileResponse(candidate)
        if _index.is_file():
            return FileResponse(_index)
        raise HTTPException(status_code=404)
