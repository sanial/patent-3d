"""Direct test of GemmaClient.extract_claims_from_pages.

Renders the last N pages of the test PDF to PNG and sends them straight to
Gemma 4. No OCR, no FastAPI, no uvicorn. Dumps Gemma's raw response when
JSON parsing fails so we can debug prompt issues.
"""
from __future__ import annotations

import asyncio
import base64
import json
import logging
import sys
import time
from pathlib import Path

import fitz  # PyMuPDF

from llm.gemma_client import GemmaClient

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")


def render_last_pages(pdf_path: Path, n: int, scale: float = 1.5) -> list[str]:
    doc = fitz.open(str(pdf_path))
    try:
        total = doc.page_count
        start = max(0, total - n)
        matrix = fitz.Matrix(scale, scale)
        out: list[str] = []
        for idx in range(start, total):
            pix = doc[idx].get_pixmap(matrix=matrix, alpha=False)
            png = pix.tobytes("png")
            out.append(base64.b64encode(png).decode("ascii"))
            print(f"[render] page {idx}: {len(png)} bytes PNG", flush=True)
        return out
    finally:
        doc.close()


async def main(pdf_path: str, last_pages: int) -> None:
    p = Path(pdf_path)
    print(f"[main] rendering last {last_pages} pages of {p.name}", flush=True)
    pngs = render_last_pages(p, last_pages)
    print(f"[main] {len(pngs)} page PNGs (~{sum(len(s) for s in pngs) // 1024} KiB base64)", flush=True)

    client = GemmaClient()
    print(f"[main] provider={client.provider} model={client.model} vision_model={client.vision_model}", flush=True)
    t0 = time.time()
    claims = await client.extract_claims_from_pages(pngs)
    print(f"[main] returned {len(claims)} claims in {time.time() - t0:.1f}s", flush=True)
    for c in claims:
        head = c["body"][:160].replace("\n", " ")
        print(f"  {c['number']:>2}. [{c['type']}] dependsOn={c['dependsOn']} refs={c['refs']}")
        print(f"      {head}")
    if claims:
        print()
        print(json.dumps(claims, indent=2)[:1500])


if __name__ == "__main__":
    pdf = sys.argv[1] if len(sys.argv) > 1 else r"C:\Users\Sunny\OneDrive\Documents\patent-3d\src\data\12596070.pdf"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    asyncio.run(main(pdf, n))
