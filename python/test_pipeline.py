"""Direct pipeline smoke test against a local PDF."""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

from ocr.patent_ocr_pipeline import PatentOCRPipeline
from llm.gemma_client import GemmaClient


def main(pdf_path: str) -> None:
    p = Path(pdf_path)
    print(f"PDF: {p} ({p.stat().st_size} bytes)")

    pipeline = PatentOCRPipeline()
    result = pipeline.run(p.read_bytes())
    d = result.to_dict()

    print("\n=== OCR result summary ===")
    print(f"  title           : {d.get('title')!r}")
    print(f"  claims          : {len(d.get('claims', []))}")
    print(f"  refEntries      : {len(d.get('refEntries', {}))}")
    print(f"  figures (hits)  : {len(d.get('figures', []))}")
    print(f"  extractedFigures: {len(d.get('extractedFigures', []))}")
    structure = d.get("structure") or {}
    parsed_claims = structure.get("claims", [])
    ref_defs = structure.get("refDefinitions", {})
    print(f"  structure.claims        : {len(parsed_claims)}")
    print(f"  structure.refDefinitions: {len(ref_defs)}")

    if d.get("extractedFigures"):
        for f in d["extractedFigures"][:3]:
            print(
                f"    - {f['figureId']} page={f['page']} "
                f"caption={f['captionText'][:40]!r} "
                f"refs={f['refNumbersOriginallyInside'][:6]} "
                f"png_bytes~{len(f.get('pngDataUrl','')) * 3 // 4}"
            )
    else:
        print("  (no extractedFigures — checking why)")
        import fitz  # type: ignore
        with fitz.open(p) as doc:
            for i, page in enumerate(doc):
                drawings = page.get_drawings() or []
                images = page.get_images(full=True)
                print(
                    f"    page {i}: drawings={len(drawings)} images={len(images)} "
                    f"size={int(page.rect.width)}x{int(page.rect.height)}"
                )
                if i >= 4:
                    break

    if not parsed_claims:
        print("\nNo claims parsed; skipping LLM analysis.")
        return

    print("\n=== Gemma analyze-claims ===")
    client = GemmaClient()
    print(f"  provider={client.provider} model={client.model}")
    analysis = asyncio.run(
        client.analyze_claims(
            claims=[{"number": c["number"], "body": c["body"]} for c in parsed_claims],
            ref_definitions=ref_defs,
            figure_pngs_b64=None,  # text-only for the smoke test
        )
    )
    out = analysis.model_dump()
    print(json.dumps(out, indent=2)[:2000])
    print("...")
    print(
        f"\n  analysis.claims        : {len(out['claims'])}"
        f"\n  component_summary keys : {len(out['component_summary'])}"
        f"\n  novelty_keywords       : {out['novelty_keywords']}"
    )


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else r"C:\Users\Sunny\OneDrive\Documents\patent-3d\src\data\12596070.pdf")
