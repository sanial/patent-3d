"""Smoke test on the first N pages of a patent PDF."""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import fitz

from ocr.patent_ocr_pipeline import PatentOCRPipeline
from llm.gemma_client import GemmaClient


def main(pdf_path: str, max_pages: int = 6) -> None:
    p = Path(pdf_path)
    print(f"PDF: {p.name} ({p.stat().st_size} bytes), trimming to first {max_pages} pages")

    src = fitz.open(p)
    dst = fitz.open()
    dst.insert_pdf(src, from_page=0, to_page=min(max_pages - 1, src.page_count - 1))
    pdf_bytes = dst.tobytes()
    src.close()
    dst.close()

    pipeline = PatentOCRPipeline()
    result = pipeline.run(pdf_bytes)
    d = result.to_dict()

    print("\n=== Result ===")
    print(f"  title           : {d.get('title')!r}")
    print(f"  claims (raw)    : {len(d.get('claims', []))}")
    print(f"  refEntries      : {len(d.get('refEntries', {}))}")
    print(f"  extractedFigures: {len(d.get('extractedFigures', []))}")
    structure = d.get("structure") or {}
    print(f"  structure.claims        : {len(structure.get('claims', []))}")
    print(f"  structure.refDefinitions: {len(structure.get('refDefinitions', {}))}")
    print(f"  sections detected       : {list((structure.get('sections') or {}).keys())}")

    for f in d.get("extractedFigures", [])[:5]:
        print(
            f"    fig: {f['figureId']} p={f['page']} "
            f"caption={f['captionText'][:60]!r} refs={f['refNumbersOriginallyInside'][:8]}"
        )

    # Show a few ref entries.
    for ref, e in list(d.get("refEntries", {}).items())[:6]:
        print(f"    ref {ref}: label={e.get('label')!r}")

    # Show first claim if any.
    parsed_claims = structure.get("claims", [])
    if parsed_claims:
        c0 = parsed_claims[0]
        print(f"\n  claim 1 ({c0['type']}): {c0['body'][:200]}...")

    # If we have claims, also run the Gemma analyzer.
    if parsed_claims:
        print("\n=== Gemma analyze (text only) ===")
        client = GemmaClient()
        print(f"  provider={client.provider} model={client.model}")
        analysis = asyncio.run(
            client.analyze_claims(
                claims=[{"number": c["number"], "body": c["body"]} for c in parsed_claims[:3]],
                ref_definitions=structure.get("refDefinitions", {}),
                figure_pngs_b64=None,
            )
        )
        print(json.dumps(analysis.model_dump(), indent=2)[:1500])


if __name__ == "__main__":
    pdf = sys.argv[1] if len(sys.argv) > 1 else r"C:\Users\Sunny\OneDrive\Documents\patent-3d\src\data\12596070.pdf"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 6
    main(pdf, n)
