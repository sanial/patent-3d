"""Time-only benchmark for the fast figure extractor."""
from __future__ import annotations

import base64
import sys
import time
from pathlib import Path

from ocr.figure_extractor import extract_figures_fast


def main(pdf_path: str, dump_dir: str | None = None, no_caption: bool = False, use_gemma: bool = False) -> None:
    p = Path(pdf_path)
    pdf_bytes = p.read_bytes()
    print(f"PDF: {p.name} ({len(pdf_bytes):,} bytes)  gemma={use_gemma}")

    t0 = time.perf_counter()
    figs = extract_figures_fast(pdf_bytes, ocr_captions=not no_caption, use_gemma=use_gemma)
    dt = time.perf_counter() - t0
    print(f"Extracted {len(figs)} figures in {dt:.2f}s "
          f"({dt / max(1, len(figs)):.2f}s/fig)")

    for f in figs:
        cap = (f.caption_text or "")[:60]
        print(f"  {f.figure_id:>10s}  page={f.page:<3d}  bbox=({f.bbox[0]:.0f},{f.bbox[1]:.0f},{f.bbox[2]:.0f},{f.bbox[3]:.0f})  cap={cap!r}")

    if dump_dir:
        out = Path(dump_dir)
        out.mkdir(parents=True, exist_ok=True)
        for f in figs:
            (out / f"{f.figure_id}_p{f.page}.png").write_bytes(base64.b64decode(f.png_base64))
        print(f"PNGs written to {out.resolve()}")


if __name__ == "__main__":
    pdf = sys.argv[1] if len(sys.argv) > 1 else r"C:\Users\Sunny\OneDrive\Documents\patent-3d\src\data\12596070.pdf"
    dump = sys.argv[2] if len(sys.argv) > 2 else None
    no_cap = "--no-caption" in sys.argv
    use_gemma = "--gemma" in sys.argv
    main(pdf, dump, no_cap, use_gemma)
