"""Debug: dump bottom-strip OCR for figure pages."""
import sys
import fitz
import numpy as np
from PIL import Image
from ocr.patent_ocr_pipeline import _get_reader

pdf = sys.argv[1] if len(sys.argv) > 1 else r"C:\Users\Sunny\OneDrive\Documents\patent-3d\src\data\12596070.pdf"
doc = fitz.open(pdf)
reader = _get_reader(["en"], False)
for i in [0, 2, 5, 9, 12, 15]:
    page = doc[i]
    h = page.rect.height
    print(f"\n--- page {i}: rect={page.rect}")
    strip = fitz.Rect(0, h - 180, page.rect.width, h)
    pix = page.get_pixmap(matrix=fitz.Matrix(2.0, 2.0), clip=strip, alpha=False)
    print(f"  strip pix: {pix.width}x{pix.height}")
    Image.frombytes("RGB", (pix.width, pix.height), pix.samples).save(f"strip_{i}.png")
    arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
    out = reader.readtext(arr)
    for _bbox, text, conf in out:
        print(f"  [{conf:.2f}] {text!r}")
