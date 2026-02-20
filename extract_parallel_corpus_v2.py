#!/usr/bin/env python3
"""
High-Accuracy Negarit Gazeta Parallel Corpus Extractor (v2)
============================================================

Pipeline (optimised for scanned bilingual legal PDFs):
  1. PDF  →  High-res page images (300 DPI)
  2. Image preprocessing with OpenCV
       • Grayscale  • Denoise  • Otsu threshold  • Deskew  • Sharpen
  3. Split each page into left (Amharic) / right (English) columns
  4. Tesseract OCR  (lang=amh / eng, --psm 6, --oem 3)
  5. Post-OCR noise cleaning  (regex, Unicode filtering)
  6. Sentence segmentation  (። for Amharic, nltk for English)
  7. Page-by-page sentence alignment
  8. Save as CSV dataset  +  debug text files

Requirements
------------
  pip install nltk pandas pytesseract pdf2image Pillow opencv-python-headless numpy
  sudo apt install tesseract-ocr tesseract-ocr-amh poppler-utils

Usage
-----
  python extract_parallel_corpus_v2.py                       # uses default PDF
  python extract_parallel_corpus_v2.py /path/to/other.pdf    # custom PDF
"""

from __future__ import annotations

import csv
import logging
import re
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytesseract
from pdf2image import convert_from_path
from PIL import Image

try:
    import nltk
    nltk.download("punkt", quiet=True)
    nltk.download("punkt_tab", quiet=True)
    from nltk.tokenize import sent_tokenize
except Exception:
    def sent_tokenize(text: str) -> list[str]:  # noqa: D401
        return re.split(r"(?<=[.!?;])\s+", text)

# ── Configuration ──────────────────────────────────────────────────────
PDF_PATH = Path("/home/girma/አዋጅ-ቁጥር-30-1988.pdf")
OUTPUT_DIR = Path("/home/girma/Desktop/negarit-gazeta-parallel-corpus")
OUTPUT_CSV = OUTPUT_DIR / "parallel_corpus_proclamation_30_v2.csv"
DPI = 300

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

# Amharic Unicode ranges
_AMH = r"\u1200-\u137F\u1380-\u139F\u2D80-\u2DDF\uAB00-\uAB2F\u1361-\u1368"
RE_AMH = re.compile(f"[{_AMH}]")

# =====================================================================
# 1. IMAGE PREPROCESSING  (OpenCV — the key to 50-80 % accuracy boost)
# =====================================================================

def preprocess_image(pil_img: Image.Image) -> np.ndarray:
    """
    Apply a full preprocessing pipeline to a PIL image and return a
    cleaned NumPy array ready for Tesseract.

    Steps:
      1. Convert to grayscale
      2. Resize (2×) for better small-character recognition
      3. Denoise (Non-Local Means)
      4. Adaptive threshold (Otsu)
      5. Deskew (correct minor rotation)
      6. Morphological cleaning (remove tiny specks)
    """
    # PIL → OpenCV (numpy)
    img = np.array(pil_img)

    # 1. Grayscale
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        gray = img

    # 2. Up-scale 2× for small text (improves Tesseract accuracy)
    h, w = gray.shape
    gray = cv2.resize(gray, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)

    # 3. Denoise
    denoised = cv2.fastNlMeansDenoising(gray, None, h=30, templateWindowSize=7,
                                         searchWindowSize=21)

    # 4. Otsu binarisation
    _, binary = cv2.threshold(denoised, 0, 255,
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 5. Deskew — find rotation angle and correct
    binary = _deskew(binary)

    # 6. Morphological opening — remove small noise dots
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    return cleaned


def _deskew(img: np.ndarray, max_angle: float = 5.0) -> np.ndarray:
    """Detect skew angle and rotate to correct it (up to ±max_angle°)."""
    coords = np.column_stack(np.where(img < 128))  # dark pixels
    if len(coords) < 50:
        return img
    try:
        angle = cv2.minAreaRect(coords)[-1]
    except cv2.error:
        return img

    # Normalise angle
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle

    if abs(angle) > max_angle:
        return img  # not a reasonable skew — skip

    h, w = img.shape[:2]
    centre = (w // 2, h // 2)
    mat = cv2.getRotationMatrix2D(centre, angle, 1.0)
    rotated = cv2.warpAffine(
        img, mat, (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return rotated

# =====================================================================
# 2. OCR
# =====================================================================

TESS_CONFIG = "--psm 6 --oem 3"


def ocr_column(page_img: Image.Image, side: str, lang: str) -> str:
    """
    Crop a column from the page, preprocess, and OCR.

    side: "left" (Amharic) or "right" (English)
    """
    w, h = page_img.size
    mid = w // 2
    margin = int(w * 0.02)  # small overlap avoids splitting chars at midline

    if side == "left":
        crop = page_img.crop((0, 0, mid + margin, h))
    else:
        crop = page_img.crop((mid - margin, 0, w, h))

    # OpenCV preprocessing
    processed = preprocess_image(crop)

    # Back to PIL for pytesseract
    pil_processed = Image.fromarray(processed)

    text = pytesseract.image_to_string(
        pil_processed, lang=lang, config=TESS_CONFIG,
    )
    return text


def ocr_all_pages(pdf_path: Path) -> list[tuple[str, str]]:
    """Convert PDF to images and OCR each page → [(amh, eng), ...]."""
    images = convert_from_path(str(pdf_path), dpi=DPI)
    log.info("PDF → %d page images at %d DPI", len(images), DPI)

    pages: list[tuple[str, str]] = []
    for i, img in enumerate(images, 1):
        amh = ocr_column(img, "left", "amh")
        eng = ocr_column(img, "right", "eng")
        log.info(
            "  Page %d  →  AM %d chars  |  EN %d chars",
            i, len(amh), len(eng),
        )
        pages.append((amh, eng))
    return pages

# =====================================================================
# 3. POST-OCR CLEANING
# =====================================================================

_AMH_NOISE = re.compile(f"[^{_AMH}0-9\\s/\\-()\".,;:!?'·]")
_ENG_HEADER = re.compile(
    r"^(Federal\s*Negarit|ederal\s*Negarit|Negarit\s*G"
    r"|Unit\s*Price|NEALE|N2|Lt\s*MA|196%|^\.\s*$)",
    re.I,
)


def clean_amharic(raw: str) -> list[str]:
    """Clean raw Amharic OCR text → list of usable lines."""
    out: list[str] = []
    for ln in raw.splitlines():
        ln = ln.strip()
        if not ln or not RE_AMH.search(ln):
            continue
        if re.search(r"ፌዴራል\s*ነጋሪት\s*ጋዜጣ", ln):
            continue
        if re.search(r"ያንዱ\s*ዋጋ", ln):
            continue
        ln = _AMH_NOISE.sub("", ln)
        ln = re.sub(r"\s{2,}", " ", ln).strip()
        if len(ln) > 2:
            out.append(ln)
    return out


def clean_english(raw: str) -> list[str]:
    """Clean raw English OCR text → list of usable lines."""
    out: list[str] = []
    for ln in raw.splitlines():
        ln = ln.strip()
        if not ln or _ENG_HEADER.match(ln):
            continue
        ln = re.sub(r"[^\x20-\x7E]", "", ln)          # keep printable ASCII
        ln = ln.replace(",,", ",").replace("..", ".")
        ln = re.sub(r"\s{2,}", " ", ln).strip()
        if len(ln) > 2:
            out.append(ln)
    return out

# =====================================================================
# 4. SENTENCE SEGMENTATION
# =====================================================================


def segment_amharic(lines: list[str]) -> list[str]:
    """Join lines → split on ። (Ethiopic full stop, U+1362)."""
    text = " ".join(lines)
    parts = re.split(r"(?<=።)\s*", text)
    return [s.strip() for s in parts if len(s.strip()) > 3]


def segment_english(lines: list[str]) -> list[str]:
    """Join lines (re-joining hyphenated words) → nltk sent_tokenize."""
    text = " ".join(lines)
    text = re.sub(r"- ", "", text)   # rejoin hyphenated breaks
    sents = sent_tokenize(text)
    return [s.strip() for s in sents if len(s.strip()) > 3]

# =====================================================================
# 5. ALIGNMENT
# =====================================================================


def align_sentences(
    amh: list[str], eng: list[str],
) -> list[tuple[str, str]]:
    """
    Align two sentence lists 1:1.
    Overflow sentences merge into the last aligned pair.
    """
    if not amh and not eng:
        return []
    if not amh:
        return [("[MISSING]", s) for s in eng]
    if not eng:
        return [(s, "[MISSING]") for s in amh]

    pairs: list[tuple[str, str]] = []
    n = min(len(amh), len(eng))
    for i in range(n):
        a, e = amh[i], eng[i]
        if i == n - 1:
            if len(amh) > n:
                a += " " + " ".join(amh[n:])
            if len(eng) > n:
                e += " " + " ".join(eng[n:])
        pairs.append((a.strip(), e.strip()))
    return pairs

# =====================================================================
# 6. VALIDATION & STATISTICS
# =====================================================================


def compute_stats(df: pd.DataFrame) -> dict:
    """Return quality statistics for the parallel corpus."""
    total = len(df)
    missing_am = int((df["amharic"] == "[MISSING]").sum())
    missing_en = int((df["english"] == "[MISSING]").sum())
    avg_am_len = df["amharic"].str.len().mean()
    avg_en_len = df["english"].str.len().mean()
    ratio = avg_am_len / avg_en_len if avg_en_len > 0 else 0

    return {
        "total_pairs": total,
        "missing_amharic": missing_am,
        "missing_english": missing_en,
        "avg_amharic_chars": round(avg_am_len, 1),
        "avg_english_chars": round(avg_en_len, 1),
        "am_en_length_ratio": round(ratio, 2),
    }

# =====================================================================
# 7. MAIN PIPELINE
# =====================================================================


def run(pdf_path: Path, out_csv: Path) -> pd.DataFrame:
    """Execute the full extraction pipeline."""
    log.info("=" * 60)
    log.info("High-Accuracy Negarit Gazeta Parallel Corpus Extractor v2")
    log.info("PDF : %s", pdf_path)
    log.info("Out : %s", out_csv)
    log.info("DPI : %d  |  Preprocessing: OpenCV (denoise+otsu+deskew)")
    log.info("=" * 60)

    # OCR
    raw_pages = ocr_all_pages(pdf_path)

    all_pairs: list[tuple[str, str]] = []
    amh_debug: list[str] = []
    eng_debug: list[str] = []

    for page_idx, (amh_raw, eng_raw) in enumerate(raw_pages, 1):
        amh_lines = clean_amharic(amh_raw)
        eng_lines = clean_english(eng_raw)

        amh_debug.append(f"\n{'='*40}  PAGE {page_idx}  {'='*40}")
        amh_debug.extend(amh_lines)
        eng_debug.append(f"\n{'='*40}  PAGE {page_idx}  {'='*40}")
        eng_debug.extend(eng_lines)

        amh_sents = segment_amharic(amh_lines)
        eng_sents = segment_english(eng_lines)

        log.info(
            "  Page %d  →  AM %d sentences  |  EN %d sentences",
            page_idx, len(amh_sents), len(eng_sents),
        )

        pairs = align_sentences(amh_sents, eng_sents)
        all_pairs.extend(pairs)

    # Save debug files
    outdir = out_csv.parent
    (outdir / "cleaned_amharic_v2.txt").write_text(
        "\n".join(amh_debug), encoding="utf-8",
    )
    (outdir / "cleaned_english_v2.txt").write_text(
        "\n".join(eng_debug), encoding="utf-8",
    )

    # Build DataFrame
    df = pd.DataFrame(all_pairs, columns=["amharic", "english"])
    df = df[~((df.amharic == "[MISSING]") & (df.english == "[MISSING]"))]
    df.drop_duplicates(inplace=True)
    df.reset_index(drop=True, inplace=True)

    df.to_csv(out_csv, index_label="id", encoding="utf-8", quoting=csv.QUOTE_ALL)

    # Stats
    stats = compute_stats(df)
    log.info("=" * 60)
    log.info("DONE — %d parallel pairs → %s", stats["total_pairs"], out_csv)
    log.info("  Missing AM: %d  |  Missing EN: %d", stats["missing_amharic"], stats["missing_english"])
    log.info("  Avg AM len: %.0f chars  |  Avg EN len: %.0f chars",
             stats["avg_amharic_chars"], stats["avg_english_chars"])
    log.info("  AM/EN length ratio: %.2f", stats["am_en_length_ratio"])
    log.info("=" * 60)

    return df

# =====================================================================
# ENTRY POINT
# =====================================================================

if __name__ == "__main__":
    pdf = Path(sys.argv[1]) if len(sys.argv) > 1 else PDF_PATH
    if not pdf.exists():
        log.error("PDF not found: %s", pdf)
        sys.exit(1)

    df = run(pdf, OUTPUT_CSV)

    # Pretty-print summary
    stats = compute_stats(df)
    print(f"\n{'═' * 62}")
    print("  HIGH-ACCURACY PARALLEL CORPUS — Proclamation 30/1996")
    print(f"{'═' * 62}")
    print(f"  Total aligned pairs   : {stats['total_pairs']}")
    print(f"  Missing Amharic       : {stats['missing_amharic']}")
    print(f"  Missing English       : {stats['missing_english']}")
    print(f"  Avg Amharic chars     : {stats['avg_amharic_chars']}")
    print(f"  Avg English chars     : {stats['avg_english_chars']}")
    print(f"  AM/EN length ratio    : {stats['am_en_length_ratio']}")
    print(f"  Output file           : {OUTPUT_CSV}")
    print(f"{'═' * 62}")
    print(f"\n{'─' * 62}")
    print("  SAMPLE PAIRS (first 10)")
    print(f"{'─' * 62}")
    for idx, row in df.head(10).iterrows():
        am = (row.amharic[:120] + "…") if len(row.amharic) > 120 else row.amharic
        en = (row.english[:120] + "…") if len(row.english) > 120 else row.english
        print(f"\n  [{idx}]")
        print(f"    AM: {am}")
        print(f"    EN: {en}")
    print()
