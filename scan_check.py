import argparse
import os
import re
import sys
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import yaml


def _parse_roi(value: Optional[str]) -> Optional[Tuple[int, int, int, int]]:
    if not value:
        return None
    if value.strip().lower() in {"x,y,w,h", "x y w h"}:
        raise ValueError("ROI must be numeric like '120,450,600,140' (x,y,w,h)")
    parts = [p.strip() for p in value.split(",")]
    if len(parts) != 4:
        raise ValueError("ROI must be 'x,y,w,h'")
    try:
        x, y, w, h = (int(p) for p in parts)
    except ValueError as e:
        raise ValueError("ROI must contain only integers like '120,450,600,140'") from e
    return x, y, w, h


def load_input_as_bgr(path: str) -> np.ndarray:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        try:
            import fitz  # PyMuPDF
        except Exception as e:
            raise RuntimeError("PDF input requires PyMuPDF. Install with: pip install pymupdf") from e

        doc = fitz.open(path)
        try:
            if doc.page_count < 1:
                raise RuntimeError("PDF has no pages")
            page = doc.load_page(0)
            mat = fitz.Matrix(2, 2)  # ~144 DPI; adjust if needed
            pix = page.get_pixmap(matrix=mat, alpha=False)
            img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
            if pix.n == 3:
                return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            if pix.n == 4:
                return cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
            raise RuntimeError(f"Unsupported PDF raster format with {pix.n} channels")
        finally:
            doc.close()

    img_bgr = cv2.imread(path)
    if img_bgr is None:
        raise RuntimeError(f"Failed to read input as image: {path}")
    return img_bgr


def _crop(img: np.ndarray, roi: Optional[Tuple[int, int, int, int]]) -> np.ndarray:
    if roi is None:
        return img
    x, y, w, h = roi
    h_img, w_img = img.shape[:2]
    x = max(0, min(x, w_img - 1))
    y = max(0, min(y, h_img - 1))
    w = max(1, min(w, w_img - x))
    h = max(1, min(h, h_img - y))
    return img[y : y + h, x : x + w]


def preprocess_for_ocr(img_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 9, 75, 75)
    gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    thr = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 10
    )
    return thr


def preprocess_for_micr(img_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    thr = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 31, 12
    )
    thr = cv2.bitwise_not(thr)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1))
    thr = cv2.morphologyEx(thr, cv2.MORPH_CLOSE, kernel, iterations=1)
    thr = cv2.bitwise_not(thr)
    thr = cv2.resize(thr, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    return thr


def _is_valid_aba_routing(r: str) -> bool:
    if not re.fullmatch(r"\d{9}", r):
        return False
    digits = [int(c) for c in r]
    checksum = (
        3 * (digits[0] + digits[3] + digits[6])
        + 7 * (digits[1] + digits[4] + digits[7])
        + 1 * (digits[2] + digits[5] + digits[8])
    )
    return checksum % 10 == 0


def easyocr_micr(
    reader: Any,
    img_bgr: np.ndarray,
    roi: Optional[Tuple[int, int, int, int]],
    band_top: float,
) -> Dict[str, Any]:
    h, w = img_bgr.shape[:2]
    if roi is not None:
        micr_band = _crop(img_bgr, roi)
    else:
        band_top = max(0.0, min(float(band_top), 0.99))
        micr_band = img_bgr[int(h * band_top) : h, 0:w]

    proc = preprocess_for_micr(micr_band)

    try:
        results = reader.readtext(proc, allowlist="0123456789")
    except Exception as e:
        return {"ok": False, "error": "easyocr execution failed", "details": str(e)}

    def _x_center(bbox: Any) -> float:
        try:
            xs = [p[0] for p in bbox]
            return float(sum(xs)) / float(len(xs))
        except Exception:
            return 0.0

    sorted_results = sorted((results or []), key=lambda r: _x_center(r[0]) if len(r) else 0.0)

    texts: List[str] = []
    for r in sorted_results:
        if len(r) >= 2 and isinstance(r[1], str):
            texts.append(r[1])

    raw_text = " ".join(texts).strip() if texts else None
    digits_only = re.sub(r"\D+", "", raw_text or "") or None

    groups = re.findall(r"\d{4,20}", raw_text or "")
    routing = None
    for g in groups:
        if len(g) == 9 and _is_valid_aba_routing(g):
            routing = g
            break
    if routing is None:
        for g in groups:
            if len(g) == 9:
                routing = g
                break

    remaining = [g for g in groups if g != routing]
    account = None
    check_number = None
    if remaining:
        account = max(remaining, key=len)
        leftover = [g for g in remaining if g != account]
        if leftover:
            check_number = min(leftover, key=len)

    return {
        "ok": True,
        "raw": raw_text,
        "digits": digits_only,
        "routing_number": routing,
        "account_number": account,
        "check_number": check_number,
        "detections": len(results or []),
    }


def _easyocr_read_amount(
    reader: Any,
    img_bgr: np.ndarray,
    roi: Optional[Tuple[int, int, int, int]],
) -> Dict[str, Any]:
    crop = _crop(img_bgr, roi)

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    results = reader.readtext(gray)

    texts: List[str] = []
    for r in results or []:
        if len(r) >= 2 and isinstance(r[1], str):
            texts.append(r[1])

    full_text = "\n".join(texts).strip() if texts else None

    candidate = None
    if full_text:
        matches = re.findall(r"\b(\d{1,3}(?:,\d{3})*|\d+)(?:\.(\d{2}))?\b", full_text)
        if matches:
            v, cents = matches[-1]
            candidate = v.replace(",", "")
            if cents:
                candidate = f"{candidate}.{cents}"

    return {
        "ok": True,
        "text": full_text,
        "amount": candidate,
        "detections": len(results or []),
    }


def easyocr_amount(
    reader: Any,
    img_bgr: np.ndarray,
    roi: Optional[Tuple[int, int, int, int]],
) -> Dict[str, Any]:
    try:
        return _easyocr_read_amount(reader, img_bgr, roi)
    except Exception as e:
        return {"ok": False, "error": "easyocr execution failed", "details": str(e)}


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("image", help="Path to check input (png/jpg/pdf)")
    p.add_argument("--printed-roi", default=None, help="ROI for printed amount as x,y,w,h")
    p.add_argument("--handwritten-roi", default=None, help="ROI for handwritten amount as x,y,w,h")
    p.add_argument("--micr-roi", default=None, help="ROI for MICR line as x,y,w,h")
    p.add_argument(
        "--micr-band-top",
        default=0.82,
        type=float,
        help="If --micr-roi is not set, use bottom band starting at this fraction of image height (default: 0.82)",
    )
    p.add_argument(
        "--easyocr-model-dir",
        default=None,
        help="Directory to store/load EasyOCR models (default: ~/.EasyOCR)",
    )
    p.add_argument("--easyocr-gpu", action="store_true", help="Use GPU for EasyOCR (if available)")
    args = p.parse_args(argv)

    try:
        printed_roi = _parse_roi(args.printed_roi)
        handwritten_roi = _parse_roi(args.handwritten_roi)
        micr_roi = _parse_roi(args.micr_roi)
    except ValueError as e:
        p.error(str(e))

    try:
        img_bgr = load_input_as_bgr(args.image)
    except Exception as e:
        sys.stderr.write(f"Failed to load input: {e}\n")
        return 2

    try:
        import easyocr
    except Exception as e:
        sys.stderr.write(f"easyocr not installed or failed to import: {e}\n")
        return 2

    try:
        reader_kwargs: Dict[str, Any] = {"gpu": args.easyocr_gpu}
        if args.easyocr_model_dir:
            reader_kwargs["model_storage_directory"] = args.easyocr_model_dir
        reader = easyocr.Reader(["en"], **reader_kwargs)
    except Exception as e:
        sys.stderr.write(f"Failed to initialize EasyOCR Reader: {e}\n")
        return 2

    out: Dict[str, Any] = {
        "input": {"image_path": args.image},
        "micr": easyocr_micr(reader, img_bgr, micr_roi, args.micr_band_top),
        "printed_amount": easyocr_amount(reader, img_bgr, printed_roi),
        "handwritten_amount": easyocr_amount(reader, img_bgr, handwritten_roi),
    }

    sys.stdout.write(yaml.safe_dump(out, sort_keys=False, default_flow_style=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
