"""Extract supermarket flyer products with Qwen and local YOLO."""

from __future__ import annotations

import base64
from contextvars import ContextVar
from functools import lru_cache
import json
import math
import re
import time
import warnings
from pathlib import Path
from typing import Any

import fitz
import pandas as pd
import requests
from langsmith import traceable
from PIL import Image, ImageDraw, ImageFont
from pydantic import ValidationError
from ultralytics import YOLO

from observability import (
    context_trace_inputs,
    context_trace_outputs,
    openrouter_trace_inputs,
    openrouter_trace_outputs,
    page_trace_inputs,
    page_trace_outputs,
    path_identifier,
    process_trace_inputs,
    process_trace_outputs,
    update_trace_metadata,
)
from schemas import FlyerContext, QwenProduct, SectionDate


OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL_ID = "qwen/qwen3-vl-32b-instruct"
PROJECT_ROOT = Path(__file__).resolve().parent
YOLO_MODEL_PATH = PROJECT_ROOT / "models" / "bestyolo.pt"
YOLO_CONFIDENCE = 0.25
YOLO_IMAGE_SIZE = 1280
CONTACT_SHEET_BATCH_SIZE = 12
CONTACT_CROP_MAX_SIZE = (360, 320)
CONTACT_CROP_MAX_UPSCALE = 1.35
DUPLICATE_IOU_THRESHOLD = 0.65
DUPLICATE_CONTAINMENT_THRESHOLD = 0.85
CROP_PADDING_RATIO = 0.05
CROP_PADDING_MIN = 12
CROP_PADDING_MAX = 60
SECTION_DATE_HINT_PATTERN = re.compile(r"\b(day|daily|hour|hourly)\b", re.I)
DEFAULT_DATE_SOURCES = {None, "", "flyer_header", "flyer_default"}
_PIPELINE_USAGE: ContextVar[dict[str, float] | None] = ContextVar(
    "pipeline_usage",
    default=None,
)


def image_to_data_url(image_path: str | Path) -> str:
    image_path = str(image_path)
    extension = image_path.lower().split(".")[-1]

    if extension == "jpg":
        extension = "jpeg"

    with open(image_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")

    return f"data:image/{extension};base64,{encoded}"


def clean_json_text(text: str) -> str:
    text = text.strip()

    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]

    if text.endswith("```"):
        text = text[:-3]

    text = text.strip()

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1:
        text = text[start:end + 1]

    return text


def _response_json(response: dict[str, Any]) -> dict[str, Any]:
    content = response["choices"][0]["message"]["content"]
    return json.loads(clean_json_text(content))


def render_pdf(
    pdf_path: str | Path,
    page_dir: str | Path,
    scale: float = 1.3,
) -> list[str]:
    """Render PDF pages to JPG files."""
    page_dir = Path(page_dir)
    page_dir.mkdir(parents=True, exist_ok=True)

    doc = fitz.open(str(pdf_path))
    page_images: list[str] = []

    try:
        for i in range(len(doc)):
            page = doc.load_page(i)
            pix = page.get_pixmap(
                matrix=fitz.Matrix(scale, scale),
                alpha=False,
            )

            path = page_dir / f"page-{i + 1}.jpg"
            pix.save(str(path))
            page_images.append(str(path))
    finally:
        doc.close()

    return page_images


@traceable(
    name="OpenRouter Vision Call",
    run_type="llm",
    metadata={"provider": "openrouter"},
    process_inputs=openrouter_trace_inputs,
    process_outputs=openrouter_trace_outputs,
)
def call_openrouter(
    image_path: str | Path,
    prompt: str,
    api_key: str,
    model_id: str = MODEL_ID,
    timeout: int = 240,
    prompt_type: str = "unspecified",
    page_number: int | None = None,
    contact_sheet_number: int | None = None,
) -> tuple[dict[str, Any], float]:
    """Send one vision request to OpenRouter and return its latency."""
    image_data_url = image_to_data_url(image_path)
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": model_id,
        "temperature": 0,
        "provider": {
            "sort": "throughput",
            "allow_fallbacks": True,
        },
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt,
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_data_url,
                        },
                    },
                ],
            }
        ],
    }

    start = time.perf_counter()

    try:
        response = requests.post(
            OPENROUTER_URL,
            headers=headers,
            json=payload,
            timeout=timeout,
        )
    except requests.exceptions.ReadTimeout:
        time.sleep(2)
        try:
            response = requests.post(
                OPENROUTER_URL,
                headers=headers,
                json=payload,
                timeout=timeout,
            )
        except requests.exceptions.ReadTimeout as exc:
            latency = time.perf_counter() - start
            update_trace_metadata(
                model_id=model_id,
                prompt_type=prompt_type,
                page_number=page_number,
                contact_sheet_number=contact_sheet_number,
                latency_seconds=latency,
                failure_details="OpenRouter timed out twice",
            )
            raise RuntimeError(
                "The AI provider timed out twice. Please try processing the "
                "flyer again in a moment."
            ) from exc

    latency = time.perf_counter() - start

    if response.status_code != 200:
        update_trace_metadata(
            model_id=model_id,
            prompt_type=prompt_type,
            page_number=page_number,
            contact_sheet_number=contact_sheet_number,
            response_status=response.status_code,
            latency_seconds=latency,
            failure_details=f"OpenRouter HTTP {response.status_code}",
        )
        raise RuntimeError(
            f"OpenRouter HTTP {response.status_code}: {response.text}"
        )

    response_data = response.json()
    usage = response_data.get("usage", {}) or {}
    pipeline_usage = _PIPELINE_USAGE.get()
    if pipeline_usage is not None:
        pipeline_usage["total_tokens"] += usage.get("total_tokens", 0) or 0
        pipeline_usage["cost"] += usage.get("cost", 0) or 0

    update_trace_metadata(
        model_id=model_id,
        prompt_type=prompt_type,
        page_number=page_number,
        contact_sheet_number=contact_sheet_number,
        response_status=response.status_code,
        latency_seconds=latency,
        total_tokens=usage.get("total_tokens", 0) or 0,
        cost=usage.get("cost", 0) or 0,
    )

    return response_data, latency


FLYER_CONTEXT_PROMPT = """
You are reading PAGE 1 of a supermarket promotional flyer.

Extract flyer-wide campaign information only.

Return VALID JSON ONLY.
No markdown.
No comments.
No explanation.

Return exactly:
{
  "shop_name": null,
  "campaign_name": null,
  "flyer_start_date": null,
  "flyer_end_date": null,
  "region": null,
  "branch": null,
  "currency": null
}

Rules:
- Dates must use YYYY-MM-DD.
- Read the overall campaign validity dates, not a product-specific badge.
- If the year is visible, use it.
- For Bahraini Dinar return BHD.
- If a field cannot be determined, return null.
- Do not invent information.
""".strip()


@traceable(
    name="Extract Flyer Context",
    run_type="chain",
    process_inputs=context_trace_inputs,
    process_outputs=context_trace_outputs,
)
def extract_flyer_context(
    image_path: str | Path,
    api_key: str,
    model_id: str = MODEL_ID,
) -> dict[str, Any]:
    response, _ = call_openrouter(
        image_path=image_path,
        prompt=FLYER_CONTEXT_PROMPT,
        api_key=api_key,
        model_id=model_id,
        prompt_type="flyer_context",
        page_number=1,
        langsmith_extra={
            "metadata": {
                "model_id": model_id,
                "page_number": 1,
                "prompt_type": "flyer_context",
            }
        },
    )

    try:
        context = FlyerContext.model_validate(_response_json(response))
    except ValidationError as exc:
        raise ValueError("Qwen returned an invalid flyer context.") from exc
    return context.model_dump()


def find_year(start_date: Any, end_date: Any) -> int | None:
    for value in (start_date, end_date):
        if value:
            match = re.search(r"\b(20\d{2})\b", str(value))
            if match:
                return int(match.group(1))
    return None


def parse_date(value: Any, fallback_year: int | None = None) -> str | None:
    if value is None:
        return None

    try:
        if pd.isna(value):
            return None
    except Exception:
        pass

    text = str(value).strip()
    parsed = pd.to_datetime(text, errors="coerce")

    if pd.isna(parsed):
        return None

    if fallback_year is not None:
        if not re.search(r"\b20\d{2}\b", text):
            parsed = parsed.replace(year=fallback_year)

    return parsed.strftime("%Y-%m-%d")


def valid_bbox(box: Any) -> bool:
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        return False

    try:
        x1, y1, x2, y2 = [float(x) for x in box]
    except Exception:
        return False

    return 0 <= x1 < x2 <= 1000 and 0 <= y1 < y2 <= 1000


@lru_cache(maxsize=1)
def load_yolo_model() -> YOLO:
    """Load the trained local detector once without downloading other weights."""
    if not YOLO_MODEL_PATH.is_file():
        raise FileNotFoundError(f"YOLO model not found: {YOLO_MODEL_PATH}")
    return YOLO(str(YOLO_MODEL_PATH))


def _normalize_pixel_bbox(
    pixel_bbox: list[float],
    image_width: int,
    image_height: int,
) -> list[float] | None:
    x1, y1, x2, y2 = pixel_bbox
    normalized = [
        round(max(0.0, min(1000.0, x1 / image_width * 1000)), 2),
        round(max(0.0, min(1000.0, y1 / image_height * 1000)), 2),
        round(max(0.0, min(1000.0, x2 / image_width * 1000)), 2),
        round(max(0.0, min(1000.0, y2 / image_height * 1000)), 2),
    ]
    return normalized if valid_bbox(normalized) else None


def sort_detections(detections: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Use a simple top-to-bottom, then left-to-right reading order."""
    return sorted(
        detections,
        key=lambda detection: (
            detection["pixel_bbox"][1],
            detection["pixel_bbox"][0],
        ),
    )


def suppress_duplicate_detections(
    detections: list[dict[str, Any]],
    iou_threshold: float = DUPLICATE_IOU_THRESHOLD,
    containment_threshold: float = DUPLICATE_CONTAINMENT_THRESHOLD,
) -> list[dict[str, Any]]:
    """Keep the highest-confidence box among heavily overlapping detections."""
    kept: list[dict[str, Any]] = []
    for detection in sorted(
        detections,
        key=lambda item: item["confidence"],
        reverse=True,
    ):
        x1, y1, x2, y2 = detection["pixel_bbox"]
        area = (x2 - x1) * (y2 - y1)
        duplicate = False

        for existing in kept:
            ex1, ey1, ex2, ey2 = existing["pixel_bbox"]
            existing_area = (ex2 - ex1) * (ey2 - ey1)
            intersection_width = max(0.0, min(x2, ex2) - max(x1, ex1))
            intersection_height = max(0.0, min(y2, ey2) - max(y1, ey1))
            intersection = intersection_width * intersection_height
            if intersection <= 0:
                continue

            union = area + existing_area - intersection
            iou = intersection / union if union else 0.0
            containment = intersection / min(area, existing_area)
            if iou >= iou_threshold or containment >= containment_threshold:
                duplicate = True
                break

        if not duplicate:
            kept.append(detection)

    return kept


def detect_products_yolo(
    image_path: str | Path,
) -> tuple[dict[int, dict[str, Any]], float]:
    """Detect product regions and assign page-local IDs in reading order."""
    with Image.open(image_path) as image:
        image_width, image_height = image.size

    started = time.perf_counter()
    results = load_yolo_model().predict(
        source=str(image_path),
        conf=YOLO_CONFIDENCE,
        imgsz=YOLO_IMAGE_SIZE,
        verbose=False,
    )
    inference_latency = time.perf_counter() - started

    detections: list[dict[str, Any]] = []
    for result in results:
        boxes = result.boxes
        if boxes is None:
            continue

        for pixel_box, confidence, class_id in zip(
            boxes.xyxy.cpu().tolist(),
            boxes.conf.cpu().tolist(),
            boxes.cls.cpu().tolist(),
        ):
            if int(class_id) != 0:
                continue

            x1 = max(0.0, min(float(image_width), float(pixel_box[0])))
            y1 = max(0.0, min(float(image_height), float(pixel_box[1])))
            x2 = max(0.0, min(float(image_width), float(pixel_box[2])))
            y2 = max(0.0, min(float(image_height), float(pixel_box[3])))
            if x1 >= x2 or y1 >= y2:
                continue

            normalized_bbox = _normalize_pixel_bbox(
                [x1, y1, x2, y2],
                image_width,
                image_height,
            )
            if normalized_bbox is None:
                continue

            detections.append(
                {
                    "bbox": normalized_bbox,
                    "pixel_bbox": [x1, y1, x2, y2],
                    "confidence": float(confidence),
                }
            )

    detections = suppress_duplicate_detections(detections)
    ordered = sort_detections(detections)
    return {
        detection_id: detection
        for detection_id, detection in enumerate(ordered, start=1)
    }, inference_latency


def crop_product(
    page_image: Image.Image,
    pixel_bbox: list[float],
) -> Image.Image:
    """Create a padded Qwen crop while leaving the YOLO bbox unchanged."""
    x1, y1, x2, y2 = pixel_bbox
    padding = min(
        CROP_PADDING_MAX,
        max(
            CROP_PADDING_MIN,
            round(max(x2 - x1, y2 - y1) * CROP_PADDING_RATIO),
        ),
    )
    left = max(0, math.floor(x1 - padding))
    top = max(0, math.floor(y1 - padding))
    right = min(page_image.width, math.ceil(x2 + padding))
    bottom = min(page_image.height, math.ceil(y2 + padding))
    return page_image.crop((left, top, right, bottom)).convert("RGB")


def create_contact_sheets(
    image_path: str | Path,
    detections: dict[int, dict[str, Any]],
    output_dir: str | Path,
    page_number: int,
    batch_size: int = CONTACT_SHEET_BATCH_SIZE,
) -> list[dict[str, Any]]:
    """Place up to twelve labeled product crops on each contact sheet."""
    if not detections:
        return []

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        label_font = ImageFont.load_default(size=18)
    except TypeError:
        label_font = ImageFont.load_default()

    sheets: list[dict[str, Any]] = []
    detection_ids = list(detections)
    with Image.open(image_path) as opened_image:
        page_image = opened_image.convert("RGB")
        for sheet_index, offset in enumerate(
            range(0, len(detection_ids), batch_size),
            start=1,
        ):
            batch_ids = detection_ids[offset:offset + batch_size]
            crops: list[tuple[int, Image.Image]] = []
            for detection_id in batch_ids:
                crop = crop_product(
                    page_image,
                    detections[detection_id]["pixel_bbox"],
                )
                scale = min(
                    CONTACT_CROP_MAX_SIZE[0] / crop.width,
                    CONTACT_CROP_MAX_SIZE[1] / crop.height,
                    CONTACT_CROP_MAX_UPSCALE,
                )
                crop = crop.resize(
                    (
                        max(1, round(crop.width * scale)),
                        max(1, round(crop.height * scale)),
                    ),
                    Image.Resampling.LANCZOS,
                )
                crops.append((detection_id, crop))

            columns = 3 if len(crops) > 6 else 2
            rows = math.ceil(len(crops) / columns)
            cell_padding = 6
            cell_width = max(crop.width for _, crop in crops) + cell_padding * 2
            label_height = 30
            row_heights = []
            for row_index in range(rows):
                row_crops = crops[row_index * columns:(row_index + 1) * columns]
                row_heights.append(
                    label_height + max(crop.height for _, crop in row_crops) + 8
                )

            sheet = Image.new(
                "RGB",
                (cell_width * columns, sum(row_heights)),
                "white",
            )
            draw = ImageDraw.Draw(sheet)
            current_y = 0
            for crop_index, (detection_id, crop) in enumerate(crops):
                row_index = crop_index // columns
                column_index = crop_index % columns
                cell_x = column_index * cell_width
                cell_y = sum(row_heights[:row_index])
                draw.text(
                    (cell_x + cell_padding, cell_y + 5),
                    f"ID {detection_id}",
                    fill="black",
                    font=label_font,
                )
                crop_x = cell_x + (cell_width - crop.width) // 2
                sheet.paste(crop, (crop_x, cell_y + label_height))
                current_y = cell_y + row_heights[row_index]

            if current_y < sheet.height:
                sheet = sheet.crop((0, 0, sheet.width, current_y))

            sheet_path = output_dir / (
                f"page-{page_number}-contact-sheet-{sheet_index}.jpg"
            )
            sheet.save(sheet_path, "JPEG", quality=92)
            sheets.append(
                {
                    "path": str(sheet_path),
                    "ids": batch_ids,
                    "number": sheet_index,
                }
            )

    return sheets


def _explicit_currency_code(currency: Any) -> str | None:
    if currency is None:
        return None

    value = str(currency).strip()
    if not value:
        return None

    compact = re.sub(r"[\s.\-_/]", "", value.casefold())
    aliases = {
        "SAR": {"sar", "sr", "saudiriyal", "saudirial", "رس", "ريالسعودي", "\u20c1"},
        "BHD": {"bhd", "bd", "bahrainidinar", "دب", "ديناربحريني"},
        "AED": {"aed", "uaedirham", "emiratidirham", "دإ", "درهمإماراتي"},
        "QAR": {"qar", "qataririyal", "qataririal", "رق", "ريالقطري"},
        "KWD": {"kwd", "kd", "kuwaitidinar", "دك", "ديناركويتي"},
        "OMR": {"omr", "omanirial", "omaniriyal", "رع", "ريالعماني"},
    }
    for code, values in aliases.items():
        if compact in values:
            return code

    upper_value = value.upper()
    if re.fullmatch(r"[A-Z]{3}", upper_value):
        return upper_value
    return None


def normalize_currency(currency: Any, context: dict[str, Any]) -> str | None:
    explicit_currency = _explicit_currency_code(currency)
    if explicit_currency:
        return explicit_currency

    region = str(context.get("region") or "").casefold()
    region_codes = (
        (("bahrain",), "BHD"),
        (("saudi",), "SAR"),
        (("united arab emirates", "uae", "dubai", "abu dhabi"), "AED"),
        (("qatar",), "QAR"),
        (("kuwait",), "KWD"),
        (("oman",), "OMR"),
    )
    for names, code in region_codes:
        if any(name in region for name in names):
            return code

    return _explicit_currency_code(context.get("currency"))


def build_product_prompt(
    flyer_context: dict[str, Any],
) -> str:
    return f"""
Read every supermarket product crop in this contact sheet. Each is labeled ID N.

Context: flyer_start={flyer_context.get("flyer_start_date")}, flyer_end={flyer_context.get("flyer_end_date")}.

Return valid JSON only, with exactly one product for every visible ID. Use the
printed integer ID; never invent or omit IDs. Read visible values only and use
null instead of guessing.

Rules:
- product_name: preserve the visible brand/product name; remove only obvious
  duplicated pack wording or formatting artifacts such as a stray "|".
- quantity: return only the physical size or unit count, such as "1 kg",
  "500 g", "2 L", "6 x 200 ml", "12 pcs", or "1 pc".
- Never include a price or promotion phrase in quantity. If "3 for 1.25" clearly
  means three units, return quantity="3 pcs"; otherwise return null. Normalize
  artifacts such as "1 PC | pack" to "1 pc". Do not guess.
- price_after: current/promotional price as a number.
- price_before: old/original price only when explicitly visible; otherwise null.
- currency: inspect the visible symbol or currency text beside this product's
  price first. Normalize Saudi Riyal / ر.س / the Saudi Riyal symbol to SAR;
  Bahraini Dinar / BD / BHD / د.ب to BHD; UAE Dirham / AED / د.إ to AED;
  Qatari Riyal / QAR / ر.ق to QAR; Kuwaiti Dinar / KWD / د.ك to KWD; and
  Omani Rial / OMR / ر.ع. to OMR. Use only visibly recognized product-level
  currency. If no currency symbol or text is visibly recognizable, return null.
- Dates use YYYY-MM-DD. A product-specific date badge overrides flyer dates:
  set date_source="product_badge", resolve it using the flyer dates, and copy its
  visible wording to date_badge_text. Without one, use the flyer dates,
  date_source="flyer_default", and date_badge_text=null.
- Return only the fields in this schema:
{{
  "products": [
    {{
      "id": 1,
      "product_name": "",
      "quantity": null,
      "price_before": null,
      "price_after": null,
      "currency": null,
      "product_start_date": null,
      "product_end_date": null,
      "date_source": "flyer_default",
      "date_badge_text": null
    }}
  ]
}}
""".strip()


def should_check_section_dates(
    pdf_path: str | Path,
    flyer_context: dict[str, Any],
) -> bool:
    hint_text = " ".join(
        [
            Path(pdf_path).stem.replace("-", "_").replace("_", " "),
            str(flyer_context.get("campaign_name") or ""),
        ]
    )
    return bool(SECTION_DATE_HINT_PATTERN.search(hint_text))


def build_section_date_prompt(
    flyer_context: dict[str, Any],
    flyer_year: int | None,
) -> str:
    return f"""
Inspect this full flyer page only for multiple horizontal product sections with
explicit DAY/date labels. Do not extract products.

Context: flyer_start={flyer_context.get("flyer_start_date")}, flyer_end={flyer_context.get("flyer_end_date")}, year={flyer_year}.

Return valid JSON only:
{{"sections":[{{"label":"DAY 1 - 05 AUG 2026","date":"2026-08-05","top":250,"bottom":440}}]}}

Rules:
- Return sections only when at least two distinct day/date sections are clear.
- date uses YYYY-MM-DD.
- top and bottom are the full section's vertical limits, normalized 0–1000.
- Use null or an empty sections list rather than guessing.
""".strip()


PRODUCT_TEXT_FIELDS = (
    "product_name",
    "quantity",
    "price_before",
    "price_after",
    "currency",
    "product_start_date",
    "product_end_date",
    "date_source",
    "date_badge_text",
)


def _add_usage(total: dict[str, Any], response: dict[str, Any]) -> None:
    usage = response.get("usage", {}) or {}
    total["total_tokens"] += usage.get("total_tokens", 0) or 0
    total["cost"] += usage.get("cost", 0) or 0


def _empty_product() -> dict[str, Any]:
    return {
        "product_name": None,
        "quantity": None,
        "price_before": None,
        "price_after": None,
        "currency": None,
        "product_start_date": None,
        "product_end_date": None,
        "date_source": "flyer_default",
        "date_badge_text": None,
    }


def _collect_qwen_products(
    parsed_response: dict[str, Any],
    detections: dict[int, dict[str, Any]],
    qwen_products: dict[int, dict[str, Any]],
    page_number: int,
) -> None:
    if not isinstance(parsed_response, dict):
        warnings.warn(
            f"Page {page_number}: Qwen response must be a JSON object.",
            RuntimeWarning,
            stacklevel=2,
        )
        return
    products = parsed_response.get("products", [])
    if not isinstance(products, list):
        warnings.warn(
            f"Page {page_number}: Qwen products must be a list.",
            RuntimeWarning,
            stacklevel=2,
        )
        return

    for raw_product in products:
        try:
            product = QwenProduct.model_validate(raw_product)
        except ValidationError as exc:
            product_id = (
                raw_product.get("id")
                if isinstance(raw_product, dict)
                else None
            )
            issues = "; ".join(
                error["msg"] for error in exc.errors(include_url=False)
            )
            warnings.warn(
                f"Page {page_number}: ignoring invalid Qwen product ID "
                f"{product_id!r}: {issues}.",
                RuntimeWarning,
                stacklevel=2,
            )
            continue

        product_id = product.id
        if product_id not in detections:
            warnings.warn(
                f"Page {page_number}: ignoring unknown Qwen product ID "
                f"{product_id}.",
                RuntimeWarning,
                stacklevel=2,
            )
            continue
        if product_id in qwen_products:
            warnings.warn(
                f"Page {page_number}: ignoring duplicate Qwen product ID "
                f"{product_id}.",
                RuntimeWarning,
                stacklevel=2,
            )
            continue
        product_data = product.model_dump(by_alias=True)
        qwen_products[product_id] = {
            field: product_data.get(field) for field in PRODUCT_TEXT_FIELDS
        }


def merge_qwen_with_yolo(
    detections: dict[int, dict[str, Any]],
    qwen_products: dict[int, dict[str, Any]],
) -> list[dict[str, Any]]:
    products = []
    for product_id, detection in detections.items():
        product = _empty_product()
        if product_id in qwen_products:
            product.update(qwen_products[product_id])
        product["bbox"] = list(detection["bbox"])
        products.append(product)
    return products


def apply_section_dates(
    products: list[dict[str, Any]],
    sections: Any,
    flyer_year: int | None,
) -> int:
    valid_sections = []
    if not isinstance(sections, list):
        return 0

    for index, raw_section in enumerate(sections, start=1):
        try:
            section = SectionDate.model_validate(raw_section)
        except ValidationError as exc:
            issues = "; ".join(
                error["msg"] for error in exc.errors(include_url=False)
            )
            warnings.warn(
                f"Ignoring invalid section date {index}: {issues}.",
                RuntimeWarning,
                stacklevel=2,
            )
            continue
        valid_sections.append(
            {
                "date": parse_date(section.date, flyer_year),
                "label": section.label,
                "top": section.top,
                "bottom": section.bottom,
            }
        )

    if len(valid_sections) < 2 or len(
        {section["date"] for section in valid_sections}
    ) < 2:
        return 0

    assignments = 0
    for product in products:
        if product.get("date_source") not in DEFAULT_DATE_SOURCES:
            continue
        bbox = product.get("bbox")
        if not valid_bbox(bbox):
            continue
        center_y = (float(bbox[1]) + float(bbox[3])) / 2
        matches = [
            section
            for section in valid_sections
            if section["top"] <= center_y <= section["bottom"]
        ]
        if len(matches) != 1:
            continue
        section = matches[0]
        product["product_start_date"] = section["date"]
        product["product_end_date"] = section["date"]
        product["date_source"] = "product_badge"
        product["date_badge_text"] = section["label"]
        assignments += 1

    return assignments


@traceable(
    name="Page Extraction",
    run_type="chain",
    process_inputs=page_trace_inputs,
    process_outputs=page_trace_outputs,
)
def extract_page_products(
    image_path: str | Path,
    prompt: str,
    api_key: str,
    model_id: str,
    page_number: int,
    contact_sheet_dir: str | Path,
    section_date_prompt: str | None = None,
    flyer_year: int | None = None,
) -> tuple[dict[str, Any], float, dict[str, Any], float, list[str]]:
    detections, yolo_latency = detect_products_yolo(image_path)
    contact_sheets = create_contact_sheets(
        image_path=image_path,
        detections=detections,
        output_dir=contact_sheet_dir,
        page_number=page_number,
    )

    qwen_products: dict[int, dict[str, Any]] = {}
    product_qwen_latency = 0.0
    section_date_qwen_latency = 0.0
    combined_usage: dict[str, Any] = {"total_tokens": 0, "cost": 0}

    for contact_sheet in contact_sheets:
        response, latency = call_openrouter(
            image_path=contact_sheet["path"],
            prompt=prompt,
            api_key=api_key,
            model_id=model_id,
            prompt_type="contact_sheet_products",
            page_number=page_number,
            contact_sheet_number=contact_sheet["number"],
            langsmith_extra={
                "metadata": {
                    "model_id": model_id,
                    "page_number": page_number,
                    "prompt_type": "contact_sheet_products",
                    "contact_sheet_number": contact_sheet["number"],
                    "expected_product_ids": contact_sheet["ids"],
                }
            },
        )
        product_qwen_latency += latency
        _add_usage(combined_usage, response)
        _collect_qwen_products(
            _response_json(response),
            detections,
            qwen_products,
            page_number,
        )

    merged_products = merge_qwen_with_yolo(detections, qwen_products)

    section_date_call_count = 0
    section_date_assignment_count = 0
    if section_date_prompt and any(
        product.get("date_source") in DEFAULT_DATE_SOURCES
        for product in merged_products
    ):
        section_date_call_count = 1
        response, latency = call_openrouter(
            image_path=image_path,
            prompt=section_date_prompt,
            api_key=api_key,
            model_id=model_id,
            prompt_type="page_section_dates",
            page_number=page_number,
            langsmith_extra={
                "metadata": {
                    "model_id": model_id,
                    "page_number": page_number,
                    "prompt_type": "page_section_dates",
                }
            },
        )
        section_date_qwen_latency += latency
        _add_usage(combined_usage, response)
        section_response = _response_json(response)
        sections = (
            section_response.get("sections")
            if isinstance(section_response, dict)
            else None
        )
        section_date_assignment_count = apply_section_dates(
            merged_products,
            sections,
            flyer_year,
        )

    qwen_latency = product_qwen_latency + section_date_qwen_latency
    prediction = {
        "products": merged_products,
        "_page": page_number,
        "_qwen_product_count": len(qwen_products),
        "_section_date_call_count": section_date_call_count,
        "_section_date_assignment_count": section_date_assignment_count,
        "_product_qwen_seconds": product_qwen_latency,
        "_section_date_qwen_seconds": section_date_qwen_latency,
    }
    contact_sheet_paths = [sheet["path"] for sheet in contact_sheets]

    update_trace_metadata(
        model_id=model_id,
        page_number=page_number,
        yolo_detection_count=len(detections),
        yolo_inference_latency_seconds=yolo_latency,
        yolo_cost=0,
        contact_sheet_count=len(contact_sheets),
        qwen_call_count=len(contact_sheets) + section_date_call_count,
        qwen_product_count=len(qwen_products),
        section_date_assignments=section_date_assignment_count,
        qwen_latency_seconds=qwen_latency,
        product_qwen_latency_seconds=product_qwen_latency,
        section_date_qwen_latency_seconds=section_date_qwen_latency,
        total_tokens=combined_usage["total_tokens"],
        cost=combined_usage["cost"],
    )

    return (
        prediction,
        qwen_latency,
        combined_usage,
        yolo_latency,
        contact_sheet_paths,
    )


def finalize_products(
    page_predictions: list[dict[str, Any]],
    flyer_context: dict[str, Any],
    flyer_year: int | None,
) -> tuple[list[dict[str, Any]], str | None, str | None]:
    flyer_start = parse_date(
        flyer_context.get("flyer_start_date"),
        flyer_year,
    )
    flyer_end = parse_date(
        flyer_context.get("flyer_end_date"),
        flyer_year,
    )
    rows = []

    for page in page_predictions:
        page_number = int(page["_page"])
        for product in page.get("products", []):
            start = parse_date(product.get("product_start_date"), flyer_year)
            end = parse_date(product.get("product_end_date"), flyer_year)
            date_source = product.get("date_source")

            if date_source in DEFAULT_DATE_SOURCES:
                start = start or flyer_start
                end = end or flyer_end
                date_source = "flyer_default"

            bbox = product.get("bbox")
            bbox = list(bbox) if valid_bbox(bbox) else None

            row = {
                "shop_name": flyer_context.get("shop_name"),
                "campaign_name": flyer_context.get("campaign_name"),
                "region": flyer_context.get("region"),
                "branch": flyer_context.get("branch"),
                "flyer_start_date": flyer_start,
                "flyer_end_date": flyer_end,
                "page": page_number,
                **product,
            }
            row["product_start_date"] = start
            row["product_end_date"] = end
            row["date_source"] = date_source
            row["currency"] = normalize_currency(
                row.get("currency"),
                flyer_context,
            )
            row["bbox"] = bbox
            rows.append(row)

    return rows, flyer_start, flyer_end


def _page_usage_row(
    page_number: int,
    prediction: dict[str, Any],
    usage: dict[str, Any],
    qwen_latency: float,
    yolo_latency: float,
    contact_sheet_paths: list[str],
) -> dict[str, Any]:
    section_calls = prediction.get("_section_date_call_count", 0)
    return {
        "page": page_number,
        "products": len(prediction.get("products", [])),
        "qwen_products": prediction.get("_qwen_product_count", 0),
        "tokens": usage.get("total_tokens", 0) or 0,
        "cost": usage.get("cost", 0) or 0,
        "latency": qwen_latency,
        "yolo_detections": len(prediction.get("products", [])),
        "yolo_latency": yolo_latency,
        "contact_sheets": len(contact_sheet_paths),
        "section_date_calls": section_calls,
        "qwen_calls": len(contact_sheet_paths) + section_calls,
    }


def _process_flyer_impl(
    pdf_path: str | Path,
    work_dir: str | Path,
    api_key: str,
    model_id: str = MODEL_ID,
) -> dict[str, Any]:
    """Run the complete PDF-to-products pipeline."""
    total_started = time.perf_counter()
    if not api_key:
        raise ValueError("OpenRouter API key is missing.")

    page_dir = Path(work_dir) / "pages"
    render_started = time.perf_counter()
    page_images = render_pdf(pdf_path, page_dir)
    pdf_render_seconds = time.perf_counter() - render_started

    if not page_images:
        raise ValueError("The PDF has no pages.")

    context_started = time.perf_counter()
    flyer_context = extract_flyer_context(
        page_images[0],
        api_key=api_key,
        model_id=model_id,
    )
    flyer_context_seconds = time.perf_counter() - context_started

    flyer_year = find_year(
        flyer_context.get("flyer_start_date"),
        flyer_context.get("flyer_end_date"),
    )

    product_prompt = build_product_prompt(flyer_context)
    section_date_prompt = None
    if should_check_section_dates(pdf_path, flyer_context):
        section_date_prompt = build_section_date_prompt(
            flyer_context,
            flyer_year,
        )

    page_predictions = []
    usage_rows = []
    contact_sheet_paths = []
    contact_sheet_dir = Path(work_dir) / "contact_sheets"
    yolo_seconds = 0.0
    product_qwen_seconds = 0.0
    section_date_qwen_seconds = 0.0

    for page_number, image_path in enumerate(page_images, start=1):
        (
            prediction,
            qwen_latency,
            usage,
            yolo_latency,
            page_contact_sheets,
        ) = extract_page_products(
            image_path=image_path,
            prompt=product_prompt,
            api_key=api_key,
            model_id=model_id,
            page_number=page_number,
            contact_sheet_dir=contact_sheet_dir,
            section_date_prompt=section_date_prompt,
            flyer_year=flyer_year,
            langsmith_extra={
                "name": f"Page {page_number} Extraction",
                "metadata": {
                    "model_id": model_id,
                    "page_number": page_number,
                },
            },
        )
        page_predictions.append(prediction)
        contact_sheet_paths.extend(page_contact_sheets)
        yolo_seconds += yolo_latency
        product_qwen_seconds += prediction.get("_product_qwen_seconds", 0.0)
        section_date_qwen_seconds += prediction.get(
            "_section_date_qwen_seconds",
            0.0,
        )
        usage_rows.append(
            _page_usage_row(
                page_number,
                prediction,
                usage,
                qwen_latency,
                yolo_latency,
                page_contact_sheets,
            )
        )

    rows, global_start, global_end = finalize_products(
        page_predictions,
        flyer_context,
        flyer_year,
    )

    result = {
        "shop_name": flyer_context.get("shop_name"),
        "campaign_name": flyer_context.get("campaign_name"),
        "flyer_start_date": global_start,
        "flyer_end_date": global_end,
        "region": flyer_context.get("region"),
        "branch": flyer_context.get("branch"),
        "currency": normalize_currency(
            flyer_context.get("currency"),
            flyer_context,
        ),
        "products": rows,
        "_page_images": page_images,
        "_contact_sheets": contact_sheet_paths,
        "_usage": usage_rows,
        "_model": model_id,
    }
    result["_timing"] = {
        "pdf_render_seconds": pdf_render_seconds,
        "flyer_context_seconds": flyer_context_seconds,
        "yolo_seconds": yolo_seconds,
        "product_qwen_seconds": product_qwen_seconds,
        "section_date_qwen_seconds": section_date_qwen_seconds,
        "total_processing_seconds": time.perf_counter() - total_started,
    }
    return result


@traceable(
    name="Process Flyer",
    run_type="chain",
    process_inputs=process_trace_inputs,
    process_outputs=process_trace_outputs,
)
def process_flyer(
    pdf_path: str | Path,
    work_dir: str | Path,
    api_key: str,
    model_id: str = MODEL_ID,
) -> dict[str, Any]:
    pipeline_started = time.perf_counter()
    pipeline_usage = {"total_tokens": 0.0, "cost": 0.0}
    usage_token = _PIPELINE_USAGE.set(pipeline_usage)

    try:
        try:
            result = _process_flyer_impl(
                pdf_path=pdf_path,
                work_dir=work_dir,
                api_key=api_key,
                model_id=model_id,
            )
        except Exception as exc:
            update_trace_metadata(
                model_id=model_id,
                flyer_name=path_identifier(pdf_path),
                success=False,
                failure_type=type(exc).__name__,
                total_latency_seconds=time.perf_counter() - pipeline_started,
                total_tokens=pipeline_usage["total_tokens"],
                total_cost=pipeline_usage["cost"],
            )
            raise

        update_trace_metadata(
            model_id=model_id,
            flyer_name=path_identifier(pdf_path),
            total_pages=len(result.get("_page_images", [])),
            total_products=len(result.get("products", [])),
            total_latency_seconds=time.perf_counter() - pipeline_started,
            total_tokens=pipeline_usage["total_tokens"],
            total_cost=pipeline_usage["cost"],
            success=True,
        )
        return result
    finally:
        _PIPELINE_USAGE.reset(usage_token)
