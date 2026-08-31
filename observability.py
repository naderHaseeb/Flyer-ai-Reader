"""LangSmith trace filtering and metadata helpers."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from langsmith import get_current_run_tree, set_run_metadata


if not os.environ.get("LANGSMITH_PROJECT"):
    os.environ["LANGSMITH_PROJECT"] = "flyer-ai-capstone"


def path_identifier(value: Any) -> str | None:
    if value is None:
        return None
    return Path(str(value)).name


def update_trace_metadata(**metadata: Any) -> None:
    if get_current_run_tree() is not None:
        set_run_metadata(**metadata)


def openrouter_trace_inputs(inputs: dict[str, Any]) -> dict[str, Any]:
    return {
        "image_id": path_identifier(inputs.get("image_path")),
        "model_id": inputs.get("model_id"),
        "prompt_type": inputs.get("prompt_type"),
        "page_number": inputs.get("page_number"),
        "contact_sheet_number": inputs.get("contact_sheet_number"),
    }


def openrouter_trace_outputs(
    output: tuple[dict[str, Any], float] | None,
) -> dict[str, Any]:
    if output is None:
        return {"success": False}

    response, latency = output
    usage = response.get("usage", {}) or {}
    return {
        "response_status": 200,
        "latency_seconds": latency,
        "total_tokens": usage.get("total_tokens", 0) or 0,
        "cost": usage.get("cost", 0) or 0,
    }


def context_trace_inputs(inputs: dict[str, Any]) -> dict[str, Any]:
    return {
        "image_id": path_identifier(inputs.get("image_path")),
        "model_id": inputs.get("model_id"),
    }


def context_trace_outputs(output: dict[str, Any] | None) -> dict[str, Any]:
    if output is None:
        return {"success": False}

    return {
        "success": True,
        "fields_found": [key for key, value in output.items() if value is not None],
    }


def page_trace_inputs(inputs: dict[str, Any]) -> dict[str, Any]:
    return {
        "image_id": path_identifier(inputs.get("image_path")),
        "page_number": inputs.get("page_number"),
        "model_id": inputs.get("model_id"),
    }


def page_trace_outputs(
    output: tuple[
        dict[str, Any],
        float,
        dict[str, Any],
        float,
        list[str],
    ]
    | None,
) -> dict[str, Any]:
    if output is None:
        return {"success": False}

    prediction, qwen_latency, usage, yolo_latency, contact_sheet_paths = output
    return {
        "success": True,
        "yolo_detection_count": len(prediction.get("products", [])),
        "yolo_inference_latency_seconds": yolo_latency,
        "contact_sheet_count": len(contact_sheet_paths),
        "qwen_call_count": (
            len(contact_sheet_paths)
            + prediction.get("_section_date_call_count", 0)
        ),
        "qwen_product_count": prediction.get("_qwen_product_count", 0),
        "section_date_assignments": prediction.get(
            "_section_date_assignment_count",
            0,
        ),
        "qwen_latency_seconds": qwen_latency,
        "total_tokens": usage.get("total_tokens", 0) or 0,
        "cost": usage.get("cost", 0) or 0,
    }


def process_trace_inputs(inputs: dict[str, Any]) -> dict[str, Any]:
    return {
        "flyer_name": path_identifier(inputs.get("pdf_path")),
        "model_id": inputs.get("model_id"),
    }


def process_trace_outputs(output: dict[str, Any] | None) -> dict[str, Any]:
    if output is None:
        return {"success": False}

    usage_rows = output.get("_usage", [])
    return {
        "success": True,
        "model_id": output.get("_model"),
        "total_pages": len(output.get("_page_images", [])),
        "total_products": len(output.get("products", [])),
        "total_page_latency_seconds": sum(
            row.get("latency", 0) or 0 for row in usage_rows
        ),
        "total_page_cost": sum(row.get("cost", 0) or 0 for row in usage_rows),
        "total_yolo_detections": sum(
            row.get("yolo_detections", 0) or 0 for row in usage_rows
        ),
        "total_yolo_latency_seconds": sum(
            row.get("yolo_latency", 0) or 0 for row in usage_rows
        ),
        "total_contact_sheets": sum(
            row.get("contact_sheets", 0) or 0 for row in usage_rows
        ),
        "total_qwen_calls": sum(
            row.get("qwen_calls", 0) or 0 for row in usage_rows
        ),
    }
