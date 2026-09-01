from __future__ import annotations

import csv
import html
import json
import os
import sqlite3
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw

st.set_page_config(page_title="Flyer AI Reader", layout="wide")

def configure_langsmith_environment():
    """Load optional LangSmith settings from Streamlit secrets."""
    secret_names = (
        "LANGSMITH_API_KEY",
        "LANGSMITH_PROJECT",
        "LANGSMITH_TRACING",
    )

    for name in secret_names:
        if os.environ.get(name):
            continue

        try:
            value = st.secrets.get(name)
        except Exception:
            value = None

        if value not in (None, ""):
            os.environ[name] = str(value).lower() if isinstance(
                value, bool
            ) else str(value)

    if not os.environ.get("LANGSMITH_PROJECT"):
        os.environ["LANGSMITH_PROJECT"] = "flyer-ai-capstone"


configure_langsmith_environment()

from flyer_pipeline import MODEL_ID, process_flyer, valid_bbox


st.markdown(
    """
    <style>
    :root {
        --background: #100719;
        --panel: #1b0d2a;
        --border: rgba(196, 181, 253, 0.18);
        --accent: #7c3aed;
        --text: #f8f7ff;
        --muted: #b9aed0;
    }
    .stApp {
        background: var(--background);
        color: var(--text);
    }
    [data-testid="stHeader"] { background: transparent; }
    h1, h2, h3, h4, label, [data-testid="stMarkdownContainer"] p {
        color: var(--text);
    }
    .stCaption, [data-testid="stCaptionContainer"] {
        color: var(--muted) !important;
    }
    .app-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 1rem;
        margin: 0 0 1rem;
        padding: 0.8rem 0;
        border-bottom: 1px solid var(--border);
    }
    .app-title {
        margin: 0;
        color: white;
        font-size: 1.55rem;
        font-weight: 750;
    }
    .app-subtitle { margin: 0.15rem 0 0; color: var(--muted); font-size: 0.86rem; }
    .status-good, .status-review {
        display: inline-flex;
        align-items: center;
        border-radius: 999px;
        padding: 0.28rem 0.55rem;
        font-size: 0.72rem;
        font-weight: 700;
        white-space: nowrap;
    }
    .status-good { color: #d1fae5; background: rgba(16, 185, 129, 0.18); }
    .status-review { color: #fef3c7; background: rgba(245, 158, 11, 0.18); }
    .page-progress {
        display: flex;
        flex-wrap: wrap;
        gap: 0.32rem;
        margin: 0.15rem 0 0.45rem;
    }
    .page-chip {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        min-width: 1.75rem;
        height: 1.75rem;
        padding: 0 0.42rem;
        border-radius: 999px;
        color: var(--muted);
        border: 1px solid var(--border);
        font-size: 0.72rem;
        font-weight: 700;
    }
    .page-chip.reviewed { color: #d1fae5; border-color: rgba(52, 211, 153, 0.35); }
    .page-chip.current { color: white; background: var(--accent); border-color: #a78bfa; }
    .issue-list { margin: 0.3rem 0 0.65rem; color: #fef3c7; font-size: 0.8rem; }
    .overview-section { margin: 1.25rem 0 0.35rem; color: #a78bfa; font-size: 0.72rem; font-weight: 750; letter-spacing: .1em; }
    .overview-row { display: flex; justify-content: space-between; gap: 2rem; padding: .52rem 0; border-bottom: 1px solid var(--border); }
    .overview-label { color: var(--muted); }
    .overview-value { color: white; text-align: right; font-weight: 650; }
    .stButton > button, .stDownloadButton > button {
        border-radius: 9px;
        border: 1px solid var(--border);
        font-weight: 700;
    }
    .stButton > button[kind="primary"], .stDownloadButton > button {
        color: white;
        background: var(--accent);
    }
    [data-testid="stFileUploaderDropzone"],
    [data-baseweb="select"] > div,
    .stNumberInput input,
    .stTextInput input {
        color: var(--text) !important;
        background: var(--panel) !important;
        border-color: var(--border) !important;
        border-radius: 9px !important;
    }
    [data-testid="stExpander"] {
        border-color: var(--border) !important;
        border-radius: 9px !important;
        background: rgba(27, 13, 42, 0.55);
    }
    hr { border-color: var(--border) !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="app-header">
      <div>
        <div class="app-title">Flyer AI Reader</div>
        <p class="app-subtitle">YOLO + Qwen assisted review</p>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

DB_PATH = Path("flyer_data.db")
CORRECTIONS_PATH = Path("corrections.csv")


def init_database():
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS reviews (
            review_id TEXT PRIMARY KEY,
            reviewed_at TEXT NOT NULL,
            status TEXT NOT NULL,
            shop_name TEXT,
            campaign_name TEXT,
            product_count INTEGER
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS approved_products (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            review_id TEXT NOT NULL,
            reviewed_at TEXT NOT NULL,
            shop_name TEXT,
            campaign_name TEXT,
            region TEXT,
            branch TEXT,
            flyer_start_date TEXT,
            flyer_end_date TEXT,
            page INTEGER,
            product_name TEXT,
            quantity TEXT,
            price_before REAL,
            price_after REAL,
            currency TEXT,
            product_start_date TEXT,
            product_end_date TEXT,
            date_source TEXT,
            date_badge_text TEXT,
            bbox TEXT
        )
        """
    )
    conn.commit()
    conn.close()


def draw_bbox(image_path, bbox):
    image = Image.open(image_path).convert("RGB")

    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return image

    try:
        x1, y1, x2, y2 = [float(v) for v in bbox]
    except Exception:
        return image

    width, height = image.size
    pixel_box = (
        x1 / 1000 * width,
        y1 / 1000 * height,
        x2 / 1000 * width,
        y2 / 1000 * height,
    )

    draw = ImageDraw.Draw(image)
    line_width = max(3, int(min(width, height) * 0.006))
    draw.rectangle(pixel_box, outline="red", width=line_width)
    return image


def crop_bbox_preview(image_path, bbox):
    if not valid_bbox(bbox):
        return None

    with Image.open(image_path) as opened_image:
        image = opened_image.convert("RGB")

    x1, y1, x2, y2 = [float(value) for value in bbox]
    width, height = image.size
    pixel_box = (
        x1 / 1000 * width,
        y1 / 1000 * height,
        x2 / 1000 * width,
        y2 / 1000 * height,
    )
    padding = max(
        8,
        round(max(pixel_box[2] - pixel_box[0], pixel_box[3] - pixel_box[1]) * 0.06),
    )
    crop_box = (
        max(0, int(pixel_box[0] - padding)),
        max(0, int(pixel_box[1] - padding)),
        min(width, int(pixel_box[2] + padding)),
        min(height, int(pixel_box[3] + padding)),
    )
    return image.crop(crop_box)


def editable_columns():
    return [
        "product_name",
        "quantity",
        "price_before",
        "price_after",
        "currency",
        "product_start_date",
        "product_end_date",
        "date_source",
        "date_badge_text",
        "bbox",
    ]


def copy_bbox(box):
    if not valid_bbox(box):
        return None
    return [float(value) for value in box]


def serialize_bbox(box):
    normalized = copy_bbox(box)
    if normalized is None:
        return None
    return json.dumps(normalized)


def initialize_bbox_column(df):
    """Normalize bbox and discard obsolete duplicate bbox fields."""
    df = df.drop(columns=["ai_bbox", "corrected_bbox"], errors="ignore")

    if "bbox" not in df.columns:
        df["bbox"] = None

    for idx in df.index:
        df.at[idx, "bbox"] = copy_bbox(df.at[idx, "bbox"])

    return df


def save_bbox_edit(row_index, box, notice_key):
    normalized = copy_bbox(box)
    if normalized is None:
        return

    edited_df = st.session_state["edited_df"].copy()
    edited_df.at[row_index, "bbox"] = list(normalized)
    st.session_state["edited_df"] = edited_df
    page = int(edited_df.at[row_index, "page"])
    st.session_state.pop(f"product_table_{page}", None)
    st.session_state[notice_key] = True


def save_bbox_from_inputs(row_index, input_keys, notice_key):
    box = [st.session_state.get(key) for key in input_keys]
    save_bbox_edit(row_index, box, notice_key)


def correction_text(value):
    if isinstance(value, (list, tuple)):
        return json.dumps(list(value))

    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass

    return str(value)


def log_corrections(original_df, edited_df, review_id):
    changes = []
    now = datetime.now(timezone.utc).isoformat()

    for idx in edited_df.index:
        if idx not in original_df.index:
            continue

        for field in editable_columns():
            old = original_df.at[idx, field] if field in original_df.columns else None
            new = edited_df.at[idx, field] if field in edited_df.columns else None

            old_text = correction_text(old)
            new_text = correction_text(new)

            if old_text != new_text:
                changes.append(
                    {
                        "review_id": review_id,
                        "timestamp": now,
                        "page": edited_df.at[idx, "page"],
                        "product_name": edited_df.at[idx, "product_name"],
                        "field": field,
                        "ai_value": old_text,
                        "corrected_value": new_text,
                    }
                )

    if changes:
        write_header = not CORRECTIONS_PATH.exists()
        with CORRECTIONS_PATH.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=changes[0].keys())
            if write_header:
                writer.writeheader()
            writer.writerows(changes)

    return len(changes)


def save_review(result, edited_df, status):
    review_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    reviewed_at = datetime.now(timezone.utc).isoformat()

    corrections = log_corrections(
        st.session_state["original_df"],
        edited_df,
        review_id,
    )

    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """
        INSERT INTO reviews (
            review_id, reviewed_at, status,
            shop_name, campaign_name, product_count
        ) VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            review_id,
            reviewed_at,
            status,
            result.get("shop_name"),
            result.get("campaign_name"),
            len(edited_df),
        ),
    )

    if status == "approved":
        for _, row in edited_df.iterrows():
            bbox = serialize_bbox(row.get("bbox"))

            conn.execute(
                """
                INSERT INTO approved_products (
                    review_id, reviewed_at,
                    shop_name, campaign_name, region, branch,
                    flyer_start_date, flyer_end_date, page,
                    product_name, quantity, price_before, price_after,
                    currency, product_start_date, product_end_date,
                    date_source, date_badge_text, bbox
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    review_id,
                    reviewed_at,
                    row.get("shop_name"),
                    row.get("campaign_name"),
                    row.get("region"),
                    row.get("branch"),
                    row.get("flyer_start_date"),
                    row.get("flyer_end_date"),
                    int(row.get("page")),
                    row.get("product_name"),
                    row.get("quantity"),
                    row.get("price_before"),
                    row.get("price_after"),
                    row.get("currency"),
                    row.get("product_start_date"),
                    row.get("product_end_date"),
                    row.get("date_source"),
                    row.get("date_badge_text"),
                    bbox,
                ),
            )

    conn.commit()
    conn.close()
    return review_id, corrections


def is_missing_value(value):
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def display_text(value):
    return "" if is_missing_value(value) else str(value)


def product_review_issues(row):
    issues = []
    if is_missing_value(row.get("product_name")):
        issues.append("Missing product name")
    if is_missing_value(row.get("price_after")):
        issues.append("Missing sale price")
    if is_missing_value(row.get("product_start_date")):
        issues.append("Missing start date")
    if is_missing_value(row.get("product_end_date")):
        issues.append("Missing end date")
    if not valid_bbox(row.get("bbox")):
        issues.append("Invalid bbox")
    return issues


def product_review_status(row):
    return "Needs Review" if product_review_issues(row) else "Complete"


def add_review_status(df):
    display_df = df.copy()
    display_df["review_status"] = display_df.apply(
        product_review_status,
        axis=1,
    )
    return display_df


TABLE_COLUMN_FIELDS = {
    "Product Name": "product_name",
    "Quantity": "quantity",
    "Old Price": "price_before",
    "New Price": "price_after",
    "Currency": "currency",
    "Start Date": "product_start_date",
    "End Date": "product_end_date",
    "Date Source": "date_source",
}


def build_product_table(page_df, selected_index=None):
    table = pd.DataFrame(index=page_df.index)
    table["Status"] = page_df.apply(product_review_status, axis=1)
    table["Preview"] = table.index == selected_index
    for label, field in TABLE_COLUMN_FIELDS.items():
        if field in {"price_before", "price_after"}:
            table[label] = pd.to_numeric(page_df[field], errors="coerce")
        else:
            table[label] = page_df[field].map(display_text)
    return table


def _table_value(field, value):
    if is_missing_value(value):
        return None
    if field in {"price_before", "price_after"}:
        return float(value)
    return value


def _same_table_value(field, old_value, new_value):
    if is_missing_value(old_value) and is_missing_value(new_value):
        return True
    if field in {"price_before", "price_after"}:
        try:
            return float(old_value) == float(new_value)
        except (TypeError, ValueError):
            return False
    return str(old_value) == str(new_value)


def sync_product_table(page, table):
    edited_df = st.session_state["edited_df"]
    updated_df = None

    for row_index in table.index:
        if row_index not in edited_df.index:
            continue
        for label, field in TABLE_COLUMN_FIELDS.items():
            new_value = _table_value(field, table.at[row_index, label])
            old_value = edited_df.at[row_index, field]
            if _same_table_value(field, old_value, new_value):
                continue
            if updated_df is None:
                updated_df = edited_df.copy()
            updated_df.at[row_index, field] = new_value
            st.session_state.pop(
                f"product_field_{page}_{row_index}_{field}",
                None,
            )

    if updated_df is None:
        return False
    st.session_state["edited_df"] = updated_df
    return True


def sync_table_selection(page, table):
    selection_key = f"table_selected_product_{page}"
    current_index = st.session_state.get(selection_key)
    selected_indices = [
        row_index
        for row_index in table.index
        if bool(table.at[row_index, "Preview"])
    ]

    if current_index in selected_indices:
        newly_selected = [
            row_index
            for row_index in selected_indices
            if row_index != current_index
        ]
        selected_index = newly_selected[-1] if newly_selected else current_index
    else:
        selected_index = selected_indices[-1] if selected_indices else None

    if selected_index == current_index:
        return False
    if selected_index is None:
        st.session_state.pop(selection_key, None)
    else:
        st.session_state[selection_key] = selected_index
        st.session_state[f"selected_product_{page}"] = selected_index
    return True


def _clean_edited_value(value):
    if isinstance(value, str) and not value.strip():
        return None
    return value


def optional_float(value):
    if is_missing_value(value):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def update_selected_product(row_index, field_keys):
    edited_df = st.session_state["edited_df"].copy()
    for field, key in field_keys.items():
        edited_df.at[row_index, field] = _clean_edited_value(
            st.session_state.get(key)
        )
    st.session_state["edited_df"] = edited_df
    for key in list(st.session_state):
        if str(key).startswith(("full_table_", "product_table_")):
            del st.session_state[key]


def save_selected_product_edit(row_index, field_keys, notice_key):
    update_selected_product(row_index, field_keys)
    st.session_state[notice_key] = True


def set_current_page(page):
    st.session_state["current_page"] = int(page)


def set_selected_product(page, row_index):
    st.session_state[f"selected_product_{page}"] = row_index


def approve_page_and_advance(page, pages, row_index, field_keys):
    if row_index is not None and field_keys:
        update_selected_product(row_index, field_keys)

    reviewed_pages = set(st.session_state.get("reviewed_pages", []))
    reviewed_pages.add(int(page))
    st.session_state["reviewed_pages"] = sorted(reviewed_pages)

    page_position = pages.index(page)
    if page_position < len(pages) - 1:
        st.session_state["current_page"] = pages[page_position + 1]
    else:
        st.session_state["show_finalize"] = True


def render_page_navigation(pages, page, key_prefix):
    page_position = pages.index(page)
    previous_page = pages[page_position - 1] if page_position > 0 else page
    next_page = (
        pages[page_position + 1]
        if page_position < len(pages) - 1
        else page
    )
    previous_col, page_col, next_col = st.columns([1, 1.2, 1])
    with previous_col:
        st.button(
            "← Previous Page",
            disabled=page_position == 0,
            width="stretch",
            key=f"{key_prefix}_previous",
            on_click=set_current_page,
            args=(previous_page,),
        )
    with page_col:
        st.markdown(
            f"<div style='text-align:center;padding:.25rem 0;font-weight:750;'>"
            f"Page {page_position + 1} of {len(pages)}</div>",
            unsafe_allow_html=True,
        )
    with next_col:
        st.button(
            "Next Page →",
            disabled=page_position == len(pages) - 1,
            width="stretch",
            key=f"{key_prefix}_next",
            on_click=set_current_page,
            args=(next_page,),
        )


def render_page_progress(pages, current_page, reviewed_pages):
    reviewed = set(reviewed_pages)
    chips = []
    for page in pages:
        if page == current_page:
            css_class = "current"
            marker = "●"
        elif page in reviewed:
            css_class = "reviewed"
            marker = "✓"
        else:
            css_class = ""
            marker = "○"
        chips.append(
            f'<span class="page-chip {css_class}">{marker}&nbsp;{page}</span>'
        )
    st.markdown(
        '<div class="page-progress">' + "".join(chips) + "</div>",
        unsafe_allow_html=True,
    )


def usage_total(result, field):
    values = []
    for row in result.get("_usage", []):
        if field not in row or row.get(field) is None:
            continue
        try:
            values.append(float(row[field]))
        except (TypeError, ValueError):
            continue
    return sum(values) if values else None


def dashboard_values(result, product_count):
    usage_rows = result.get("_usage", [])
    yolo_detections = usage_total(result, "yolo_detections")
    qwen_calls = usage_total(result, "qwen_calls")
    if qwen_calls is None and usage_rows:
        qwen_calls = usage_total(result, "contact_sheets")

    return {
        "products": product_count,
        "pages": len(result.get("_page_images", [])),
        "processing_time": st.session_state.get("processing_time_seconds"),
        "cost": usage_total(result, "cost"),
        "yolo_detections": yolo_detections,
        "qwen_calls": qwen_calls,
        "contact_sheets": usage_total(result, "contact_sheets"),
        "yolo_latency": usage_total(result, "yolo_latency"),
    }


def format_seconds(value):
    try:
        return f"{float(value):.2f} s"
    except (TypeError, ValueError):
        return None


def render_overview_section(title, rows):
    st.markdown(
        f'<div class="overview-section">{html.escape(title.upper())}</div>',
        unsafe_allow_html=True,
    )
    for label, value in rows:
        safe_value = "—" if value in (None, "") else str(value)
        st.markdown(
            '<div class="overview-row">'
            f'<span class="overview-label">{html.escape(label)}</span>'
            f'<span class="overview-value">{html.escape(safe_value)}</span>'
            "</div>",
            unsafe_allow_html=True,
        )


def reset_current_flyer():
    exact_keys = {
        "flyer_result",
        "original_df",
        "edited_df",
        "processing_time_seconds",
        "current_page",
        "reviewed_pages",
        "show_finalize",
        "finalized",
        "final_review_status",
        "final_correction_count",
        "processing_complete_notice",
        "app_view",
        "review_mode",
        "flyer_pdf_upload",
    }
    prefixes = (
        "bbox_input_",
        "bbox_saved_",
        "selected_product_",
        "product_field_",
        "product_saved_",
        "approve_page_",
        "product_table_",
        "table_selected_product_",
        "previous_product_",
        "next_product_",
    )
    for key in list(st.session_state):
        if key in exact_keys or str(key).startswith(prefixes):
            del st.session_state[key]


init_database()

api_key = os.environ.get("OPENROUTER_API_KEY", "").strip()

if "flyer_result" not in st.session_state:
    st.markdown("### Upload Flyer")
    uploaded_pdf = st.file_uploader(
        "Flyer PDF",
        type=["pdf"],
        help="Upload the original supermarket flyer in PDF format.",
        key="flyer_pdf_upload",
    )

    if not api_key:
        st.warning("Set OPENROUTER_API_KEY locally to enable live extraction.")

    ready = uploaded_pdf is not None and bool(api_key)
    process_clicked = st.button(
        "Process Flyer",
        type="primary",
        disabled=not ready,
        width="stretch",
    )

    if process_clicked and uploaded_pdf is not None:
        session_dir = tempfile.mkdtemp(prefix="flyer_ai_local_")
        pdf_path = Path(session_dir) / uploaded_pdf.name
        pdf_path.write_bytes(uploaded_pdf.getvalue())
        processing_started = time.perf_counter()

        try:
            with st.status("Processing flyer…", expanded=True) as processing_status:
                st.write("Rendering pages")
                st.write("Detecting products with YOLO")
                st.write("Reading products with Qwen")
                st.write("Resolving dates")
                result = process_flyer(
                    pdf_path=pdf_path,
                    work_dir=session_dir,
                    api_key=api_key,
                    model_id=MODEL_ID,
                )

                df = pd.DataFrame(result["products"])
                required = {
                    "page": None,
                    "product_name": None,
                    "quantity": None,
                    "price_before": None,
                    "price_after": None,
                    "currency": None,
                    "product_start_date": None,
                    "product_end_date": None,
                    "date_source": None,
                    "date_badge_text": None,
                    "bbox": None,
                }
                for col, default in required.items():
                    if col not in df.columns:
                        df[col] = default

                df = initialize_bbox_column(df)
                flyer_fields = [
                    "shop_name",
                    "campaign_name",
                    "region",
                    "branch",
                    "flyer_start_date",
                    "flyer_end_date",
                ]
                for field in flyer_fields:
                    if field not in df.columns:
                        df[field] = result.get(field)

                st.session_state["flyer_result"] = result
                st.session_state["original_df"] = df.copy(deep=True)
                st.session_state["edited_df"] = df.copy(deep=True)
                st.session_state["processing_time_seconds"] = (
                    time.perf_counter() - processing_started
                )
                st.session_state["current_page"] = 1
                st.session_state["reviewed_pages"] = []
                st.session_state["show_finalize"] = False
                st.session_state["finalized"] = False
                st.session_state["app_view"] = "Review"
                st.session_state["review_mode"] = "Table View"
                st.session_state["processing_complete_notice"] = True
                st.session_state.pop("final_review_status", None)
                st.session_state.pop("final_correction_count", None)

                for key in list(st.session_state):
                    if str(key).startswith(
                        (
                            "bbox_input_",
                            "selected_product_",
                            "product_field_",
                        )
                    ):
                        del st.session_state[key]

                processing_status.update(
                    label="Flyer ready for review",
                    state="complete",
                    expanded=False,
                )
            st.rerun()

        except Exception as exc:
            st.session_state.pop("processing_time_seconds", None)
            st.error(
                "The flyer could not be processed. Check the input and connection "
                "settings, then try again."
            )
            with st.expander("Technical error details"):
                st.exception(exc)

    st.stop()


if "flyer_result" in st.session_state:
    result = st.session_state["flyer_result"]
    original_df = st.session_state["original_df"]
    edited_df = st.session_state["edited_df"]
    
    nav_col, new_flyer_col = st.columns([4, 1])
    with nav_col:
        view = st.radio(
            "View",
            ["Review", "Run Overview"],
            horizontal=True,
            label_visibility="collapsed",
            key="app_view",
        )
    with new_flyer_col:
        st.button(
            "New Flyer",
            width="stretch",
            on_click=reset_current_flyer,
        )
    
    if st.session_state.pop("processing_complete_notice", False):
        st.success("Flyer processed. Review can begin.")
    
    if view == "Run Overview":
        dashboard = dashboard_values(result, len(edited_df))
        timing = result.get("_timing", {})
        shop_name = display_text(result.get("shop_name")) or "Processed flyer"
        campaign_name = display_text(result.get("campaign_name"))
        start_date = display_text(result.get("flyer_start_date"))
        end_date = display_text(result.get("flyer_end_date"))
        flyer_dates = (
            f"{start_date} → {end_date}"
            if start_date or end_date
            else "—"
        )
    
        st.markdown(f"## {html.escape(shop_name)}", unsafe_allow_html=True)
        if campaign_name:
            st.caption(campaign_name)
        st.caption(flyer_dates)
    
        render_overview_section(
            "Flyer",
            [
                ("Shop", shop_name),
                ("Campaign", campaign_name),
                ("Flyer dates", flyer_dates),
                ("Region", display_text(result.get("region"))),
                ("Branch", display_text(result.get("branch"))),
            ],
        )
        render_overview_section(
            "Processing",
            [
                ("Products detected", dashboard["products"]),
                ("Pages", dashboard["pages"]),
                (
                    "Processing time",
                    f'{dashboard["processing_time"]:.1f} s'
                    if dashboard["processing_time"] is not None
                    else None,
                ),
                (
                    "API cost",
                    "$" + f'{dashboard["cost"]:.6f}'
                    if dashboard["cost"] is not None
                    else None,
                ),
                (
                    "YOLO detections",
                    int(dashboard["yolo_detections"])
                    if dashboard["yolo_detections"] is not None
                    else None,
                ),
                (
                    "YOLO latency",
                    f'{dashboard["yolo_latency"]:.2f} s'
                    if dashboard["yolo_latency"] is not None
                    else None,
                ),
                (
                    "Qwen calls",
                    int(dashboard["qwen_calls"])
                    if dashboard["qwen_calls"] is not None
                    else None,
                ),
                (
                    "Contact sheets",
                    int(dashboard["contact_sheets"])
                    if dashboard["contact_sheets"] is not None
                    else None,
                ),
            ],
        )
        render_overview_section(
            "Pipeline Performance",
            [
                (
                    "PDF Rendering",
                    format_seconds(timing.get("pdf_render_seconds")),
                ),
                (
                    "Flyer Context",
                    format_seconds(timing.get("flyer_context_seconds")),
                ),
                ("YOLO Detection", format_seconds(timing.get("yolo_seconds"))),
                (
                    "Product Reading",
                    format_seconds(timing.get("product_qwen_seconds")),
                ),
                (
                    "Section Dates",
                    format_seconds(timing.get("section_date_qwen_seconds")),
                ),
                (
                    "Total",
                    format_seconds(timing.get("total_processing_seconds")),
                ),
            ],
        )
        tracing_value = os.environ.get("LANGSMITH_TRACING", "").strip().lower()
        langsmith_enabled = bool(os.environ.get("LANGSMITH_API_KEY")) and (
            tracing_value in {"1", "true", "yes", "on"}
        )
        render_overview_section(
            "Technical",
            [
                ("Qwen model", result.get("_model") or MODEL_ID),
                ("Detection", "YOLO11n · bestyolo.pt"),
                ("Tracing", "LangSmith enabled" if langsmith_enabled else "Not configured"),
            ],
        )
        st.stop()
    
    
    page_images = result.get("_page_images", [])
    if not page_images:
        st.warning("No rendered flyer pages are available.")
        st.stop()
    
    pages = list(range(1, len(page_images) + 1))
    if st.session_state.get("current_page") not in pages:
        st.session_state["current_page"] = pages[0]
    page = int(st.session_state["current_page"])
    reviewed_pages = sorted(
        {
            int(reviewed_page)
            for reviewed_page in st.session_state.get("reviewed_pages", [])
            if int(reviewed_page) in pages
        }
    )
    st.session_state["reviewed_pages"] = reviewed_pages
    
    shop_name = display_text(result.get("shop_name")) or "Flyer Review"
    st.markdown(f"## {html.escape(shop_name)}", unsafe_allow_html=True)
    render_page_navigation(pages, page, "review_navigation")
    st.caption(f"Reviewed {len(reviewed_pages)} / {len(pages)}")
    render_page_progress(pages, page, reviewed_pages)
    
    page_df = edited_df[edited_df["page"].fillna(0).astype(int) == page].copy()
    image_path = page_images[page - 1]
    selected_idx = None
    field_keys = {}

    _, mode_col = st.columns([1.0, 1.4], gap="large")
    with mode_col:
        review_mode = st.radio(
            "Review mode",
            ["Table View", "Flyer View", "Product View"],
            horizontal=True,
            label_visibility="collapsed",
            key="review_mode",
        )
    
    if review_mode == "Table View":
        table_col = st.container()
        with table_col:
            statuses = page_df.apply(product_review_status, axis=1)
            complete_count = int((statuses == "Complete").sum())
            needs_review_count = int((statuses == "Needs Review").sum())
            st.markdown("### Products")
            st.caption(
                f"{len(page_df)} Products · {complete_count} Complete · "
                f"{needs_review_count} Needs Review"
            )
            if page_df.empty:
                st.info("No products were detected on this page.")
            else:
                table = build_product_table(
                    page_df,
                    st.session_state.get(f"table_selected_product_{page}"),
                )
                edited_table = st.data_editor(
                    table,
                    width="stretch",
                    height=480,
                    hide_index=True,
                    num_rows="fixed",
                    disabled=["Status"],
                    key=f"product_table_{page}",
                    column_config={
                        "Status": st.column_config.TextColumn(width="small"),
                        "Preview": st.column_config.CheckboxColumn(
                            "Preview",
                            help="Select one product to show its bounding box.",
                            width="small",
                        ),
                        "Product Name": st.column_config.TextColumn(width="large"),
                        "Quantity": st.column_config.TextColumn(width="medium"),
                        "Old Price": st.column_config.NumberColumn(
                            min_value=0.0,
                            format="%.3f",
                            width="small",
                        ),
                        "New Price": st.column_config.NumberColumn(
                            min_value=0.0,
                            format="%.3f",
                            width="small",
                        ),
                        "Currency": st.column_config.TextColumn(width="small"),
                        "Start Date": st.column_config.TextColumn(width="medium"),
                        "End Date": st.column_config.TextColumn(width="medium"),
                        "Date Source": st.column_config.TextColumn(width="medium"),
                    },
                )
                table_changed = sync_product_table(page, edited_table)
                selection_changed = sync_table_selection(page, edited_table)
                if selection_changed:
                    st.session_state.pop(f"product_table_{page}", None)
                    st.rerun()
                if table_changed:
                    st.rerun()
    else:
        left, right = st.columns([1.45, 1.0], gap="large")
        saved_bbox = None

        with right:
            st.markdown("### Products")
            if page_df.empty:
                st.info("No products were detected on this page.")
            else:
                product_indices = page_df.index.tolist()
                selected_idx = st.selectbox(
                    "Select Product",
                    product_indices,
                    format_func=lambda idx: (
                        display_text(page_df.at[idx, "product_name"])
                        or f"Product {product_indices.index(idx) + 1}"
                    ),
                    key=f"selected_product_{page}",
                )
                product_position = product_indices.index(selected_idx)
                previous_product = (
                    product_indices[product_position - 1]
                    if product_position > 0
                    else selected_idx
                )
                next_product = (
                    product_indices[product_position + 1]
                    if product_position < len(product_indices) - 1
                    else selected_idx
                )
                previous_product_col, next_product_col = st.columns(2)
                with previous_product_col:
                    st.button(
                        "← Previous Product",
                        disabled=product_position == 0,
                        width="stretch",
                        key=f"previous_product_{page}",
                        on_click=set_selected_product,
                        args=(page, previous_product),
                    )
                with next_product_col:
                    st.button(
                        "Next Product →",
                        disabled=product_position == len(product_indices) - 1,
                        width="stretch",
                        key=f"next_product_{page}",
                        on_click=set_selected_product,
                        args=(page, next_product),
                    )

                selected_row = st.session_state["edited_df"].loc[selected_idx]
                selected_issues = product_review_issues(selected_row)
                selected_status = (
                    "Needs Review" if selected_issues else "Complete"
                )
                status_class = (
                    "status-review" if selected_issues else "status-good"
                )
                selected_name = display_text(
                    selected_row.get("product_name")
                ) or f"Product {product_position + 1}"
                st.markdown(
                    '<div style="display:flex;align-items:center;'
                    'justify-content:space-between;gap:1rem;margin:.2rem 0 .5rem;">'
                    f'<strong>{html.escape(selected_name)}</strong>'
                    f'<span class="{status_class}">{selected_status}</span>'
                    "</div>",
                    unsafe_allow_html=True,
                )
                if selected_issues:
                    st.markdown(
                        '<div class="issue-list">'
                        + " · ".join(
                            html.escape(issue) for issue in selected_issues
                        )
                        + "</div>",
                        unsafe_allow_html=True,
                    )

                field_keys = {
                    field: f"product_field_{page}_{selected_idx}_{field}"
                    for field in (
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
                }
                st.text_input(
                    "Product Name",
                    value=display_text(selected_row.get("product_name")),
                    key=field_keys["product_name"],
                )
                quantity_col, currency_col = st.columns(2)
                with quantity_col:
                    st.text_input(
                        "Quantity",
                        value=display_text(selected_row.get("quantity")),
                        key=field_keys["quantity"],
                    )
                with currency_col:
                    st.text_input(
                        "Currency",
                        value=display_text(selected_row.get("currency")),
                        key=field_keys["currency"],
                    )
                old_price_col, new_price_col = st.columns(2)
                with old_price_col:
                    st.number_input(
                        "Old Price",
                        min_value=0.0,
                        value=optional_float(selected_row.get("price_before")),
                        step=0.001,
                        format="%.3f",
                        key=field_keys["price_before"],
                    )
                with new_price_col:
                    st.number_input(
                        "New Price",
                        min_value=0.0,
                        value=optional_float(selected_row.get("price_after")),
                        step=0.001,
                        format="%.3f",
                        key=field_keys["price_after"],
                    )
                start_date_col, end_date_col = st.columns(2)
                with start_date_col:
                    st.text_input(
                        "Start Date",
                        value=display_text(
                            selected_row.get("product_start_date")
                        ),
                        key=field_keys["product_start_date"],
                    )
                with end_date_col:
                    st.text_input(
                        "End Date",
                        value=display_text(selected_row.get("product_end_date")),
                        key=field_keys["product_end_date"],
                    )
                date_source_col, date_badge_col = st.columns(2)
                with date_source_col:
                    st.text_input(
                        "Date Source",
                        value=display_text(selected_row.get("date_source")),
                        key=field_keys["date_source"],
                    )
                with date_badge_col:
                    st.text_input(
                        "Date Badge",
                        value=display_text(selected_row.get("date_badge_text")),
                        key=field_keys["date_badge_text"],
                    )

                edit_notice_key = f"product_saved_{page}_{selected_idx}"
                st.button(
                    "Save Changes",
                    type="primary",
                    width="stretch",
                    key=f"save_product_{page}_{selected_idx}",
                    on_click=save_selected_product_edit,
                    args=(selected_idx, field_keys, edit_notice_key),
                )
                if st.session_state.pop(edit_notice_key, False):
                    st.success("Changes saved.")

                saved_bbox = copy_bbox(selected_row.get("bbox"))
                input_bbox = saved_bbox or [0.0, 0.0, 0.0, 0.0]
                with st.expander("Adjust Bounding Box"):
                    box_cols = st.columns(4)
                    coordinate_names = ("X1", "Y1", "X2", "Y2")
                    input_keys = [
                        f"bbox_input_{page}_{selected_idx}_{name.lower()}"
                        for name in coordinate_names
                    ]
                    coordinate_values = []
                    for position, (column, name, input_key) in enumerate(
                        zip(box_cols, coordinate_names, input_keys)
                    ):
                        with column:
                            coordinate_values.append(
                                st.number_input(
                                    name,
                                    min_value=0.0,
                                    max_value=1000.0,
                                    value=float(input_bbox[position]),
                                    step=1.0,
                                    key=input_key,
                                )
                            )

                    candidate_bbox = [
                        float(value) for value in coordinate_values
                    ]
                    box_is_valid = valid_bbox(candidate_bbox)
                    if not box_is_valid:
                        st.warning("Use X1 < X2 and Y1 < Y2.")
                    bbox_notice_key = f"bbox_saved_{page}_{selected_idx}"
                    st.button(
                        "Save Bounding Box",
                        width="stretch",
                        disabled=not box_is_valid,
                        key=f"save_bbox_{page}_{selected_idx}",
                        on_click=save_bbox_from_inputs,
                        args=(selected_idx, input_keys, bbox_notice_key),
                    )
                    if st.session_state.pop(bbox_notice_key, False):
                        st.success("Bounding box saved.")

        with left:
            if review_mode == "Flyer View":
                st.markdown("### Flyer")
                if selected_idx is not None and valid_bbox(saved_bbox):
                    st.image(draw_bbox(image_path, saved_bbox), width="stretch")
                else:
                    st.image(image_path, width="stretch")
            else:
                st.markdown("### Product")
                zoomed_crop = (
                    crop_bbox_preview(image_path, saved_bbox)
                    if selected_idx is not None
                    else None
                )
                if zoomed_crop is not None:
                    st.image(zoomed_crop, width="stretch")
                else:
                    st.info("Product crop unavailable")
    
    st.write("")
    approve_col = st.columns([1, 2, 1])[1]
    page_position = pages.index(page)
    approve_page_label = (
        "Approve Page & Finish"
        if page_position == len(pages) - 1
        else "Approve Page & Next"
    )
    with approve_col:
        st.button(
            approve_page_label,
            type="primary",
            width="stretch",
            key=f"approve_page_{page}",
            on_click=approve_page_and_advance,
            args=(page, pages, selected_idx, field_keys),
        )
    
    reviewed_pages = sorted(
        {
            int(reviewed_page)
            for reviewed_page in st.session_state.get("reviewed_pages", [])
            if int(reviewed_page) in pages
        }
    )
    show_finalize = (
        bool(st.session_state.get("show_finalize"))
        or len(reviewed_pages) == len(pages)
        or bool(st.session_state.get("finalized"))
    )
    
    if show_finalize:
        st.divider()
        finalized = bool(st.session_state.get("finalized"))
        final_status = st.session_state.get("final_review_status")
    
        if not finalized:
            st.markdown("### Review complete")
            st.caption(f"{len(reviewed_pages)} / {len(pages)} pages reviewed")
            if len(reviewed_pages) < len(pages):
                st.warning(
                    f"{len(pages) - len(reviewed_pages)} page(s) are not marked reviewed."
                )
            finalize_col, reject_col = st.columns(2)
            with finalize_col:
                if st.button(
                    "Finalize Flyer",
                    type="primary",
                    width="stretch",
                ):
                    review_id, correction_count = save_review(
                        result,
                        st.session_state["edited_df"],
                        "approved",
                    )
                    st.session_state["finalized"] = True
                    st.session_state["final_review_status"] = "approved"
                    st.session_state["final_correction_count"] = correction_count
                    st.rerun()
            with reject_col:
                if st.button("Reject Flyer", width="stretch"):
                    review_id, correction_count = save_review(
                        result,
                        st.session_state["edited_df"],
                        "rejected",
                    )
                    st.session_state["finalized"] = True
                    st.session_state["final_review_status"] = "rejected"
                    st.session_state["final_correction_count"] = correction_count
                    st.rerun()
    
        elif final_status == "approved":
            correction_count = st.session_state.get("final_correction_count", 0)
            st.success("Flyer approved")
            st.markdown(f"### {len(st.session_state['edited_df'])} products reviewed")
            st.caption(f"{correction_count} correction(s) logged")
            st.download_button(
                "Download Reviewed CSV",
                data=st.session_state["edited_df"].to_csv(index=False).encode("utf-8"),
                file_name="reviewed_flyer_products.csv",
                mime="text/csv",
                type="primary",
                width="stretch",
            )
        else:
            st.warning("Flyer rejected")
