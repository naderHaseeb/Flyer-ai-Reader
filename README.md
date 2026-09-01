# Flyer AI Reader

Flyer AI Reader is an AI-powered system that extracts structured product information from supermarket promotional flyers.

It combines:

- **YOLO** for detecting products and bounding boxes
- **Qwen Vision-Language Model** for reading product names, quantities, prices, currencies, and promotional dates
- **Pydantic** for validating AI-generated data
- **Streamlit** for human review, editing, approval, and rejection
- **LangSmith** for monitoring model calls, latency, tokens, and cost

## How It Works

Flyer PDF → YOLO Detection → Product Crops → Qwen Extraction → Validation → Human Review → Saved Results

The main idea is simple:

**YOLO finds it.  
Qwen reads it.  
Python connects it.  
Pydantic validates it.  
Human verifies it.**

## Technologies

- Python
- Streamlit
- YOLO
- Qwen3-VL-32B
- OpenRouter
- Pydantic
- PyMuPDF
- Pillow
- Pandas
- SQLite
- LangSmith


## Client
- Voix Me Technologies

## Author
**Nader Haseeb**

General Assembly Data Science Capstone Project
