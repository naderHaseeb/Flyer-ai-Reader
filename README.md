# Flyer AI Reader — Local Version

Use this version locally instead of Streamlit Cloud.

The office OpenRouter key works in Colab but the same generation request is returning 401 from Streamlit Cloud. The capstone brief explicitly does not require deployment, so this local version is enough for the project.

## Mac

Open Terminal in this folder and run:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export OPENROUTER_API_KEY='paste-your-key-here'
streamlit run app.py
```

Then open http://localhost:8501

The OpenRouter key is read from the `OPENROUTER_API_KEY` environment
variable. It is not entered or displayed in the Streamlit page. Set the
variable in the same Terminal window used to launch the app.

## Modes

1. Live extraction with OpenRouter
   - Paste the same office key that works in Colab into the sidebar.
   - Upload the flyer PDF.
   - Click Process flyer.

2. Review existing Colab JSON
   - Run the working final extraction notebook in Colab.
   - Download its final JSON.
   - Upload the PDF + JSON here.
   - Review, edit, highlight boxes, approve/reject, and save corrections.
