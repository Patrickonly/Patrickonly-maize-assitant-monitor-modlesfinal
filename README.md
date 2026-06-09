# Maize Assistant (Flask)

This small Flask app provides a minimal UI to ask questions (text) and upload images for the maize models in this workspace.

Files added:
- `app.py` — Flask server that loads `askingmodelmaize.joblib` (if present) and `models/best_mobilenet.h5` (if present) and exposes `/api/predict`.
- `templates/index.html` — Single-page UI for typing questions and uploading images.
- `.env` — Placeholder for `OPENAI_API_KEY` if you want to enable OpenAI responses.

Run (from project folder):

```bash
python -m pip install -r requirements.txt
python app.py
```

Open http://localhost:5000 in your browser.

Notes:
- The app tries to use `askingmodelmaize.joblib` for text Q&A and `askingmodelmaize_response_map.json` to map labels to responses.
- If you want ChatGPT responses, set `OPENAI_API_KEY` in `.env` and extend `app.py` to call the OpenAI API.
