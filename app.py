"""
Maize Disease Progression API
─────────────────────────────
Optimised for Render free tier deployment.

Key design decisions:
  • Models load LAZILY on first request (not at import time) so gunicorn
    can bind the port immediately and pass Render's port scan.
  • After first load, models are cached globally for instant subsequent requests.
  • Two-stage pipeline: YOLOv8 (maize vs non-maize) -> Keras (disease).
  • Environment variables set BEFORE any heavy imports.
"""

# ─── CRITICAL: Set environment variables BEFORE any heavy imports ──────────────
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["YOLO_CONFIG_DIR"] = "/tmp/Ultralytics"
os.environ["MPLBACKEND"] = "Agg"
os.environ["OMP_NUM_THREADS"] = "1"

import base64
import json
import uuid
import warnings
from datetime import datetime
from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename

warnings.filterwarnings("ignore")

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# ─── Config ────────────────────────────────────────────────────────────────────
UPLOAD_FOLDER = "storage/uploads"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
API_VERSION = "v1"
FRONTEND_ORIGINS = [
    o.strip() for o in os.getenv("FRONTEND_ORIGINS", "*").split(",") if o.strip()
]

# ─── CORS ──────────────────────────────────────────────────────────────────────
@app.after_request
def add_cors_headers(response):
    origin = request.headers.get("Origin")
    if "*" in FRONTEND_ORIGINS:
        response.headers["Access-Control-Allow-Origin"] = "*"
    elif origin and origin in FRONTEND_ORIGINS:
        response.headers["Access-Control-Allow-Origin"] = origin
        response.headers["Vary"] = "Origin"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return response


# ─── Global error handlers — always return valid JSON ─────────────────────────
@app.errorhandler(400)
def bad_request(e):
    return api_response(False, status=400, error=str(e.description) if hasattr(e, 'description') else 'Bad request')

@app.errorhandler(404)
def not_found(e):
    return api_response(False, status=404, error='Not found')

@app.errorhandler(405)
def method_not_allowed(e):
    return api_response(False, status=405, error='Method not allowed')

@app.errorhandler(413)
def payload_too_large(e):
    return api_response(False, status=413, error='File too large (max 16 MB)')

@app.errorhandler(500)
def internal_error(e):
    return api_response(False, status=500, error='Internal server error')

@app.errorhandler(Exception)
def handle_exception(e):
    import traceback
    print(f"[ERROR] Unhandled exception: {e}", flush=True)
    traceback.print_exc()
    return api_response(False, status=500, error='Server error — please try again')


def api_response(ok, status=200, **payload):
    data = {
        "ok": ok,
        "api_version": API_VERSION,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        **payload,
    }
    resp = jsonify(data)
    resp.status_code = status
    return resp


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL PATHS
# ═══════════════════════════════════════════════════════════════════════════════
YOLO_MODEL_PATHS = ["best1.pt", "best.pt"]
KERAS_MODEL_PATHS = ["maizediseaseprogression.keras"]
DISEASE_LABELS_PATH = "disease_labels.json"
DISEASE_KNOWLEDGE_BASE_PATHS = [
    "disease_knowledge_base_v1.0.json",
    "disease_knowledge_base.json",
]
JOBLIB_MODEL_PATHS = ["askinggreetingmodel.joblib"]
RESPONSE_MAP_PATHS = [
    "askinggreetingmodel_response_map.json",
    "askingmodelmaize_response_map_v1.0.json",
    "askingmodelmaize_response_map_v1.json",
    "askingmodelmaize_response_map.json",
]
INTENT_LABELS_PATHS = [
    "askinggreetingmodel_labels.json",
    "askingmodelmaize_labels_v1.0.json",
    "askingmodelmaize_labels_v1.json",
    "askingmodelmaize_labels.json",
]

# ═══════════════════════════════════════════════════════════════════════════════
# LAZY-LOADED MODEL STATE  (loaded on first request, cached thereafter)
# ═══════════════════════════════════════════════════════════════════════════════
yolo_model = None
keras_model = None
keras_input_shape = None
joblib_model = None
_models_loaded = False

response_map = {}
intent_labels = []
labels_list = []
knowledge_base = {}

loaded_yolo_path = None
loaded_keras_path = None
loaded_joblib_path = None
loaded_response_map_path = None
loaded_intent_labels_path = None


def _load_yolo():
    global yolo_model, loaded_yolo_path
    try:
        from ultralytics import YOLO
        for p in YOLO_MODEL_PATHS:
            if os.path.exists(p):
                yolo_model = YOLO(p)
                loaded_yolo_path = p
                print(f"[MODEL] YOLO loaded: {p}  classes={yolo_model.names}", flush=True)
                break
        else:
            print(f"[MODEL] WARNING - YOLO files not found: {', '.join(YOLO_MODEL_PATHS)}", flush=True)
    except ImportError:
        print("[MODEL] WARNING - ultralytics not installed", flush=True)
    except Exception as e:
        print(f"[MODEL] WARNING - YOLO load failed: {e}", flush=True)


def _load_keras():
    global keras_model, keras_input_shape, loaded_keras_path
    try:
        import tensorflow as tf
        tf.get_logger().setLevel("ERROR")
        from tensorflow.keras.models import load_model
        for p in KERAS_MODEL_PATHS:
            if os.path.exists(p):
                keras_model = load_model(p)
                keras_input_shape = getattr(keras_model, "input_shape", (None, 224, 224, 3))
                loaded_keras_path = p
                print(f"[MODEL] Keras loaded: {p}  input_shape={keras_input_shape}", flush=True)
                break
    except Exception as e:
        print(f"[MODEL] WARNING - Keras load failed: {e}", flush=True)


def _load_joblib():
    global joblib_model, loaded_joblib_path
    try:
        import joblib
        for p in JOBLIB_MODEL_PATHS:
            if os.path.exists(p):
                joblib_model = joblib.load(p)
                loaded_joblib_path = p
                print(f"[MODEL] Joblib loaded: {p}", flush=True)
                break
    except Exception as e:
        print(f"[MODEL] WARNING - Joblib load failed: {e}", flush=True)


def _load_json_artefacts():
    global response_map, intent_labels, labels_list, knowledge_base

    for kb_path in DISEASE_KNOWLEDGE_BASE_PATHS:
        if os.path.exists(kb_path):
            try:
                with open(kb_path, "r", encoding="utf-8") as f:
                    knowledge_base = json.load(f)
                print(f"[DATA] Knowledge base: {kb_path} ({len(knowledge_base)} entries)", flush=True)
                break
            except Exception:
                pass

    for rp in RESPONSE_MAP_PATHS:
        if os.path.exists(rp):
            try:
                with open(rp, "r", encoding="utf-8") as f:
                    raw = json.load(f)
                for k, v in raw.items():
                    response_map[k] = [s.strip() for s in v.split(",")] if isinstance(v, str) and "," in v else v
                loaded_response_map_path = rp
                print(f"[DATA] Response map: {rp} ({len(response_map)} categories)", flush=True)
                break
            except Exception:
                pass

    for lp in INTENT_LABELS_PATHS:
        if os.path.exists(lp):
            try:
                with open(lp, "r", encoding="utf-8") as f:
                    d = json.load(f)
                intent_labels.extend(d.get("intent_list", d) if isinstance(d, dict) else d)
                loaded_intent_labels_path = lp
                print(f"[DATA] Intent labels: {lp} ({len(intent_labels)} intents)", flush=True)
                break
            except Exception:
                pass

    if os.path.exists(DISEASE_LABELS_PATH):
        try:
            with open(DISEASE_LABELS_PATH, "r", encoding="utf-8") as f:
                labels_list.extend(json.load(f))
            print(f"[DATA] Disease labels: {DISEASE_LABELS_PATH} ({len(labels_list)} classes)", flush=True)
        except Exception:
            pass
    else:
        labels_list.extend([
            "Healthy_maize", "Common_Rust_disease", "Gray_Leaf_Spot_disease",
            "Leaf_Blight_disease", "Downy_Mildew_disease",
            "Maize_Streak_Virus_disease", "Maize_Lethal_Necrosis_disease",
        ])
        print(f"[DATA] Using default disease labels ({len(labels_list)} classes)", flush=True)


def _ensure_models_loaded():
    """Load all models once on first request. Safe to call multiple times."""
    global _models_loaded
    if _models_loaded:
        return True
    print("[STARTUP] Loading models on first request...", flush=True)
    try:
        _load_yolo()
        _load_keras()
        _load_joblib()
        _models_loaded = True
        print("[STARTUP] All models loaded.", flush=True)
        return True
    except Exception as e:
        print(f"[STARTUP] Model loading failed: {e}", flush=True)
        return False


# ─── Load JSON artefacts at import time (fast, no heavy deps) ─────────────────
_load_json_artefacts()


# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════════
def normalize_disease_name(label_name):
    mapping = {
        "Healthy_maize": "Healthy Maize",
        "Leaf_Blight_disease": "Leaf Blight Disease",
        "Common_Rust_disease": "Common Rust Disease",
        "Gray_Leaf_Spot_disease": "Gray Leaf Spot Disease",
        "Downy_Mildew_disease": "Downy Mildew Disease",
        "Maize_Lethal_Necrosis_disease": "Maize Lethal Necrosis Disease",
        "Maize_Streak_Virus_disease": "Maize Streak Virus Disease",
    }
    return mapping.get(label_name, str(label_name).replace("_", " ").strip())


def knowledge_key_from_label(label_name):
    key = str(label_name).strip().lower().replace(" ", "_").replace("-", "_")
    if key.endswith("_disease"):
        key = key[:-8]
    return key


def _to_list(value):
    if value is None:
        return []
    return value if isinstance(value, list) else [str(value)]


def get_disease_entry(label_name):
    key = knowledge_key_from_label(label_name)
    if key in knowledge_base:
        return key, knowledge_base[key]
    aliases = {
        "healthy": "healthy_maize",
        "common_rust_disease": "common_rust",
        "gray_leaf_spot_disease": "gray_leaf_spot",
        "leaf_blight_disease": "leaf_blight",
        "downy_mildew_disease": "downy_mildew",
        "maize_streak_virus_disease": "maize_streak_virus",
        "maize_lethal_necrosis_disease": "maize_lethal_necrosis",
    }
    alias_key = aliases.get(key)
    if alias_key and alias_key in knowledge_base:
        return alias_key, knowledge_base[alias_key]
    return key, None


def get_disease_stage(label_name):
    _, entry = get_disease_entry(label_name)
    if entry:
        s = entry.get("stage")
        if s:
            return s
    return "Predicted by model"


def _bullet_text(items, fallback="- Not specified"):
    items = [str(i) for i in items if str(i).strip()]
    return "\n".join(f"- {i}" for i in items) if items else fallback


def build_non_maize_report():
    return (
        "NOT MAIZE\n\n"
        "This image does not contain a maize leaf.\n\n"
        "Please upload a clear image of a maize leaf for disease detection.\n\n"
        "Tips:\n"
        "- Use close-up leaf images\n"
        "- Avoid people, animals, vehicles, or background clutter\n"
        "- Ensure good lighting"
    )


def build_maize_health_report(label_name):
    condition = normalize_disease_name(label_name)
    _, entry = get_disease_entry(label_name)

    if entry:
        stage = entry.get("stage", "Predicted by model")
        causes = []
        if entry.get("cause"):
            causes.append(entry["cause"])
        if entry.get("favorable_conditions"):
            causes.append(entry["favorable_conditions"])
        mgmt = _to_list(entry.get("management"))
        prev = _to_list(entry.get("prevention"))
        desc = entry.get("description", "No description available.")
        prog = entry.get("progression_time", "Not specified")
        risk = entry.get("yield_risk", "Not specified")
        urgency = entry.get("urgency", "Not specified")
        return (
            "MAIZE HEALTH ANALYSIS REPORT\n"
            "---------------------------\n\n"
            f"Disease:\n{condition}\n\n"
            f"Stage:\n{stage}\n\n"
            f"Description:\n{desc}\n\n"
            f"Possible Causes:\n{_bullet_text(causes)}\n\n"
            f"Recommended Actions:\n{_bullet_text(mgmt)}\n\n"
            f"Prevention Tips:\n{_bullet_text(prev)}\n\n"
            f"Urgency Level:\n{urgency}\n\n"
            f"Progression Time:\n{prog}\n\n"
            f"Yield Risk:\n{risk}"
        )

    return (
        "MAIZE HEALTH ANALYSIS REPORT\n"
        "---------------------------\n\n"
        f"Disease:\n{condition}\n\n"
        "Stage:\nPredicted by model\n\n"
        "Description:\nDetailed disease guidance is not available for this label.\n\n"
        "Possible Causes:\n- Not specified\n\n"
        "Recommended Actions:\n- Review the prediction in the UI\n- Inspect the plant visually\n\n"
        "Prevention Tips:\n- Keep using clean seed and regular monitoring\n\n"
        "Urgency Level:\nCheck needed\n\n"
        "Progression Time:\nNot specified"
    )


def _decode_image_data(image_data):
    if not image_data:
        raise ValueError("missing image data")
    if isinstance(image_data, str) and image_data.startswith("data:") and "," in image_data:
        image_data = image_data.split(",", 1)[1]
    image_bytes = base64.b64decode(image_data)
    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
    filename = secure_filename(f"upload_{uuid.uuid4().hex}.png")
    save_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    with open(save_path, "wb") as f:
        f.write(image_bytes)
    return save_path, filename


def _get_payload():
    if request.is_json:
        return request.get_json(silent=True) or {}
    return request.form.to_dict(flat=True)


# ═══════════════════════════════════════════════════════════════════════════════
# INFERENCE
# ═══════════════════════════════════════════════════════════════════════════════
def classify_maize(image_path):
    """Stage 1: YOLOv8 - returns (is_maize, class_name, confidence)."""
    global yolo_model
    if yolo_model is None:
        _load_yolo()
    if yolo_model is None:
        raise RuntimeError("YOLO model not available")
    results = yolo_model(image_path, verbose=False)
    r = results[0]
    if hasattr(r, "probs") and r.probs is not None:
        idx = int(r.probs.top1)
        conf = float(r.probs.top1conf)
        name = yolo_model.names[idx]
    elif hasattr(r, "boxes") and r.boxes is not None and len(r.boxes) > 0:
        best = max(r.boxes, key=lambda b: float(b.conf))
        idx = int(best.cls)
        conf = float(best.conf)
        name = yolo_model.names[idx]
    else:
        raise RuntimeError("YOLO returned no predictions")
    is_maize = name.lower().replace("-", "_").replace(" ", "_") == "maize"
    return is_maize, name, conf


def classify_disease(image_path):
    """Stage 2: Keras - returns (label_name, condition_name, confidence)."""
    global keras_model
    if keras_model is None:
        _load_keras()
    if keras_model is None:
        raise RuntimeError("Keras model not available")
    import numpy as np
    from tensorflow.keras.utils import load_img, img_to_array

    target = tuple(keras_input_shape[1:3]) if keras_input_shape and len(keras_input_shape) > 2 else (224, 224)
    img = load_img(image_path, target_size=target)
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0) / 255.0
    preds = keras_model.predict(x, verbose=0)
    arr = preds[0] if len(preds.shape) > 1 else preds
    idx = int(np.argmax(arr))
    conf = float(arr[idx])
    if labels_list and 0 <= idx < len(labels_list):
        label = str(labels_list[idx])
    else:
        raise RuntimeError("Prediction index out of range")
    return label, normalize_disease_name(label), conf


def handle_image_prediction(save_path, filename, input_source):
    """Full two-stage pipeline."""
    if not _ensure_models_loaded():
        return api_response(False, status=503, error='Models are still loading. Please wait a moment and try again.')
    print(f"\n[PREDICT] {filename}  source={input_source}", flush=True)

    # Stage 1: YOLO maize / non-maize
    try:
        is_maize, yolo_cls, yolo_conf = classify_maize(save_path)
    except Exception as e:
        print(f"  YOLO error: {e}", flush=True)
        return api_response(False, status=500, error=f"Image check failed: {e}")

    print(f"  YOLO: {yolo_cls} ({yolo_conf:.3f})  is_maize={is_maize}", flush=True)

    if not is_maize:
        report = build_non_maize_report()
        return api_response(
            False, status=400, type="image", input_source=input_source,
            valid_image=False, is_maize=False,
            yolo_class=yolo_cls, yolo_confidence=round(yolo_conf, 4),
            error="This is not a maize image. Please upload a maize leaf image.",
            answer=report, report=report, file_name=filename,
        )

    # Stage 2: Keras disease classification
    try:
        label, condition, disease_conf = classify_disease(save_path)
    except Exception as e:
        print(f"  Keras error: {e}", flush=True)
        return api_response(False, status=500, error=f"Disease prediction failed: {e}")

    stage = get_disease_stage(label)
    report = build_maize_health_report(label)
    print(f"  Disease: {condition} ({disease_conf:.3f})  stage={stage}\n", flush=True)

    return api_response(
        True, type="image", input_source=input_source,
        valid_image=True, is_maize=True,
        yolo_class=yolo_cls, yolo_confidence=round(yolo_conf, 4),
        label=label, condition=condition, stage=stage, severity=stage,
        disease_confidence=round(disease_conf, 4),
        answer=report, report=report,
        yolo_model_path=loaded_yolo_path,
        disease_model_path=loaded_keras_path,
        file_name=filename,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# ROUTES
# ═══════════════════════════════════════════════════════════════════════════════
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api", methods=["GET"])
def api_root():
    return api_response(True, message="Maize Disease Progression API", endpoints=[
        "/api/health", "/api/warmup", "/api/model-info", "/api/predict", "/api/chat",
    ])


@app.route("/api/health", methods=["GET"])
def api_health():
    return api_response(True, service="maize-disease-api", status="healthy")


@app.route("/api/warmup", methods=["GET"])
def api_warmup():
    """
    Warmup endpoint - call after deploy to load models before real users hit the app.
    Render / uptime monitors can ping this to pre-load models.
    """
    _ensure_models_loaded()
    return api_response(
        True, status="warm",
        yolo_loaded=yolo_model is not None,
        yolo_classes=yolo_model.names if yolo_model else None,
        keras_loaded=keras_model is not None,
        keras_input_shape=keras_input_shape,
        disease_classes=len(labels_list),
        text_intents=len(intent_labels),
    )


@app.route("/api/model-info", methods=["GET"])
def api_model_info():
    _ensure_models_loaded()
    return api_response(
        True,
        yolo_model=loaded_yolo_path,
        yolo_classes=yolo_model.names if yolo_model else None,
        disease_model=loaded_keras_path,
        disease_input_shape=keras_input_shape,
        disease_classes=len(labels_list),
        text_model=loaded_joblib_path,
        text_intents=len(intent_labels),
        response_map=loaded_response_map_path,
        intent_labels=loaded_intent_labels_path,
    )


@app.route("/api/predict", methods=["POST"])
def api_predict():
    payload = _get_payload()

    # Multipart upload
    if "image" in request.files:
        file = request.files["image"]
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
            save_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(save_path)
            return handle_image_prediction(save_path, filename, "multipart/form-data")
        return api_response(False, status=400, error="Invalid file type")

    # Base64 JSON
    image_data = payload.get("image_base64") or payload.get("image")
    if image_data and isinstance(image_data, str):
        try:
            save_path, filename = _decode_image_data(image_data)
            return handle_image_prediction(save_path, filename, "json-base64")
        except Exception as e:
            return api_response(False, status=400, error=f"Image decode failed: {e}")

    # Text question
    question = (payload.get("question") or payload.get("message") or request.form.get("question") or "").strip()
    if question:
        _ensure_models_loaded()
        response_text = "Sorry, I don't have an answer for that yet."
        label = None
        print(f"\n[TEXT] {question}", flush=True)

        if joblib_model is not None:
            try:
                cleaned = question.lower().strip()
                pred = joblib_model.predict([cleaned])
                label = str(pred[0]).strip().lower()
                raw = response_map.get(label)
                if raw is None:
                    for k in response_map:
                        if k.lower() == label:
                            raw = response_map[k]
                            label = k
                            break
                if raw is None:
                    raw = response_text
                response_text = raw[0] if isinstance(raw, list) and raw else str(raw) if raw else response_text
            except Exception as e:
                print(f"  Joblib error: {e}", flush=True)
                for k in response_map:
                    if k.lower() in question.lower():
                        label, raw = k, response_map[k]
                        response_text = raw[0] if isinstance(raw, list) and raw else str(raw)
                        break
        else:
            for k in response_map:
                if k.lower() in question.lower():
                    label, raw = k, response_map[k]
                    response_text = raw[0] if isinstance(raw, list) and raw else str(raw)
                    break

        print(f"  Response: {response_text[:80]}...\n", flush=True)
        return api_response(True, type="text", label=label, response=response_text, model_path=loaded_joblib_path)

    return api_response(False, status=400, error="No input received (provide question or image)")


@app.route("/api/chat", methods=["POST"])
def api_chat():
    data = _get_payload()
    message = (data.get("message") or data.get("question") or "").strip()
    if not message:
        return api_response(False, status=400, error="empty message")
    for k, v in response_map.items():
        if k.lower() in message.lower():
            return api_response(True, response=v, source="local-response-map", label=k)
    return api_response(
        True,
        response="I am the Maize AI Assistant! I can help you with questions about maize diseases and treatments. Please feel free to ask or upload an image for diagnosis.",
        source="fallback",
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
