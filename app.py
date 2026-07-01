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
import gc
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"      # Reduce TF memory footprint
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"       # Force CPU — skip GPU probing entirely
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
os.environ["MALLOC_TRIM_THRESHOLD_"] = "65536"  # Help Linux reclaim freed memory
os.environ["YOLO_CONFIG_DIR"] = "/tmp/Ultralytics"
os.environ["MPLBACKEND"] = "Agg"
os.environ["OMP_NUM_THREADS"] = "1"

import base64
import json
import threading
import uuid
import warnings
import sys
from unittest.mock import MagicMock
from datetime import datetime

# Prevent ultralytics from loading matplotlib and building font cache
sys.modules["matplotlib"] = MagicMock()
sys.modules["matplotlib.pyplot"] = MagicMock()
sys.modules["matplotlib.font_manager"] = MagicMock()

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
    return api_response(False, http_status=400, error=str(e.description) if hasattr(e, 'description') else 'Bad request')

@app.errorhandler(404)
def not_found(e):
    return api_response(False, http_status=404, error='Not found')

@app.errorhandler(405)
def method_not_allowed(e):
    return api_response(False, http_status=405, error='Method not allowed')

@app.errorhandler(413)
def payload_too_large(e):
    return api_response(False, http_status=413, error='File too large (max 16 MB)')

@app.errorhandler(500)
def internal_error(e):
    return api_response(False, http_status=500, error='Internal server error')

@app.errorhandler(Exception)
def handle_exception(e):
    import traceback
    print(f"[ERROR] Unhandled exception: {e}", flush=True)
    traceback.print_exc()
    return api_response(False, http_status=500, error='Server error — please try again')


def api_response(ok, http_status=200, **payload):
    data = {
        "ok": ok,
        "api_version": API_VERSION,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        **payload,
    }
    resp = jsonify(data)
    resp.status_code = http_status
    return resp


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL PATHS
# ═══════════════════════════════════════════════════════════════════════════════
YOLO_MODEL_PATHS = ["best2.pt"]
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
_yolo_ready = False
_keras_ready = False
_joblib_ready = False
_models_loaded = False   # True when ALL models are loaded
_models_lock = threading.Lock()

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
    global yolo_model, loaded_yolo_path, _yolo_ready
    try:
        from ultralytics import YOLO
        for p in YOLO_MODEL_PATHS:
            if os.path.exists(p):
                yolo_model = YOLO(p)
                yolo_model.fuse()
                loaded_yolo_path = p
                _yolo_ready = True
                print(f"[MODEL] YOLO loaded: {p}  classes={yolo_model.names}", flush=True)
                gc.collect()
                break
        else:
            print(f"[MODEL] WARNING - YOLO not found: {', '.join(YOLO_MODEL_PATHS)}", flush=True)
    except ImportError:
        print("[MODEL] WARNING - ultralytics not installed", flush=True)
    except Exception as e:
        print(f"[MODEL] WARNING - YOLO load failed: {e}", flush=True)


def _load_keras():
    global keras_model, keras_input_shape, loaded_keras_path, _keras_ready
    try:
        # Try loading TFLite first (uses 90% less memory and imports instantly)
        tflite_path = "maizediseaseprogression.tflite"
        if os.path.exists(tflite_path):
            try:
                import tflite_runtime.interpreter as tflite
            except ImportError:
                import tensorflow.lite as tflite
                
            keras_model = tflite.Interpreter(model_path=tflite_path, num_threads=1)
            keras_model.allocate_tensors()
            
            input_details = keras_model.get_input_details()
            # shape is e.g. [1, 224, 224, 3]
            keras_input_shape = input_details[0]['shape']
            loaded_keras_path = tflite_path
            _keras_ready = True
            print(f"[MODEL] TFLite loaded: {tflite_path}  input_shape={keras_input_shape}", flush=True)
            gc.collect()
            return

        # Fallback to Keras if TFLite missing (requires full tensorflow)
        import tensorflow as tf
        tf.get_logger().setLevel("ERROR")
        tf.config.threading.set_intra_op_parallelism_threads(1)
        tf.config.threading.set_inter_op_parallelism_threads(1)
        from tensorflow.keras.models import load_model
        for p in KERAS_MODEL_PATHS:
            if os.path.exists(p):
                keras_model = load_model(p, compile=False)
                keras_input_shape = getattr(keras_model, "input_shape", (None, 224, 224, 3))
                loaded_keras_path = p
                _keras_ready = True
                print(f"[MODEL] Keras loaded: {p}  input_shape={keras_input_shape}", flush=True)
                gc.collect()
                break
    except Exception as e:
        print(f"[MODEL] WARNING - Model load failed: {e}", flush=True)


def _load_joblib():
    global joblib_model, loaded_joblib_path, _joblib_ready
    try:
        import joblib as jl
        for p in JOBLIB_MODEL_PATHS:
            if os.path.exists(p):
                joblib_model = jl.load(p)
                loaded_joblib_path = p
                _joblib_ready = True
                print(f"[MODEL] Joblib loaded: {p}", flush=True)
                gc.collect()
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
    """Load all models once. Thread-safe — only one thread loads at a time."""
    global _models_loaded
    if _models_loaded:
        return True
    with _models_lock:
        if _models_loaded:
            return True
        print("[STARTUP] Loading models...", flush=True)
        try:
            # Load joblib FIRST (tiny, instant) so text queries work immediately
            _load_joblib()
            # Load YOLO next (moderate, ~30s)
            _load_yolo()
            gc.collect()  # Free memory before heavy TF load
            # Load Keras LAST (heaviest, can take 2-5 min on free tier)
            _load_keras()
            gc.collect()
            _models_loaded = True
            print("[STARTUP] All models loaded.", flush=True)
            return True
        except Exception as e:
            print(f"[STARTUP] Model loading failed: {e}", flush=True)
            # Mark as loaded even if Keras failed — partial functionality is better than none
            _models_loaded = True
            return False


# ─── Load JSON artefacts at import time (fast, no heavy deps) ─────────────────
_load_json_artefacts()

# ─── Load models in a BACKGROUND THREAD so gunicorn can bind the port immediately

def _background_model_load():
    """Load models in background so the port binds first (critical for Render)."""
    print("[STARTUP] Background model loading started...", flush=True)
    _ensure_models_loaded()
    print("[STARTUP] Background model loading complete.", flush=True)

_loader_thread = None

@app.before_request
def start_background_thread_if_needed():
    global _loader_thread
    if not _models_loaded and (_loader_thread is None or not _loader_thread.is_alive()):
        _loader_thread = threading.Thread(target=_background_model_load, daemon=True)
        _loader_thread.start()

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
    # Run YOLO with minimal output for speed
    results = yolo_model.predict(image_path, verbose=False)
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
    # Match user's exact verification script
    is_maize = "maize" in name.lower()
    return is_maize, name, conf


def classify_disease(image_path):
    """Stage 2: Keras/TFLite - returns (label_name, condition_name, confidence)."""
    global keras_model
    if keras_model is None:
        _load_keras()
    if keras_model is None:
        raise RuntimeError("Disease model not available")
    
    import numpy as np
    from PIL import Image

    target = tuple(keras_input_shape[1:3]) if keras_input_shape and len(keras_input_shape) > 2 else (224, 224)
    img = Image.open(image_path).convert('RGB')
    
    # Resize the image using BILINEAR resampling (compatible with older and newer Pillow versions)
    resample_filter = getattr(Image, 'Resampling', Image).BILINEAR
    img = img.resize(target, resample_filter)
    
    x = np.array(img, dtype=np.float32)
    x = np.expand_dims(x, axis=0) / 255.0
    
    if hasattr(keras_model, 'invoke'):
        # TFLite inference
        input_details = keras_model.get_input_details()
        output_details = keras_model.get_output_details()
        keras_model.set_tensor(input_details[0]['index'], x.astype(np.float32))
        keras_model.invoke()
        preds = keras_model.get_tensor(output_details[0]['index'])
    else:
        # Standard Keras inference
        preds = keras_model.predict(x, verbose=0, batch_size=1)
        
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
    if not _yolo_ready or not _keras_ready:
        return api_response(False, http_status=503, error='Image models are still loading. Please wait a moment and try again.')
    print(f"\n[PREDICT] {filename}  source={input_source}", flush=True)

    # Stage 1: YOLO maize / non-maize
    try:
        is_maize, yolo_cls, yolo_conf = classify_maize(save_path)
    except Exception as e:
        print(f"  YOLO error: {e}", flush=True)
        return api_response(False, http_status=500, error=f"Image check failed: {e}")

    print(f"  YOLO: {yolo_cls} ({yolo_conf:.3f})  is_maize={is_maize}", flush=True)

    if not is_maize:
        report = build_non_maize_report()
        return api_response(
            False, http_status=400, type="image", input_source=input_source,
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
        return api_response(False, http_status=500, error=f"Disease prediction failed: {e}")

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
    Warmup endpoint — returns model loading status instantly (never blocks).
    Render / uptime monitors can ping this to check readiness.
    """
    return api_response(
        True, status="warm" if _models_loaded else "loading",
        models_ready=_models_loaded,
        yolo_loaded=yolo_model is not None,
        keras_loaded=keras_model is not None,
        disease_classes=len(labels_list),
        text_intents=len(intent_labels),
    )


@app.route("/api/status", methods=["GET"])
def api_status():
    """Lightweight status check — frontend polls this to know when models are ready."""
    return api_response(True,
        models_ready=_models_loaded,
        yolo_ready=_yolo_ready,
        keras_ready=_keras_ready,
        joblib_ready=_joblib_ready,
    )


@app.route("/api/model-info", methods=["GET"])
def api_model_info():
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
        return api_response(False, http_status=400, error="Invalid file type")

    # Base64 JSON
    image_data = payload.get("image_base64") or payload.get("image")
    if image_data and isinstance(image_data, str):
        try:
            save_path, filename = _decode_image_data(image_data)
            return handle_image_prediction(save_path, filename, "json-base64")
        except Exception as e:
            return api_response(False, http_status=400, error=f"Image decode failed: {e}")

    # Text question
    question = (payload.get("question") or payload.get("message") or request.form.get("question") or "").strip()
    if question:
        response_text = "Sorry, I cannot assist you."
        if joblib_model is None:
            response_text = "Model not loaded well. Sorry, I cannot assist you."
        
        label = None
        print(f"\n[TEXT] {question}", flush=True)
        
        # 1. First, check if they are asking about a specific disease
        q_lower = question.lower()
        disease_keywords = {
            "rust": "common_rust", "gray leaf": "gray_leaf_spot", 
            "blight": "leaf_blight", "mildew": "downy_mildew", 
            "streak virus": "maize_streak_virus", "msv": "maize_streak_virus",
            "lethal necrosis": "maize_lethal_necrosis", "mln": "maize_lethal_necrosis"
        }
        for kw, key in disease_keywords.items():
            if kw in q_lower and key in knowledge_base:
                info = knowledge_base[key]
                label = key
                response_text = f"{info.get('description', '')} Recommended management: {', '.join(info.get('management', []))}."
                break
                
        # 2. If no direct disease match, try the NLP model
        if label is None:
            if joblib_model is not None:
                try:
                    pred = joblib_model.predict([q_lower])
                    label = str(pred[0]).strip().lower()
                    raw = response_map.get(label)
                    
                    # Some inputs might predict "smalltalk" but be actual questions
                    # Fallback to response_map scan if we didn't get a good match
                    if raw is None:
                        for k in response_map:
                            if k.lower() == label:
                                raw = response_map[k]
                                label = k
                                break
                    if raw is not None:
                        response_text = raw[0] if isinstance(raw, list) and raw else str(raw)
                except Exception as e:
                    print(f"  Joblib error: {e}", flush=True)

            # 3. Final fallback: keyword scan in response_map
            if response_text == "Sorry, I cannot assist you." or response_text == "Model not loaded well. Sorry, I cannot assist you." or label == "smalltalk":
                for k in response_map:
                    if k.lower() in q_lower:
                        label, raw = k, response_map[k]
                        response_text = raw[0] if isinstance(raw, list) and raw else str(raw)
                        break

        print(f"  Response: {response_text[:80]}...\n", flush=True)
        return api_response(True, type="text", label=label, response=response_text, model_path=loaded_joblib_path)

    return api_response(False, http_status=400, error="No input received (provide question or image)")


@app.route("/api/chat", methods=["POST"])
def api_chat():
    data = _get_payload()
    message = (data.get("message") or data.get("question") or "").strip()
    if not message:
        return api_response(False, http_status=400, error="empty message")
    for k, v in response_map.items():
        if k.lower() in message.lower():
            return api_response(True, response=v, source="local-response-map", label=k)
    return api_response(
        True,
        response="I am the Maize AI Assistant! I can help you with questions about maize diseases and treatments. Please feel free to ask or upload an image for diagnosis.",
        source="fallback",
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port, threaded=True, use_reloader=False)
