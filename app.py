import base64
import os
import json
import uuid
from datetime import datetime
import warnings
from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename

try:
    import cv2
except Exception:
    cv2 = None

try:
    from dotenv import load_dotenv
except Exception:
    def load_dotenv(*args, **kwargs):
        return False

load_dotenv()

# Reduce noisy sklearn model-version warnings (non-fatal for current pipeline)
try:
    from sklearn.exceptions import InconsistentVersionWarning
    warnings.filterwarnings('ignore', category=InconsistentVersionWarning)
except Exception:
    pass

UPLOAD_FOLDER = 'storage/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
API_VERSION = 'v1'
FRONTEND_ORIGINS = [o.strip() for o in os.getenv('FRONTEND_ORIGINS', '*').split(',') if o.strip()]


@app.after_request
def add_cors_headers(response):
    origin = request.headers.get('Origin')
    if '*' in FRONTEND_ORIGINS:
        response.headers['Access-Control-Allow-Origin'] = '*'
    elif origin and origin in FRONTEND_ORIGINS:
        response.headers['Access-Control-Allow-Origin'] = origin
        response.headers['Vary'] = 'Origin'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
    response.headers['Access-Control-Expose-Headers'] = 'Content-Type'
    return response


def api_response(ok, status=200, **payload):
    data = {
        'ok': ok,
        'api_version': API_VERSION,
        'timestamp': datetime.utcnow().isoformat() + 'Z',
        **payload,
    }
    response = jsonify(data)
    response.status_code = status
    return response


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# ─── Model paths ───────────────────────────────────────────────────────────────
JOBLIB_MODEL_PATHS = [
    'askinggreetingmodel.joblib'
]
RESPONSE_MAP_PATHS = [
    'askinggreetingmodel_response_map.json',
    'askingmodelmaize_response_map_v1.0.json',
    'askingmodelmaize_response_map_v1.json',
    'askingmodelmaize_response_map.json'
]
INTENT_LABELS_PATHS = [
    'askinggreetingmodel_labels.json',
    'askingmodelmaize_labels_v1.0.json',
    'askingmodelmaize_labels_v1.json',
    'askingmodelmaize_labels.json'
]
DISEASE_LABELS_PATH = 'disease_labels.json'
DISEASE_KNOWLEDGE_BASE_PATHS = [
    'disease_knowledge_base_v1.0.json',
    'disease_knowledge_base.json',
]

# YOLO maize / non-maize model
YOLO_MODEL_PATH = 'best.pt'

# Keras disease classification model
KERAS_MODEL_PATHS = [
    'maizediseaseprogression.keras'
]

# ─── Global state ──────────────────────────────────────────────────────────────
joblib_model = None
response_map = {}
intent_labels = []
labels_list = []
knowledge_base = {}
yolo_model = None
keras_model = None
keras_input_shape = None

loaded_joblib_model_path = None
loaded_response_map_path = None
loaded_intent_labels_path = None
loaded_keras_model_path = None
loaded_yolo_model_path = None


# ─── Helpers ───────────────────────────────────────────────────────────────────
def normalize_disease_name(label_name):
    mapping = {
        'Healthy_maize': 'Healthy Maize',
        'Leaf_Blight_disease': 'Leaf Blight Disease',
        'Common_Rust_disease': 'Common Rust Disease',
        'Gray_Leaf_Spot_disease': 'Gray Leaf Spot Disease',
        'Downy_Mildew_disease': 'Downy Mildew Disease',
        'Maize_Lethal_Necrosis_disease': 'Maize Lethal Necrosis Disease',
        'Maize_Streak_Virus_disease': 'Maize Streak Virus Disease',
    }
    return mapping.get(label_name, str(label_name).replace('_', ' ').strip())


def knowledge_key_from_label(label_name):
    key = str(label_name).strip().lower().replace(' ', '_').replace('-', '_')
    if key.endswith('_disease'):
        key = key[:-8]
    return key


def _to_list(value):
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [str(value)]


def load_knowledge_base():
    for kb_path in DISEASE_KNOWLEDGE_BASE_PATHS:
        if os.path.exists(kb_path):
            try:
                with open(kb_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    print('Loaded disease knowledge base:', kb_path, f'({len(data)} entries)')
                    return data
            except Exception:
                pass
    return {}


knowledge_base = load_knowledge_base()


def get_disease_entry(label_name):
    key = knowledge_key_from_label(label_name)
    if key in knowledge_base:
        return key, knowledge_base[key]

    aliases = {
        'healthy': 'healthy_maize',
        'common_rust_disease': 'common_rust',
        'gray_leaf_spot_disease': 'gray_leaf_spot',
        'leaf_blight_disease': 'leaf_blight',
        'downy_mildew_disease': 'downy_mildew',
        'maize_streak_virus_disease': 'maize_streak_virus',
        'maize_lethal_necrosis_disease': 'maize_lethal_necrosis',
    }
    alias_key = aliases.get(key)
    if alias_key and alias_key in knowledge_base:
        return alias_key, knowledge_base[alias_key]

    return key, None


def _format_bullet_list(items):
    return '\n'.join([f'- {item}' for item in items])


def _bullet_text(items, fallback='- Not specified'):
    items = [str(item) for item in items if str(item).strip()]
    return _format_bullet_list(items) if items else fallback


def build_non_maize_report():
    return (
        'NOT MAIZE\n\n'
        'This image does not contain a maize leaf.\n\n'
        'Please upload a clear image of a maize leaf for disease detection.\n\n'
        'Tips:\n'
        '- Use close-up leaf images\n'
        '- Avoid people, animals, vehicles, or background clutter\n'
        '- Ensure good lighting'
    )


def build_maize_health_report(label_name):
    condition = normalize_disease_name(label_name)
    _, entry = get_disease_entry(label_name)

    if entry:
        stage_text = entry.get('stage', 'Predicted by model')

        causes = []
        if entry.get('cause'):
            causes.append(entry['cause'])
        if entry.get('favorable_conditions'):
            causes.append(entry['favorable_conditions'])

        management = _to_list(entry.get('management'))
        prevention = _to_list(entry.get('prevention'))
        description = entry.get('description', 'No description available.')
        progression_time = entry.get('progression_time', 'Not specified')
        yield_risk = entry.get('yield_risk', 'Not specified')
        urgency = entry.get('urgency', 'Not specified')

        return (
            'MAIZE HEALTH ANALYSIS REPORT\n'
            '---------------------------\n\n'
            f'Disease:\n{condition}\n\n'
            f'Stage:\n{stage_text}\n\n'
            'Description:\n'
            f'{description}\n\n'
            'Possible Causes:\n'
            f'{_bullet_text(causes)}\n\n'
            'Recommended Actions:\n'
            f'{_bullet_text(management)}\n\n'
            'Prevention Tips:\n'
            f'{_bullet_text(prevention)}\n\n'
            f'Urgency Level:\n{urgency}\n\n'
            f'Progression Time:\n{progression_time}'
        )

    return (
        'MAIZE HEALTH ANALYSIS REPORT\n'
        '---------------------------\n\n'
        f'Disease:\n{condition}\n\n'
        'Stage:\nPredicted by model\n\n'
        'Description:\n'
        'Detailed disease guidance is not available for this label.\n\n'
        'Possible Causes:\n- Not specified\n\n'
        'Recommended Actions:\n'
        '- Review the prediction in the UI\n'
        '- Inspect the plant visually for confirmation\n'
        '- Use local agronomy guidance if needed\n\n'
        'Prevention Tips:\n'
        '- Keep using clean seed and regular monitoring\n'
        '- Maintain good field hygiene\n\n'
        'Urgency Level:\nCheck needed\n\n'
        'Progression Time:\nNot specified'
    )


def get_disease_stage(label_name):
    _, entry = get_disease_entry(label_name)
    if entry:
        stage_text = entry.get('stage')
        if stage_text:
            return stage_text
    return 'Predicted by model'


def _decode_image_data(image_data):
    if not image_data:
        raise ValueError('missing image data')

    if isinstance(image_data, str) and image_data.startswith('data:') and ',' in image_data:
        image_data = image_data.split(',', 1)[1]

    image_bytes = base64.b64decode(image_data)
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    filename = secure_filename(f'upload_{uuid.uuid4().hex}.png')
    save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    with open(save_path, 'wb') as f:
        f.write(image_bytes)
    return save_path, filename


def _get_payload():
    if request.is_json:
        return request.get_json(silent=True) or {}
    return request.form.to_dict(flat=True)


# ─── Model loaders ─────────────────────────────────────────────────────────────
def get_yolo_model():
    """Load the YOLOv8 maize / non-maize classifier (best.pt)."""
    global yolo_model, loaded_yolo_model_path
    if yolo_model is None:
        try:
            from ultralytics import YOLO
            if os.path.exists(YOLO_MODEL_PATH):
                yolo_model = YOLO(YOLO_MODEL_PATH)
                loaded_yolo_model_path = YOLO_MODEL_PATH
                print(f'Loaded YOLO model: {YOLO_MODEL_PATH}  classes: {yolo_model.names}')
            else:
                print(f'YOLO model file not found: {YOLO_MODEL_PATH}')
        except ImportError:
            print('ultralytics not installed — run: pip install ultralytics')
        except Exception as e:
            print('YOLO load error:', e)
    return yolo_model


def get_keras_model():
    """Load the Keras disease classifier."""
    global keras_model, keras_input_shape, loaded_keras_model_path
    if keras_model is None:
        try:
            import tensorflow as tf
            tf.get_logger().setLevel('ERROR')
            from tensorflow.keras.models import load_model
            for model_path in KERAS_MODEL_PATHS:
                if os.path.exists(model_path):
                    keras_model = load_model(model_path)
                    keras_input_shape = getattr(keras_model, 'input_shape', (None, 224, 224, 3))
                    loaded_keras_model_path = model_path
                    print(f'Loaded Keras model: {model_path}')
                    break
        except Exception as e:
            print('Keras load error:', e)
    return keras_model


def get_joblib_model():
    global joblib_model, loaded_joblib_model_path
    if joblib_model is None:
        try:
            import joblib
            for model_path in JOBLIB_MODEL_PATHS:
                if os.path.exists(model_path):
                    joblib_model = joblib.load(model_path)
                    loaded_joblib_model_path = model_path
                    print('Loaded joblib text model:', model_path)
                    break
        except Exception as e:
            print('Joblib load error:', e)
    return joblib_model


# ─── Load static artefacts at startup ──────────────────────────────────────────
# Response map
for response_path in RESPONSE_MAP_PATHS:
    if os.path.exists(response_path):
        try:
            with open(response_path, 'r', encoding='utf-8') as f:
                raw_response_map = json.load(f)
                for key, value in raw_response_map.items():
                    if isinstance(value, str) and ',' in value:
                        response_map[key] = [v.strip() for v in value.split(',')]
                    else:
                        response_map[key] = value
                print('Loaded response map:', response_path, f'({len(response_map)} categories)')
                loaded_response_map_path = response_path
                break
        except Exception:
            pass

# Intent labels
for labels_path in INTENT_LABELS_PATHS:
    if os.path.exists(labels_path):
        try:
            with open(labels_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, dict) and 'intent_list' in data:
                    intent_labels = data['intent_list']
                elif isinstance(data, list):
                    intent_labels = data
                print('Loaded intent labels:', labels_path, f'({len(intent_labels)} intents)')
                loaded_intent_labels_path = labels_path
                break
        except Exception:
            pass

# Disease labels
if os.path.exists(DISEASE_LABELS_PATH):
    try:
        with open(DISEASE_LABELS_PATH, 'r', encoding='utf-8') as f:
            labels_list = json.load(f)
            print('Loaded disease labels:', DISEASE_LABELS_PATH, f'({len(labels_list)} diseases)')
    except Exception:
        pass
else:
    labels_list = [
        'Healthy_maize', 'Common_Rust_disease', 'Gray_Leaf_Spot_disease',
        'Leaf_Blight_disease', 'Downy_Mildew_disease', 'Maize_Streak_Virus_disease',
        'Maize_Lethal_Necrosis_disease'
    ]
    print(f'Using default disease labels ({len(labels_list)} classes)')


# ─── Routes ────────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api', methods=['GET'])
def api_root():
    return api_response(True, message='Maize Disease Progression API is running', endpoints=[
        '/api/health',
        '/api/model-info',
        '/api/predict',
        '/api/chat',
    ])


@app.route('/api/health', methods=['GET'])
def api_health():
    return api_response(True, status=200, service='maize-disease-api', status_message='healthy')


@app.route('/api/model-info', methods=['GET'])
def api_model_info():
    ym = get_yolo_model()
    return api_response(
        True,
        text_model=loaded_joblib_model_path,
        response_map=loaded_response_map_path,
        intent_labels=loaded_intent_labels_path,
        yolo_model=loaded_yolo_model_path,
        yolo_classes=ym.names if ym else None,
        disease_model=loaded_keras_model_path,
        disease_input_shape=keras_input_shape,
        disease_classes=len(labels_list),
        text_intents=len(intent_labels),
        supports={
            'text_question': True,
            'json_chat': True,
            'multipart_image_upload': True,
            'json_base64_image': True,
        },
    )


# ─── Core prediction helpers ───────────────────────────────────────────────────
def _classify_with_yolo(image_path):
    """
    Run YOLOv8 on the image.
    Returns (is_maize: bool, class_name: str, confidence: float).
    """
    model = get_yolo_model()
    if model is None:
        raise RuntimeError('YOLO model not available')

    results = model(image_path, verbose=False)
    r = results[0]

    # r.probs is available for classification models
    if hasattr(r, 'probs') and r.probs is not None:
        top_idx = int(r.probs.top1)
        top_conf = float(r.probs.top1conf)
        class_name = model.names[top_idx]
    elif hasattr(r, 'boxes') and r.boxes is not None and len(r.boxes) > 0:
        # detection model fallback — pick highest-confidence box
        best = max(r.boxes, key=lambda b: float(b.conf))
        top_idx = int(best.cls)
        top_conf = float(best.conf)
        class_name = model.names[top_idx]
    else:
        raise RuntimeError('YOLO returned no predictions')

    is_maize = class_name.lower().replace('-', '_').replace(' ', '_') == 'maize'
    return is_maize, class_name, top_conf


def _classify_disease_with_keras(image_path):
    """
    Run the Keras disease classifier on a maize image.
    Returns (label_name, condition_name, confidence).
    """
    k_model = get_keras_model()
    if k_model is None:
        raise RuntimeError('Keras disease model not available')

    from tensorflow.keras.utils import load_img, img_to_array
    import numpy as np

    if keras_input_shape:
        target_size = tuple(keras_input_shape[1:3]) if len(keras_input_shape) > 2 else (224, 224)
    else:
        target_size = (224, 224)

    img = load_img(image_path, target_size=target_size)
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = x / 255.0

    preds = k_model.predict(x, verbose=0)
    preds_array = preds[0] if len(preds.shape) > 1 else preds
    label_index = int(np.argmax(preds_array))
    confidence = float(preds_array[label_index])

    if labels_list and len(labels_list) > 0 and label_index < len(labels_list):
        label_name = str(labels_list[label_index])
    else:
        raise RuntimeError('Prediction label is outside supported maize disease classes')

    condition_name = normalize_disease_name(label_name)
    return label_name, condition_name, confidence


def _handle_image_prediction(save_path, filename, input_source):
    """Shared image prediction pipeline for both multipart and base64 uploads."""
    print(f"\n[IMAGE PREDICTION] File: {filename}  source: {input_source}")

    # ── Stage 1: YOLO maize / non-maize check ──────────────────────────────
    try:
        is_maize, yolo_class, yolo_conf = _classify_with_yolo(save_path)
    except RuntimeError as e:
        return api_response(False, status=503, error=f'YOLO model error: {e}')
    except Exception as e:
        print(f'  YOLO prediction failed: {e}')
        import traceback; traceback.print_exc()
        return api_response(False, status=500, error=f'Image check failed: {e}')

    print(f'  YOLO result: class={yolo_class}  conf={yolo_conf:.4f}  is_maize={is_maize}')

    if not is_maize:
        report = build_non_maize_report()
        return api_response(
            False,
            status=400,
            type='image',
            input_source=input_source,
            valid_image=False,
            is_maize=False,
            yolo_class=yolo_class,
            yolo_confidence=round(yolo_conf, 4),
            error='This is not a maize image. Please upload a maize leaf image.',
            answer=report,
            report=report,
            file_name=filename,
        )

    # ── Stage 2: Keras disease classification ──────────────────────────────
    try:
        label_name, condition_name, disease_conf = _classify_disease_with_keras(save_path)
    except RuntimeError as e:
        return api_response(False, status=503, error=f'Disease model error: {e}')
    except Exception as e:
        print(f'  Disease prediction failed: {e}')
        import traceback; traceback.print_exc()
        return api_response(False, status=500, error=f'Disease prediction failed: {e}')

    stage = get_disease_stage(label_name)
    report = build_maize_health_report(label_name)

    print(f'  Disease: {condition_name}  conf={disease_conf:.4f}  stage={stage}\n')

    return api_response(
        True,
        type='image',
        input_source=input_source,
        valid_image=True,
        is_maize=True,
        yolo_class=yolo_class,
        yolo_confidence=round(yolo_conf, 4),
        label=label_name,
        condition=condition_name,
        stage=stage,
        severity=stage,
        disease_confidence=round(disease_conf, 4),
        answer=report,
        report=report,
        yolo_model_path=loaded_yolo_model_path,
        disease_model_path=loaded_keras_model_path,
        file_name=filename,
    )


# ─── /api/predict ──────────────────────────────────────────────────────────────
@app.route('/api/predict', methods=['POST'])
def api_predict():
    payload = _get_payload()

    # ── Multipart file upload ──────────────────────────────────────────────
    if 'image' in request.files:
        file = request.files['image']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
            save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(save_path)
            return _handle_image_prediction(save_path, filename, 'multipart/form-data')
        return api_response(False, status=400, error='Invalid file type')

    # ── JSON base64 image ──────────────────────────────────────────────────
    image_data = payload.get('image_base64') or payload.get('image')
    if image_data and isinstance(image_data, str):
        try:
            save_path, filename = _decode_image_data(image_data)
            return _handle_image_prediction(save_path, filename, 'json-base64')
        except Exception as e:
            return api_response(False, status=400, error=f'Image decode failed: {e}')

    # ── Text question ──────────────────────────────────────────────────────
    question = (payload.get('question') or payload.get('message') or request.form.get('question') or '').strip()
    if question:
        response_text = "Sorry, I don't have an answer for that yet."
        label = None

        print(f"\n[TEXT PREDICTION] Question: {question}")

        j_model = get_joblib_model()
        if j_model is not None:
            try:
                cleaned_question = question.lower().strip()
                pred = j_model.predict([cleaned_question])
                label = str(pred[0]).strip().lower()
                print(f"  Model predicted label: {label}")

                raw_response = response_map.get(label, None)
                if raw_response is None:
                    for key in response_map.keys():
                        if key.lower() == label:
                            raw_response = response_map[key]
                            label = key
                            break

                if raw_response is None:
                    raw_response = response_text

                if isinstance(raw_response, list):
                    response_text = raw_response[0] if len(raw_response) > 0 else response_text
                else:
                    response_text = str(raw_response) if raw_response else response_text

            except Exception as e:
                print(f'  Text model prediction error: {e}')
                for k in response_map.keys():
                    if k.lower() in question.lower():
                        label = k
                        raw_response = response_map.get(k, response_text)
                        if isinstance(raw_response, list):
                            response_text = raw_response[0] if raw_response else response_text
                        else:
                            response_text = raw_response
                        break
        else:
            print("  No joblib model available, using keyword matching")
            for k in response_map.keys():
                if k.lower() in question.lower():
                    label = k
                    raw_response = response_map.get(k, response_text)
                    if isinstance(raw_response, list):
                        response_text = raw_response[0] if raw_response else response_text
                    else:
                        response_text = raw_response
                    break

        print(f"  Final response: {response_text[:100]}...\n")
        return api_response(True, type='text', input_source='json-or-form', label=label, response=response_text, model_path=loaded_joblib_model_path)

    return api_response(False, status=400, error='No input received (provide question or image)')


# ─── /api/chat ─────────────────────────────────────────────────────────────────
@app.route('/api/chat', methods=['POST'])
def api_chat():
    data = _get_payload()
    message = (data.get('message') or data.get('question') or '').strip()
    if not message:
        return api_response(False, status=400, error='empty message')

    for k, v in response_map.items():
        if k.lower() in message.lower():
            return api_response(True, response=v, source='local-response-map', label=k)

    return api_response(
        True,
        response="I am the Maize AI Assistant! I can help you with questions about maize diseases and treatments. Please feel free to ask or upload an image for diagnosis.",
        source='fallback',
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
