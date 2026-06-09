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


# Try loading a joblib text model and response maps (with version support)
# Try versioned files first (_v1.0, _v1, etc.), then fall back to legacy names
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
DISEASE_LABELS_PATH = 'disease_labels.json'  # For image classification
DISEASE_KNOWLEDGE_BASE_PATHS = [
    'disease_knowledge_base_v1.0.json',
    'disease_knowledge_base.json',
]

joblib_model = None
response_map = {}
intent_labels = []  # For text
labels_list = []    # For images (diseases)
knowledge_base = {}
loaded_joblib_model_path = None
loaded_response_map_path = None
loaded_intent_labels_path = None
loaded_keras_model_path = None
loaded_yolo_model_path = None

YOLO_MODEL_PATHS = [
    'yolov8n.pt',
]

yolo_model = None


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


def build_invalid_image_report(detected_objects):
    objects_text = ', '.join(detected_objects) if detected_objects else 'No clear maize leaf detected'
    return (
        'INVALID IMAGE DETECTED\n\n'
        'This system only works with maize leaf images.\n\n'
        f'Detected Objects: {objects_text}\n\n'
        'Please upload a clear maize leaf image only.\n\n'
        'Tips:\n'
        '- Use close-up leaf images\n'
        '- Avoid people, animals, vehicles, or background clutter\n'
        '- Ensure good lighting'
    )

def _format_bullet_list(items):
    return '\n'.join([f'- {item}' for item in items])


def _bullet_text(items, fallback='- Not specified'):
    items = [str(item) for item in items if str(item).strip()]
    return _format_bullet_list(items) if items else fallback


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


def is_likely_maize_leaf(image_path):
    if cv2 is None:
        try:
            from PIL import Image
            import numpy as np

            img = Image.open(image_path).convert('RGB')
            width, height = img.size
            if width < 120 or height < 120:
                return False

            arr = np.array(img)
            green = (arr[:, :, 1] > arr[:, :, 0] + 15) & (arr[:, :, 1] > arr[:, :, 2] + 15)
            green_ratio = float(green.mean())
            return green_ratio >= 0.10
        except Exception:
            return False

    img = cv2.imread(image_path)
    if img is None:
        return False

    height, width = img.shape[:2]
    if height < 120 or width < 120:
        return False

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_green = (25, 35, 25)
    upper_green = (95, 255, 255)
    mask = cv2.inRange(hsv, lower_green, upper_green)
    green_ratio = float(cv2.countNonZero(mask)) / float(height * width)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    edge_ratio = float(cv2.countNonZero(edges)) / float(height * width)

    return green_ratio >= 0.10 and edge_ratio >= 0.01


def validate_maize_image_with_yolo(image_path):
    y_model = get_yolo_model()
    if y_model is None:
        return {
            'valid': False,
            'detected_objects': ['Validation model unavailable'],
        }

    try:
        results = y_model.predict(source=image_path, conf=0.25, verbose=False)
        detected_objects = []
        for result in results:
            if result.boxes is None:
                continue
            for cls_idx in result.boxes.cls.tolist():
                obj_name = result.names.get(int(cls_idx), f'class_{int(cls_idx)}')
                if obj_name not in detected_objects:
                    detected_objects.append(obj_name)

        # Strict rule requested: if YOLO detects any known object, reject.
        if len(detected_objects) > 0:
            return {
                'valid': False,
                'detected_objects': detected_objects,
            }

        # If YOLO detects nothing, still ensure image looks like a clear maize leaf.
        if not is_likely_maize_leaf(image_path):
            return {
                'valid': False,
                'detected_objects': ['No clear maize leaf structure'],
            }

        return {
            'valid': True,
            'detected_objects': [],
        }
    except Exception:
        return {
            'valid': False,
            'detected_objects': ['Validation failed'],
        }

def build_farmer_image_report(label_name, confidence):
    pretty_name = label_name.replace('_', ' ').replace('disease', 'disease').strip()
    _, entry = get_disease_entry(label_name)
    if entry:
        stage_text = get_disease_stage(label_name)
        description = entry.get('description', 'No description available.')
        causes = []
        if entry.get('cause'):
            causes.append(entry['cause'])
        if entry.get('favorable_conditions'):
            causes.append(entry['favorable_conditions'])
        management = _to_list(entry.get('management'))
        prevention = _to_list(entry.get('prevention'))
        urgency = entry.get('urgency', 'Not specified')

        return (
            f'Disease:\n{pretty_name}\n\n'
            f'Stage:\n{stage_text}\n\n'
            'Description:\n'
            f'{description}\n\n'
            'Possible Causes:\n'
            f'{_bullet_text(causes)}\n\n'
            'Recommended Actions:\n'
            f'{_bullet_text(management)}\n\n'
            'Prevention Tips:\n'
            f'{_bullet_text(prevention)}\n\n'
            f'Urgency Level:\n{urgency}'
        )
    return (
        f'Disease:\n{pretty_name}\n\n'
        'Stage:\nPredicted by model\n\n'
        'Description:\nDetailed disease guidance is not available for this label.\n\n'
        'Possible Causes:\n- Not specified\n\n'
        'Recommended Actions:\n- Review the result and confirm with field inspection or local agronomy support\n\n'
        'Prevention Tips:\n- Keep using clean seed and regular monitoring\n\n'
        'Urgency Level:\nCheck needed'
    )

# Load and parse response map for text answers (versioned first)
response_map_loaded = False
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
                response_map_loaded = True
                break
        except Exception as e:
            pass

# Load intent labels for text classification (versioned first)
labels_loaded = False
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
                labels_loaded = True
                break
        except Exception as e:
            pass

# Load disease labels for image classification
if os.path.exists(DISEASE_LABELS_PATH):
    try:
        with open(DISEASE_LABELS_PATH, 'r', encoding='utf-8') as f:
            labels_list = json.load(f)
            print('Loaded disease labels:', DISEASE_LABELS_PATH, f'({len(labels_list)} diseases)')
    except Exception as e:
        pass
else:
    labels_list = [
        'Healthy_maize', 'Common_Rust_disease', 'Gray_Leaf_Spot_disease',
        'Leaf_Blight_disease', 'Downy_Mildew_disease', 'Maize_Streak_Virus_disease',
        'Maize_Lethal_Necrosis_disease'
    ]
    print(f'Using default disease labels ({len(labels_list)} classes)')

keras_model = None
keras_input_shape = None
KERAS_MODEL_PATHS = [
    'maizediseaseprogression.keras'
]

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
            print("Joblib load error:", e)
    return joblib_model

def get_keras_model():
    global keras_model, keras_input_shape, loaded_keras_model_path
    if keras_model is None:
        try:
            import tensorflow as tf
            # Reduce verbose TF logs
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
            print("Keras load error:", e)
    return keras_model

def get_yolo_model():
    global yolo_model, loaded_yolo_model_path
    if yolo_model is None:
        try:
            from ultralytics import YOLO
            for model_path in YOLO_MODEL_PATHS:
                if os.path.exists(model_path):
                    yolo_model = YOLO(model_path)
                    loaded_yolo_model_path = model_path
                    print(f'Loaded YOLO validation model: {model_path}')
                    break
        except Exception as e:
            print("YOLO load error:", e)
    return yolo_model


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
    return api_response(
        True,
        text_model=loaded_joblib_model_path,
        response_map=loaded_response_map_path,
        intent_labels=loaded_intent_labels_path,
        image_model=loaded_keras_model_path,
        validation_model=loaded_yolo_model_path,
        image_input_shape=keras_input_shape,
        image_classes=len(labels_list),
        text_intents=len(intent_labels),
        supports={
            'text_question': True,
            'json_chat': True,
            'multipart_image_upload': True,
            'json_base64_image': True,
        },
    )


@app.route('/api/predict', methods=['POST'])
def api_predict():
    payload = _get_payload()

    # Prefer image classification when an uploaded image is present.
    if 'image' in request.files:
        file = request.files['image']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
            save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(save_path)
            
            print(f"\n[IMAGE PREDICTION] File: {filename}")
            print(f"  Labels loaded: {len(labels_list)} classes")
            print(f"  Model available: {get_keras_model() is not None}")

            # Stage 1: YOLOv8 image validation gate
            validation = validate_maize_image_with_yolo(save_path)
            if not validation['valid']:
                invalid_report = build_invalid_image_report(validation['detected_objects'])
                return api_response(
                    False,
                    status=400,
                    type='image',
                    valid_image=False,
                    error='This is not a maize image. Use maize image only.',
                    answer=invalid_report,
                    report=invalid_report,
                )

            k_model = get_keras_model()
            if k_model is not None:
                try:
                    from tensorflow.keras.preprocessing import image
                    import numpy as np

                    # Load and preprocess image
                    if keras_input_shape:
                        target_size = tuple(keras_input_shape[1:3]) if len(keras_input_shape) > 2 else (224, 224)
                    else:
                        target_size = (224, 224)
                    
                    print(f"  Target size: {target_size}")
                    
                    img = image.load_img(save_path, target_size=target_size)
                    x = image.img_to_array(img)
                    x = np.expand_dims(x, axis=0)
                    x = x / 255.0
                    
                    # Predict
                    preds = k_model.predict(x, verbose=0)
                    print(f"  Predictions shape: {preds.shape}")
                    
                    if preds is None or len(preds) == 0:
                        raise Exception("Empty predictions from model")
                    
                    preds_array = preds[0] if len(preds.shape) > 1 else preds
                    label_index = int(np.argmax(preds_array))
                    
                    print(f"  Top prediction index: {label_index}")
                    print(f"  Labels list length: {len(labels_list)}")

                    # Map index to label
                    if labels_list and len(labels_list) > 0 and label_index < len(labels_list):
                        label_name = str(labels_list[label_index])
                        print(f"  Mapped to label: {label_name}")
                    else:
                        raise Exception('Prediction label is outside supported maize disease classes')

                    condition_name = normalize_disease_name(label_name)
                    stage = get_disease_stage(label_name)
                    report = build_maize_health_report(label_name)

                    print(f"  Condition: {condition_name}\n")

                    return api_response(
                        True,
                        type='image',
                        input_source='multipart/form-data',
                        valid_image=True,
                        label=label_name,
                        condition=condition_name,
                        stage=stage,
                        severity=stage,
                        answer=report,
                        report=report,
                        model_path=loaded_keras_model_path,
                        validation_model_path=loaded_yolo_model_path,
                        file_name=filename,
                    )
                except Exception as e:
                    print(f'  Image model prediction failed: {str(e)}')
                    import traceback
                    traceback.print_exc()
                    return api_response(False, status=500, error=f'Image prediction failed: {str(e)}')

            return api_response(False, status=503, error='No image classification model available on server')

    # JSON base64 image path for external frontends.
    image_data = payload.get('image_base64') or payload.get('image')
    if image_data and isinstance(image_data, str):
        try:
            save_path, filename = _decode_image_data(image_data)
            print(f"\n[IMAGE PREDICTION] File: {filename}")
            print(f"  Labels loaded: {len(labels_list)} classes")
            print(f"  Model available: {get_keras_model() is not None}")

            validation = validate_maize_image_with_yolo(save_path)
            if not validation['valid']:
                invalid_report = build_invalid_image_report(validation['detected_objects'])
                return api_response(
                    False,
                    status=400,
                    type='image',
                    input_source='json-base64',
                    valid_image=False,
                    error='This is not a maize image. Use maize image only.',
                    answer=invalid_report,
                    report=invalid_report,
                    file_name=filename,
                )

            k_model = get_keras_model()
            if k_model is None:
                return api_response(False, status=503, error='No image classification model available on server')

            from tensorflow.keras.preprocessing import image
            import numpy as np

            if keras_input_shape:
                target_size = tuple(keras_input_shape[1:3]) if len(keras_input_shape) > 2 else (224, 224)
            else:
                target_size = (224, 224)

            img = image.load_img(save_path, target_size=target_size)
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = x / 255.0

            preds = k_model.predict(x, verbose=0)
            preds_array = preds[0] if len(preds.shape) > 1 else preds
            label_index = int(np.argmax(preds_array))

            if labels_list and len(labels_list) > 0 and label_index < len(labels_list):
                label_name = str(labels_list[label_index])
            else:
                return api_response(False, status=500, error='Prediction label is outside supported maize disease classes')

            condition_name = normalize_disease_name(label_name)
            stage = get_disease_stage(label_name)
            report = build_maize_health_report(label_name)

            return api_response(
                True,
                type='image',
                input_source='json-base64',
                valid_image=True,
                label=label_name,
                condition=condition_name,
                stage=stage,
                severity=stage,
                answer=report,
                report=report,
                model_path=loaded_keras_model_path,
                validation_model_path=loaded_yolo_model_path,
                file_name=filename,
            )
        except Exception as e:
            print(f'  Image model prediction failed: {str(e)}')
            return api_response(False, status=500, error=f'Image prediction failed: {str(e)}')

    # Text question path
    question = (payload.get('question') or payload.get('message') or request.form.get('question') or '').strip()
    if question:
        response_text = "Sorry, I don't have an answer for that yet."
        label = None

        print(f"\n[TEXT PREDICTION] Question: {question}")

        j_model = get_joblib_model()
        # If we have a joblib model, use it for text prediction
        if j_model is not None:
            try:
                cleaned_question = question.lower().strip()
                pred = j_model.predict([cleaned_question])
                label = str(pred[0]).strip().lower()
                print(f"  Model predicted label: {label}")
                print(f"  Cleaned question: {cleaned_question}")

                # Get response from map using predicted label
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
                    print(f"  Selected response from {len(raw_response)} trained responses")
                else:
                    response_text = str(raw_response) if raw_response else response_text
                    print("  Using trained response")

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
                        print(f"  Fallback keyword match: {k}")
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
                    print(f"  Keyword match found: {k}")
                    break

        print(f"  Final response: {response_text[:100]}...\n")
        return api_response(True, type='text', input_source='json-or-form', label=label, response=response_text, model_path=loaded_joblib_model_path)

    return api_response(False, status=400, error='No input received (provide question or image)')


@app.route('/api/chat', methods=['POST'])
def api_chat():
    data = _get_payload()
    message = (data.get('message') or data.get('question') or '').strip()
    if not message:
        return api_response(False, status=400, error='empty message')

    # Prefer using OpenAI if API key is provided
    openai_key = os.getenv('OPENAI_API_KEY')
    if openai_key:
        try:
            import openai
            openai.api_key = openai_key
            resp = openai.ChatCompletion.create(
                model='gpt-3.5-turbo',
                messages=[{'role':'user','content': message}],
                max_tokens=300,
            )
            text = resp['choices'][0]['message']['content']
            return api_response(True, response=text, source='openai')
        except Exception as e:
            print('OpenAI request failed:', e)

    # Fallback: use local response_map lookup
    for k, v in response_map.items():
        if k.lower() in message.lower():
            return api_response(True, response=v, source='local-response-map', label=k)

    return api_response(True, response="I'm sorry — I can't call ChatGPT right now. Try another question.", source='fallback')


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
