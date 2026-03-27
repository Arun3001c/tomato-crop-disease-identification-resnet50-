from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications import resnet
from PIL import Image
import numpy as np
import io
import os
import requests

# 🔧 Fix for keras config issue
from tensorflow.keras.layers import Dense
_original_dense_from_config = Dense.from_config

@classmethod
def _dense_from_config(cls, config):
    config.pop('quantization_config', None)
    return _original_dense_from_config(config)

Dense.from_config = _dense_from_config

app = Flask(__name__)
CORS(app)

# =========================
# 📦 MODEL CONFIG
# =========================

BASE_DIR = os.path.dirname(__file__)
MODEL_DIR = os.path.join(BASE_DIR, "model")
os.makedirs(MODEL_DIR, exist_ok=True)

MODEL_PATH = os.path.join(MODEL_DIR, "resnet50_plantvillage_model.keras")

MODEL_URL = "https://drive.google.com/uc?export=download&id=1jBRqY6xvdzqbMoWO0OWrNEa6GxtSL3cW"

# =========================
# 📥 DOWNLOAD MODEL
# =========================

def download_model():
    print("📥 Downloading model...")

    session = requests.Session()
    response = session.get(MODEL_URL, stream=True)

    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            params = {'id': MODEL_URL.split('id=')[1], 'confirm': value}
            response = session.get(
                "https://drive.google.com/uc?export=download",
                params=params,
                stream=True
            )

    with open(MODEL_PATH, "wb") as f:
        for chunk in response.iter_content(32768):
            if chunk:
                f.write(chunk)

    print("✅ Model saved at:", MODEL_PATH)

# =========================
# 📦 LOAD MODEL
# =========================

if not os.path.exists(MODEL_PATH):
    download_model()

print("🚀 Loading model...")
model = load_model(MODEL_PATH)

# =========================
# 🏷️ LABELS
# =========================

CLASS_NAMES = [
    'Bacterial Spot',
    'Early Blight',
    'Late Blight',
    'Leaf Mold',
    'Septoria Leaf Spot',
    'Spider Mites',
    'Target Spot',
    'Tomato Yellow Leaf Curl Virus',
    'Tomato Mosaic Virus',
    'Healthy'
]

# =========================
# 🖼️ PREPROCESS
# =========================

def preprocess_image(image):
    if image.mode != 'RGB':
        image = image.convert('RGB')

    image = image.resize((224, 224))
    image = img_to_array(image)
    image = resnet.preprocess_input(image)
    image = np.expand_dims(image, axis=0)

    return image

# =========================
# 📂 VALIDATION
# =========================

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'jpg', 'jpeg', 'png'}

# =========================
# 🌿 LEAF CHECK
# =========================

def is_leaf_like(image):
    arr = np.array(image.convert('RGB'))

    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]

    green_mask = (g > r * 1.1) & (g > b * 1.1) & (g > 70)
    green_ratio = float(np.mean(green_mask))

    gray = np.mean(arr, axis=2)
    texture_variance = np.var(gray)

    is_leaf = (green_ratio > 0.25) and (texture_variance > 500)

    return is_leaf, green_ratio

# =========================
# 🚀 API
# =========================

@app.route('/predict', methods=['POST'])
def predict():

    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if not allowed_file(file.filename):
        return jsonify({'error': 'Only JPG, JPEG, PNG allowed'}), 400

    try:
        image = Image.open(io.BytesIO(file.read()))

        leaf_detected, leaf_score = is_leaf_like(image)

        if not leaf_detected:
            return jsonify({
                'prediction': 'Not a plant leaf image',
                'confidence': 0,
                'leaf_score': leaf_score
            })

        processed_image = preprocess_image(image)

        prediction = model.predict(processed_image)
        probabilities = tf.nn.softmax(prediction[0]).numpy()

        predicted_class = int(np.argmax(probabilities))
        confidence = float(probabilities[predicted_class])
        class_label = CLASS_NAMES[predicted_class]

        if confidence < 0.7:
            return jsonify({
                'prediction': 'Uncertain prediction (low confidence)',
                'confidence': confidence,
                'leaf_score': leaf_score
            })

        return jsonify({
            'prediction': class_label,
            'confidence': confidence,
            'leaf_score': leaf_score
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# =========================
# ▶️ RUN
# =========================

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)