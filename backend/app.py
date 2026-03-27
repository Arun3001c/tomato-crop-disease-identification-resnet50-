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

# 🔧 Patch Dense.from_config to ignore quantization_config
from tensorflow.keras.layers import Dense
_original_dense_from_config = Dense.from_config

@classmethod
def _dense_from_config(cls, config):
    config.pop('quantization_config', None)
    return _original_dense_from_config(config)

Dense.from_config = _dense_from_config

app = Flask(__name__)
CORS(app)

# 📦 Load model
model_path = os.path.join(os.path.dirname(__file__), 'model', 'resnet50_plantvillage_model.keras')
if not os.path.exists(model_path):
    model_path = os.path.join(os.path.dirname(__file__), 'model', 'resnet50_plantvillage_model.h5')

model = load_model(model_path)

# 🏷️ Class labels
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

# 🖼️ Image preprocessing
def preprocess_image(image):
    if image.mode != 'RGB':
        image = image.convert('RGB')

    image = image.resize((224, 224))
    image = img_to_array(image)
    image = resnet.preprocess_input(image)
    image = np.expand_dims(image, axis=0)

    return image

# 📂 File validation
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'jpg', 'jpeg', 'png'}

# 🌿 Leaf detection heuristic
def is_leaf_like(image):
    rgb_image = image.convert('RGB')
    arr = np.array(rgb_image)

    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]

    # 🌿 Green detection
    green_mask = (g > r * 1.1) & (g > b * 1.1) & (g > 70)
    green_ratio = float(np.mean(green_mask))

    # 🧠 Texture detection (variance)
    gray = np.mean(arr, axis=2)
    texture_variance = np.var(gray)

    # 🎯 Combined condition
    is_leaf = (green_ratio > 0.25) and (texture_variance > 500)

    return is_leaf, green_ratio
# def is_leaf_like(image):
#     rgb_image = image.convert('RGB')
#     arr = np.array(rgb_image)

#     r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]

#     green_mask = (g > r * 1.1) & (g > b * 1.1) & (g > 70)
#     green_ratio = float(np.mean(green_mask))

#     return green_ratio >= 0.25, green_ratio   # 🔥 stricter threshold

# 🚀 Prediction API
@app.route('/predict', methods=['POST'])
def predict():

    # ❌ No file
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']

    # ❌ No filename
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    # ❌ Invalid type
    if not allowed_file(file.filename):
        return jsonify({'error': 'Only JPG, JPEG, PNG allowed'}), 400

    try:
        # 🖼️ Read image
        image = Image.open(io.BytesIO(file.read()))

        # 🌿 Check if it's leaf-like FIRST
        leaf_detected, leaf_score = is_leaf_like(image)

        if not leaf_detected:
            return jsonify({
                'prediction': 'Not a plant leaf image',
                'confidence': 0,
                'leaf_score': leaf_score
            })

        # 🔄 Preprocess
        processed_image = preprocess_image(image)

        # 🤖 Predict
        prediction = model.predict(processed_image)
        probabilities = tf.nn.softmax(prediction[0]).numpy()

        predicted_class = int(np.argmax(probabilities))
        confidence = float(probabilities[predicted_class])
        class_label = CLASS_NAMES[predicted_class]

        # 🎯 Confidence threshold
        THRESHOLD = 0.7

        if confidence < THRESHOLD:
            return jsonify({
                'prediction': 'Uncertain prediction (low confidence)',
                'confidence': confidence,
                'leaf_score': leaf_score
            })

        # ✅ Final valid prediction
        return jsonify({
            'prediction': class_label,
            'confidence': confidence,
            'leaf_score': leaf_score
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ▶️ Run server
if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5000)