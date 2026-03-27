import os
import tensorflow as tf
import numpy as np
from PIL import Image

# Patch Dense.from_config to ignore quantization_config
from tensorflow.keras.layers import Dense
_original_dense_from_config = Dense.from_config

@classmethod
def _dense_from_config(cls, config):
    config.pop('quantization_config', None)
    return _original_dense_from_config(config)

Dense.from_config = _dense_from_config

# Define class names (same as in training)
class_names = [
    'Tomato___Bacterial_spot',
    'Tomato___Early_blight',
    'Tomato___Late_blight',
    'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy'
]

# Path to model and test image
model_dir = os.path.join(os.path.dirname(__file__), 'model')
model_path = os.path.join(model_dir, 'resnet50_plantvillage_model.keras')
if not os.path.exists(model_path):
    model_path = os.path.join(model_dir, 'resnet50_plantvillage_model.h5')

test_image_path = os.path.join(model_dir, 'test.jpg')

# Load the model
print("Loading model...")
loaded_model = tf.keras.models.load_model(model_path, compile=False)
print("Model loaded successfully!")

# Function to preprocess image
def preprocess_image(image_path, target_size=(224, 224)):
    img = Image.open(image_path).convert('RGB')
    img = img.resize(target_size)
    img_array = np.array(img).astype(np.float32)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.resnet.preprocess_input(img_array)
    return img_array

# Load and preprocess test image
if os.path.exists(test_image_path):
    print("Preprocessing test image...")
    processed_image = preprocess_image(test_image_path)

    # Make prediction
    print("Making prediction...")
    predictions = loaded_model.predict(processed_image, verbose=0)
    probabilities = tf.nn.softmax(predictions, axis=-1).numpy()[0]

    # Get predicted class
    predicted_index = np.argmax(probabilities)
    confidence = probabilities[predicted_index]
    predicted_label = class_names[predicted_index]

    print(f"Predicted Disease: {predicted_label}")
    print(f"Confidence: {confidence:.4f}")
else:
    print(f"Test image not found at {test_image_path}")