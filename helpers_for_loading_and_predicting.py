import json
import numpy as np
import cv2
from tensorflow.keras.models import load_model

MODEL_PATH = 'models/model.h5'
CLASSES_PATH = 'classes.json'
IMG_SIZE = (224, 224)

_model = None
_class_map = None


def load_resources():
    global _model, _class_map
    if _model is None:
        _model = load_model(MODEL_PATH)
    if _class_map is None:
        with open(CLASSES_PATH, 'r') as f:
            class_indices = json.load(f)
        # invert mapping: index -> class
        _class_map = {v: k for k, v in class_indices.items()}


def preprocess_image(image_path):
    # Read image with OpenCV
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError('Could not read image: ' + image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, IMG_SIZE)
    img = img.astype('float32') / 255.0
    return np.expand_dims(img, axis=0)


def predict(image_path, top_k=3):
    load_resources()
    x = preprocess_image(image_path)
    preds = _model.predict(x)[0]
    top_idx = preds.argsort()[-top_k:][::-1]
    results = [( _class_map[int(i)], float(preds[int(i)]) ) for i in top_idx]
    return results