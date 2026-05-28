import joblib
import json
import numpy as np
import base64
import cv2
from wavelet import w2d
import os

# Absolute path tracking ensures artifacts load correctly regardless of where the script runs from
CURRENT_DIR = os.path.dirname(__file__)
ARTIFACTS_DIR = os.path.join(CURRENT_DIR, "artifacts")

__class_name_to_number = {}
__class_number_to_name = {}
__model = None


def classify_image(image_base64_data, file_path=None):
    imgs = get_cropped_image_if_2_eyes(file_path, image_base64_data)
    result = []

    for img in imgs:
        # Resize raw cropped portrait matrix
        scalled_raw_img = cv2.resize(img, (32, 32))
        # Generate raw transform wavelet signature overlay
        img_har = w2d(img, 'db1', 5)
        scalled_img_har = cv2.resize(img_har, (32, 32))

        # Combine pixel intensities and frequency distributions into one unified model array
        combined_img = np.vstack((scalled_raw_img.reshape(
            32 * 32 * 3, 1), scalled_img_har.reshape(32 * 32, 1)))
        len_image_array = 32*32*3 + 32*32
        final_img = combined_img.reshape(1, len_image_array).astype(float)

        # Run inference matrix projections
        predicted_class_idx = __model.predict(final_img)[0]
        raw_probabilities = __model.predict_proba(final_img)[0]

        # Format exact prediction metadata mapping dictionary to send back to UI JavaScript
        result.append({
            'class': __class_number_to_name.get(predicted_class_idx, "unknown"),
            'class_probability': np.round(raw_probabilities * 100, 2).tolist(),
            'class_dictionary': __class_name_to_number
        })

    return result


def get_cv2_image_from_base64_string(b64str):
    if ',' in b64str:
        b64str = b64str.split(',')[1]
    encoded_data = b64str.encode('utf-8')
    nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img


def get_cropped_image_if_2_eyes(file_path, image_base64_data):
    # Load default OpenCV cascades safely via inner python bindings
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_eye.xml')

    if file_path:
        img = cv2.imread(file_path)
    else:
        img = get_cv2_image_from_base64_string(image_base64_data)

    if img is None:
        return []

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    cropped_faces = []
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        if len(eyes) >= 2:
            cropped_faces.append(roi_color)

    return cropped_faces


def load_saved_artifacts():
    print("loading saved artifacts...start")
    global __class_name_to_number
    global __class_number_to_name
    global __model

    dict_path = os.path.join(ARTIFACTS_DIR, "class_dictionary.json")
    model_path = os.path.join(ARTIFACTS_DIR, "saved_model.pkl")

    with open(dict_path, "r") as f:
        __class_name_to_number = json.load(f)
        __class_number_to_name = {v: k for k,
                                  v in __class_name_to_number.items()}

    if __model is None:
        __model = joblib.load(model_path)
    print("loading saved artifacts...done")
