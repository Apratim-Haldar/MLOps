from flask import Flask, request, jsonify
import cv2
import numpy as np
import pickle
import os
import librosa
import tempfile
import matplotlib.pyplot as plt
import io
import base64
from dotenv import load_dotenv
from flask_cors import CORS
import logging

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask application
app = Flask(__name__)
CORS(app)

# Load the pre-trained models
try:
    with open('model_efficientnet.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('audio_model.pkl', 'rb') as f:
        audio_model = pickle.load(f)
    logger.info("✅ Models loaded successfully.")
except Exception as e:
    logger.error(f"❌ Error loading models: {e}")
    raise e

@app.route('/', methods=['GET'])
def welcome():
    try:
        return jsonify({"message": "Welcome to the Deep End!"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Image Prediction Function
def predict_image(image_path, model):
    img = cv2.imread(image_path)
    if img is None:
        return None, None

    img = cv2.resize(img, (224, 224))
    img = np.expand_dims(img, axis=0)
    prediction = model.predict(img)
    predicted_label = (prediction > 0.5).astype(int)[0][0]
    probability = prediction[0][0] * 100
    return predicted_label, probability

@app.route('/predict_image', methods=['POST'])
def predict_image_endpoint():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_image:
            file.save(temp_image.name)
            image_path = temp_image.name
        predicted_label, probability = predict_image(image_path, model)
        os.remove(image_path)
        if predicted_label is not None:
            result = 'REAL' if predicted_label == 0 else 'FAKE'
            return jsonify({'label': result, 'probability': probability})
        else:
            return jsonify({'error': 'Failed to process the image'}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Function to generate frame-wise prediction plot and return base64 encoded image
def plot_framewise_predictions(video_path, model, frame_interval=30):
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    frame_indices = []
    predictions = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            frame_resized = cv2.resize(frame, (224, 224))
            frame_resized = np.expand_dims(frame_resized, axis=0)
            prediction = model.predict(frame_resized)
            prediction_score = prediction[0][0]
            frame_indices.append(frame_count)
            predictions.append(prediction_score)

        frame_count += 1

    cap.release()

    binary_predictions = [1 if pred > 0.5 else 0 for pred in predictions]
    plt.figure(figsize=(14, 6))
    plt.plot(frame_indices, predictions, marker='o', linestyle='-', color='b', label='Prediction Score')
    plt.fill_between(frame_indices, 0, 1, where=[pred > 0.5 for pred in predictions], color='red', alpha=0.3, label='Fake Prediction')
    plt.fill_between(frame_indices, 0, 1, where=[pred <= 0.5 for pred in predictions], color='green', alpha=0.3, label='Real Prediction')
    plt.axhline(y=0.5, color='gray', linestyle='--', label='Decision Boundary')
    plt.ylim(0, 1)
    plt.xlabel('Frame Index')
    plt.ylabel('Prediction Score')
    plt.title('Frame-wise Fake/Real Prediction')
    plt.legend(loc='upper right')
    plt.grid(True, linestyle='--', alpha=0.6)

    plot_image = io.BytesIO()
    plt.savefig(plot_image, format='png')
    plot_image.seek(0)
    plt.close()
    plot_image_base64 = base64.b64encode(plot_image.getvalue()).decode('utf-8')
    avg_prediction = np.mean(predictions)
    overall_result = "FAKE" if avg_prediction > 0.5 else "REAL"
    return frame_indices, predictions, binary_predictions, plot_image_base64, overall_result

@app.route('/predict_video', methods=['POST'])
def predict_video_endpoint():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_video:
            file.save(temp_video.name)
            video_path = temp_video.name
        video_result, plot_image_base64 = process_video(video_path, model)
        os.remove(video_path)
        return jsonify({'label': video_result, 'plot_image': plot_image_base64})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Audio Feature Extraction and Prediction
def extract_features(file_path):
    audio, sample_rate = librosa.load(file_path, sr=None)
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=13)
    return np.mean(mfccs.T, axis=0)

def predict_fake_audio(file_path):
    features = extract_features(file_path)
    prediction = audio_model.predict([features])[0]
    return "FAKE" if prediction == 1 else "REAL"

@app.route('/predict_audio', methods=['POST'])
def predict_audio_endpoint():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_audio:
            file.save(temp_audio.name)
            audio_path = temp_audio.name
        result = predict_fake_audio(audio_path)
        os.remove(audio_path)
        return jsonify({'label': result})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    from waitress import serve
    serve(app, host='0.0.0.0', port=int(os.environ.get("PORT", 5000)))
