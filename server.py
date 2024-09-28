from flask import Flask, request, jsonify
import cv2
import numpy as np
import pickle
import os
import librosa
import tempfile

app = Flask(__name__)

# Load the pre-trained models (Update the paths as required)
with open('model_efficientnet.pkl', 'rb') as f:
    model = pickle.load(f)

with open('audio_model.pkl', 'rb') as f:
    audio_model = pickle.load(f)

# Image Prediction Function
def predict_image(image_path, model):
    img = cv2.imread(image_path)
    if img is None:
        return None, None

    # Preprocess the image
    img = cv2.resize(img, (224, 224))
    img = np.expand_dims(img, axis=0)

    # Make prediction
    prediction = model.predict(img)
    predicted_label = (prediction > 0.5).astype(int)[0][0]
    probability = prediction[0][0] * 100
    return predicted_label, probability

@app.route('/predict_image', methods=['POST'])
def predict_image_endpoint():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_image:
        file.save(temp_image.name)
        image_path = temp_image.name

    predicted_label, probability = predict_image(image_path, model)

    # Remove the temporary file after processing
    os.remove(image_path)

    if predicted_label is not None:
        if predicted_label == 0:
            result = 'REAL'
        else:
            result= 'FAKE'
        return jsonify({'label': result, 'probability': probability})
    else:
        return jsonify({'error': 'Failed to process the image'}), 500

# Video Processing Function
def process_video(video_path, model, frame_interval=30):
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    predictions = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            frame = cv2.resize(frame, (224, 224))
            frame = np.expand_dims(frame, axis=0)

            prediction = model.predict(frame)
            predictions.append(prediction[0][0])

        frame_count += 1

    cap.release()
    avg_prediction = np.mean(predictions)
    if avg_prediction < 0.25:
        return "FAKE"
    else:
        return "REAL"

@app.route('/predict_video', methods=['POST'])
def predict_video_endpoint():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_video:
        file.save(temp_video.name)
        video_path = temp_video.name

    video_result = process_video(video_path, model)

    # Remove the temporary file after processing
    os.remove(video_path)
    
    return jsonify({'label': video_result})

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
    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_audio:
        file.save(temp_audio.name)
        audio_path = temp_audio.name

    result = predict_fake_audio(audio_path)

    # Remove the temporary file after processing
    os.remove(audio_path)
    
    return jsonify({'label': result})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
