from flask import Flask, request, jsonify
import cv2
import numpy as np
import pickle
import os
import librosa
import tempfile
import io
import base64
from dotenv import load_dotenv
from flask_cors import CORS

load_dotenv()
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})  # Allow CORS from any origin

    
# Load the pre-trained models (Update the paths as required)
with open('model_efficientnet.pkl', 'rb') as f:
    model = pickle.load(f)

with open('audio_model.pkl', 'rb') as f:
    audio_model = pickle.load(f)

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

# Function to generate frame-wise prediction plot and return base64 encoded image
def plot_framewise_predictions(video_path, model, frame_interval=30):
    """
    Process the video and generate a plot showing the frame-wise fake/real predictions.
    Returns the frame indices, predictions, binary predictions, and the Base64 encoded plot image.
    """
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    frame_indices = []  # To store frame indices
    predictions = []    # To store prediction probabilities

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Process every frame_interval frame
        if frame_count % frame_interval == 0:
            # Preprocess frame
            frame_resized = cv2.resize(frame, (224, 224))
            frame_resized = np.expand_dims(frame_resized, axis=0)

            # Make prediction
            prediction = model.predict(frame_resized)
            prediction_score = prediction[0][0]

            # Store the frame index and prediction score
            frame_indices.append(frame_count)
            predictions.append(prediction_score)

        frame_count += 1

    cap.release()

    # DEBUG: Print prediction values to see if they vary
    print("Predictions:", predictions)  # Check the prediction values
    print("Frame Indices:", frame_indices)

    # If predictions are all the same, generate dummy data for testin

    # Calculate overall video result based on the average of frame predictions
    avg_prediction = np.mean(predictions)
    overall_result = "FAKE" if avg_prediction > 0.5 else "REAL"

    return overall_result


# Video Processing Function
def process_video(video_path, model, frame_interval=30):
    overall_result = plot_framewise_predictions(video_path, model, frame_interval)
    return overall_result

@app.route('/predict_video', methods=['POST'])
def predict_video_endpoint():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_video:
        file.save(temp_video.name)
        video_path = temp_video.name

    # Get prediction and Base64 encoded plot image
    video_result = process_video(video_path, model)
    
    # Remove the temporary file after processing
    os.remove(video_path)
    
    return jsonify({
        'label': video_result,
    })

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
     app.run(debug=True)
    
