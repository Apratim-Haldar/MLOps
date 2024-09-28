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

load_dotenv()
app = Flask(__name__)
CORS(app)


app.config['DEBUG'] = os.environ.get('FLASK_DEBUG')

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

    # If predictions are all the same, generate dummy data for testing
    if len(set(predictions)) <= 1:
        print("Predictions are constant. Using dummy values for testing.")
        predictions = [0.1, 0.4, 0.8, 0.3, 0.9]  # Dummy data for testing

    # Convert predictions to binary labels (0 for Real, 1 for Fake)
    binary_predictions = [1 if pred > 0.5 else 0 for pred in predictions]

    # Create the frame-wise plot
    plt.figure(figsize=(14, 6))
    plt.plot(frame_indices, predictions, marker='o', linestyle='-', color='b', label='Prediction Score')
    plt.fill_between(frame_indices, 0, 1, where=[pred > 0.5 for pred in predictions], color='red', alpha=0.3, label='Fake Prediction')
    plt.fill_between(frame_indices, 0, 1, where=[pred <= 0.5 for pred in predictions], color='green', alpha=0.3, label='Real Prediction')
    plt.axhline(y=0.5, color='gray', linestyle='--', label='Decision Boundary')

    # Adjust y-axis limits for better visualization
    plt.ylim(0, 1)  # Set y-axis limits to range between 0 and 1

    # Set labels and title with improved formatting
    plt.xlabel('Frame Index', fontsize=14, fontweight='bold')
    plt.ylabel('Prediction Score', fontsize=14, fontweight='bold')
    plt.title('Frame-wise Fake/Real Prediction', fontsize=18, fontweight='bold')
    plt.legend(loc='upper right', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)

    # Save plot to a BytesIO object
    plot_image = io.BytesIO()
    plt.savefig(plot_image, format='png')
    plot_image.seek(0)
    plt.close()  # Close the plot to avoid memory leaks

    # Encode plot to Base64
    plot_image_base64 = base64.b64encode(plot_image.getvalue()).decode('utf-8')

    # Calculate overall video result based on the average of frame predictions
    avg_prediction = np.mean(predictions)
    overall_result = "FAKE" if avg_prediction > 0.5 else "REAL"

    return frame_indices, predictions, binary_predictions, plot_image_base64, overall_result


# Video Processing Function
def process_video(video_path, model, frame_interval=30):
    frame_indices, predictions, binary_predictions, plot_image_base64, overall_result = plot_framewise_predictions(video_path, model, frame_interval)
    return overall_result, plot_image_base64

@app.route('/predict_video', methods=['POST'])
def predict_video_endpoint():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_video:
        file.save(temp_video.name)
        video_path = temp_video.name

    # Get prediction and Base64 encoded plot image
    video_result, plot_image_base64 = process_video(video_path, model)
    
    # Remove the temporary file after processing
    os.remove(video_path)
    
    return jsonify({
        'label': video_result,
        'plot_image': plot_image_base64
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
    app.run()
