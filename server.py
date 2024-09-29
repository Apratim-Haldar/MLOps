import tempfile
import matplotlib.pyplot as plt
import io
import base64
import pickle
from dotenv import load_dotenv
from flask import Flask, jsonify
from flask_cors import CORS

load_dotenv()
app = Flask(__name__)
CORS(app)

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

if __name__ == '__main__':
    app.run(debug=True)