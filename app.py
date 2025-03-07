import os
import librosa
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-GUI backend
import matplotlib.pyplot as plt
from PIL import Image
import torch
from transformers import ViTForImageClassification, ViTFeatureExtractor
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS

# Load model
model_dir = os.path.join(os.path.dirname(__file__), "modelll")
model = ViTForImageClassification.from_pretrained(model_dir)
model.eval()
feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224-in21k")

def create_spectrogram(audio_chunk, sr, save_path):
    mel_spectrogram = librosa.feature.melspectrogram(y=audio_chunk, sr=sr, n_mels=128, fmax=8000)
    mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mel_spectrogram_db, x_axis='time', y_axis='mel', sr=sr)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()

def classify_spectrogram(image_path, model, feature_extractor):
    image = Image.open(image_path).convert("RGB")
    inputs = feature_extractor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class_idx = logits.argmax(-1).item()
    return predicted_class_idx

def process_audio(file_path, model, feature_extractor, output_dir):
    y, sr = librosa.load(file_path, sr=None)
    os.makedirs(output_dir, exist_ok=True)
    chunk_length = 2 * sr
    total_chunks = int(np.ceil(len(y) / chunk_length))
    bee_count = 0

    for i in range(total_chunks):
        start_sample = i * chunk_length
        end_sample = min((i + 1) * chunk_length, len(y))
        audio_chunk = y[start_sample:end_sample]
        spectrogram_path = os.path.join(output_dir, f"chunk_{i}.png")
        create_spectrogram(audio_chunk, sr, spectrogram_path)
        predicted_class_idx = classify_spectrogram(spectrogram_path, model, feature_extractor)
        if predicted_class_idx == 1:
            bee_count += 1

    return "Queen" if bee_count >= 0.7 * total_chunks else "No Queen"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'audio' not in request.files:
            return jsonify({"error": "No audio file provided"}), 400

        audio_file = request.files['audio']
        temp_path = "temp_audio.mp3"
        output_dir = "spectrograms"

        audio_file.save(temp_path)
        result = process_audio(temp_path, model, feature_extractor, output_dir)

        # Clean up safely
        if os.path.exists(temp_path):
            os.remove(temp_path)
        if os.path.exists(output_dir):
            for file in os.listdir(output_dir):
                file_path = os.path.join(output_dir, file)
                if os.path.exists(file_path):
                    os.remove(file_path)

        return jsonify({"result": result})
    except Exception as e:
        print(f"Error in predict: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5005, debug=True)