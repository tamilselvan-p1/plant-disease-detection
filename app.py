import os
import json
import numpy as np
import cv2
import tensorflow as tf
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename

# ==========================================
# 1. INITIALIZATION & CONFIGURATION
# ==========================================
app = Flask(__name__)

# Configure upload settings
UPLOAD_FOLDER = os.path.join("static", "uploads")
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB max limit

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load trained model and class indices map
MODEL_PATH = "plant_disease_model.keras"
CLASS_INDICES_PATH = "class_indices.json"

print("--> Loading trained model...")
model = tf.keras.models.load_model(MODEL_PATH)

print("--> Loading class mapping...")
with open(CLASS_INDICES_PATH, "r") as f:
    class_indices = json.load(f)

# Preventive recommendations dictionary
DISEASE_INFO = {
    "Potato___Early_blight": {
        "status": "Diseased",
        "description": "Fungal infection causing dark, concentric ring spots on leaves.",
        "prevention": [
            "Apply copper-based fungicides early in the season.",
            "Ensure proper crop rotation with non-solanaceous crops.",
            "Avoid overhead irrigation to keep foliage dry."
        ]
    },
    "Potato___Late_blight": {
        "status": "Diseased",
        "description": "Destructive water mold disease causing water-soaked dark spots and white mold growth.",
        "prevention": [
            "Destroy infected plants immediately to prevent spore dispersal.",
            "Apply protective fungicides prior to high-humidity weather.",
            "Plant certified disease-free seed tubers."
        ]
    },
    "Potato___healthy": {
        "status": "Healthy",
        "description": "The leaf exhibits vibrant coloration with no visible signs of pathogen infection.",
        "prevention": [
            "Maintain optimal watering and soil nutrient balance.",
            "Perform regular field monitoring for pests.",
            "Ensure adequate sunlight and weed control."
        ]
    }
}

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(image_path):
    """Load image using OpenCV, resize to 224x224, convert color space, and batchify."""
    # Read image in BGR
    img = cv2.imread(image_path)
    # Convert BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Resize to MobileNetV2 target resolution
    img = cv2.resize(img, (224, 224))
    # Convert to float array
    img_array = np.array(img, dtype=np.float32)
    # Add batch dimension: shape (1, 224, 224, 3)
    img_batch = np.expand_dims(img_array, axis=0)
    return img_batch

# ==========================================
# 2. APPLICATION ROUTES
# ==========================================
@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files["file"]
    
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(filepath)
        
        # Preprocess and run inference
        processed_img = preprocess_image(filepath)
        predictions = model.predict(processed_img)
        
        predicted_class_idx = str(np.argmax(predictions[0]))
        confidence = float(np.max(predictions[0])) * 100
        
        raw_disease_name = class_indices.get(predicted_class_idx, "Unknown Condition")
        formatted_name = raw_disease_name.replace("___", " - ").replace("_", " ")
        
        # Get disease preventive details
        details = DISEASE_INFO.get(raw_disease_name, {
            "status": "Unknown",
            "description": "Detailed diagnosis information not available for this label.",
            "prevention": ["Consult a local agricultural extension specialist for treatment."]
        })
        
        return jsonify({
            "image_url": f"/static/uploads/{filename}",
            "disease_name": formatted_name,
            "confidence": f"{confidence:.2f}%",
            "status": details["status"],
            "description": details["description"],
            "prevention": details["prevention"]
        })
    
    return jsonify({"error": "Invalid file format. Please upload JPG or PNG images."}), 400

if __name__ == "__main__":
    app.run(debug=True, port=5000)