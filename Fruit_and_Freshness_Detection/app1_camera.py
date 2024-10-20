import cv2
import requests
import base64
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from flask import Flask, render_template
from inference_sdk import InferenceHTTPClient

# ------------------------------
# Configuration
# ------------------------------

# Roboflow API details
CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="M7zyZ70m4lkTuBrzyP8e"
)

# Model ID for Roboflow API
MODEL_ID = "fruits-by-yolo/1"

# Path to your TensorFlow model
TF_MODEL_PATH = 'E:/Flipkart/Fruit/freshness_detection_model2.h5'  # Update with your actual path

# Define the class labels for the TensorFlow model
TF_LABELS = [
    'rotten banana',
    'fresh banana',
    'rotten apple',
    'fresh apple',
    'rotten oranges',
    'fresh oranges'
]

# Relevant YOLO classes
RELEVANT_CLASSES = ['Banana', 'Apple', 'Orange']

# Initialize Flask app
app = Flask(__name__)

# ------------------------------
# Helper Functions
# ------------------------------

# Function to convert image to base64
def image_to_base64(image):
    _, buffer = cv2.imencode('.jpg', image)
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    return img_base64

# Function to preprocess the Region of Interest (ROI) for TensorFlow model
def preprocess_roi(roi):
    try:
        img_resized = cv2.resize(roi, (150, 150))  # Resize to model's expected size
    except Exception as e:
        print(f"Error resizing ROI: {e}")
        return None
    img_array = image.img_to_array(img_resized)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array /= 255.0  # Normalize to [0, 1]
    return img_array

# Function to map TensorFlow prediction to its class label
def map_tf_prediction(pred_class_idx):
    if 0 <= pred_class_idx < len(TF_LABELS):
        return TF_LABELS[pred_class_idx]
    else:
        return "Unknown"

# ------------------------------
# Load TensorFlow Model
# ------------------------------

print("Loading TensorFlow model...")
tf_model = load_model(TF_MODEL_PATH)
print("TensorFlow model loaded.")

# ------------------------------
# Initialize Webcam
# ------------------------------

print("Initializing webcam...")
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# ------------------------------
# Flask Route for Real-Time Detection
# ------------------------------

@app.route('/')
def index():
    return render_template('web_camera.html')

def detect_and_update_image():
    print("Starting real-time detection...")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        # Convert the frame to base64 for Roboflow API submission
        img_base64 = image_to_base64(frame)

        # Send the image to the Roboflow API for fruit detection
        result = CLIENT.infer(img_base64, model_id=MODEL_ID)

        # Extract predictions from the result
        predictions = result.get('predictions', [])

        for prediction in predictions:
            # Only consider relevant classes (banana, apple, orange)
            label = prediction['class']
            if label not in RELEVANT_CLASSES:
                continue

            # Extract bounding box coordinates
            x = int(prediction['x'])
            y = int(prediction['y'])
            width = int(prediction['width'])
            height = int(prediction['height'])
            confidence = prediction['confidence']

            # Extract ROI from the frame
            x1 = max(0, x - width // 2)
            y1 = max(0, y - height // 2)
            x2 = min(frame.shape[1], x + width // 2)
            y2 = min(frame.shape[0], y + height // 2)
            roi = frame[y1:y2, x1:x2]

            # Preprocess the ROI for the TensorFlow model
            roi_preprocessed = preprocess_roi(roi)
            if roi_preprocessed is None:
                continue

            # Classify the ROI using the TensorFlow model
            preds = tf_model.predict(roi_preprocessed)
            pred_class_idx = np.argmax(preds)
            freshness_label = map_tf_prediction(pred_class_idx)

            # Display the detection and freshness classification
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} - {freshness_label} ({confidence:.2f})", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Save the latest frame with detections
        cv2.imwrite('static/output.jpg', frame)

# Run the detection in a separate thread
import threading
detection_thread = threading.Thread(target=detect_and_update_image)
detection_thread.start()

# ------------------------------
# Start the Flask app
# ------------------------------

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)  # Adjust the host and port as needed

# ------------------------------
# Cleanup
# ------------------------------

cap.release()
cv2.destroyAllWindows()
print("Program terminated gracefully.")
