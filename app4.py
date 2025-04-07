import cv2
import torch
import numpy as np
import pandas as pd
from flask import Flask, render_template, Response, request, jsonify
from transformers import OwlViTProcessor, OwlViTForObjectDetection
from datetime import datetime
import threading
import time

app = Flask(__name__)

# Initialize model and video capture in thread-safe way
model_lock = threading.Lock()
cap_lock = threading.Lock()

# Load OWL-ViT model
processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32")
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# Shared state with thread locking
shared_state = {
    "custom_labels": ["a lightbulb", "a matchstick", "a monitor", "a lion", "a gaming console"],
    "log": [],
    "last_frame": None,
    "is_running": True,
    "source": 0
}

def detection_loop():
    with cap_lock:
        cap = cv2.VideoCapture(shared_state["source"])
    
    while shared_state["is_running"]:
        with cap_lock:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.1)
                continue

        with model_lock:
            inputs = processor(text=shared_state["custom_labels"], images=frame, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = model(**inputs)
            
            target_sizes = torch.tensor([frame.shape[:2]]).to(device)
            results = processor.post_process_object_detection(
                outputs=outputs, target_sizes=target_sizes, threshold=0.25
            )[0]

        # Annotate frame
        for box, score, label_id in zip(results["boxes"], results["scores"], results["labels"]):
            box = box.int().cpu().numpy()
            label = shared_state["custom_labels"][label_id]
            conf = round(score.item(), 2)
            x1, y1, x2, y2 = box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (50, 255, 50), 2)
            text = f"{label}: {conf}"
            cv2.putText(frame, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 255, 50), 2)

        # Update shared state
        timestamp = datetime.now().isoformat()
        for box, score, label_id in zip(results["boxes"], results["scores"], results["labels"]):
            shared_state["log"].append({
                "timestamp": timestamp,
                "label": shared_state["custom_labels"][label_id],
                "score": round(score.item(), 2)
            })
        
        _, jpeg = cv2.imencode('.jpg', frame)
        shared_state["last_frame"] = jpeg.tobytes()

@app.route('/')
def index():
    return render_template('index.html', labels=shared_state["custom_labels"])

@app.route('/video_feed')
def video_feed():
    def generate():
        while True:
            if shared_state["last_frame"]:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + 
                       shared_state["last_frame"] + b'\r\n')
            time.sleep(0.05)
    return Response(generate(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/update_labels', methods=['POST'])
def update_labels():
    new_labels = request.json.get('labels', [])
    with model_lock:
        shared_state["custom_labels"] = new_labels
    return jsonify(success=True)

@app.route('/get_logs')
def get_logs():
    return jsonify(shared_state["log"][-20:])  # Return last 20 entries

@app.route('/export_logs')
def export_logs():
    df = pd.DataFrame(shared_state["log"])
    csv = df.to_csv(index=False)
    return Response(
        csv,
        mimetype="text/csv",
        headers={"Content-disposition":
                 "attachment; filename=detection_log.csv"})

def run_app():
    detector_thread = threading.Thread(target=detection_loop)
    detector_thread.daemon = True
    detector_thread.start()
    app.run(host='0.0.0.0', port=5000, threaded=True)
    
if __name__ == '__main__':
    run_app()