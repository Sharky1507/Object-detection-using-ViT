# 🦉 Object Detection using ViT

**Detecting objects in camera through ViT (Vision Transformer)**

## 📸 OWL-ViT Object Detection Dashboard

A Flask-based web application for **real-time zero-shot object detection** using Google’s **OWL-ViT** model.  
Detect any object simply by describing it in natural language!

---

## 🚀 Features

- 🎥 Real-time video streaming with object detection  
- ✍️ Web-based control of detection labels  
- 📊 Live detection logging  
- 📁 CSV export of detection history  

---

## 🧠 How It Works

**OWL-ViT** leverages **vision-language pretraining** to enable zero-shot object detection. Here's a quick breakdown:

- Images are processed using a **Vision Transformer (ViT)** backbone  
- Text prompts are encoded via a **text transformer**
- **Cross-attention layers** align visual and textual features
- This allows the system to localize and detect objects using **natural language** descriptions — no task-specific retraining needed!

The Flask interface offers real-time control over detection and instant visual feedback.

---

## ⚠️ Challenges Faced

- **Text Ambiguity:**  
  Prompt phrasing significantly affects performance. (e.g., _"a watch"_ vs _"a digital wristwatch with metal band"_)

- **Resource Constraints:**  
  Requires ~4GB GPU memory for 640px images. Not ideal for low-resource environments.

---

## 🔮 Future Improvements

- **Prompt Optimization:**  
  Use LLMs to auto-generate better object descriptions (prompt engineering)

- **Temporal Filtering:**  
  Add consistency between frames to reduce flicker/noise in detection

- **Edge Device Optimization:**  
  Quantize the model for Raspberry Pi using **ONNX Runtime**

---

## ⚙️ Setup

### ✅ Prerequisites

- Python 3.8+  
- NVIDIA GPU with CUDA 11.7 (Recommended)

### 📦 Installation

```bash
git clone https://github.com/Sharky1507/Object-detection-using-ViT.git
cd Object-detection-using-ViT
pip install -r requirements.txt
python app.py
```
- Open http://localhost:5000 in your browser.
### Usage
- Enter comma-separated object descriptions (e.g., "a wristwatch, a charging cable")

- View real-time detections in the video feed

- Check the sidebar for detection logs

- Click "Export Full Log" to save results as CSV

