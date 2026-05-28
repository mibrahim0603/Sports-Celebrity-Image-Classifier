# ⚽ Sports Celebrity Image Classifier

An end-to-end Machine Learning web application that automatically identifies sports celebrities from uploaded image files. 

The application uses an **OpenCV Haar Cascade** pipeline to detect facial regions and eye landmarks, transforms spatial pixel features using a **Wavelet Decomposition** framework, and processes classification inferences via a trained **Scikit-Learn Classifier**.





---

## 🛠️ Tech Stack & Architecture

### Frontend
* **HTML5 & CSS3:** Structural styling, page grids, and UI themes.
* **Bootstrap v4.5:** Responsive grid components and alert notifications.
* **Dropzone.js:** Visual drag-and-drop file upload container block.
* **jQuery:** Handles asynchronous network event handlers (`$.post` AJAX streams).

### Backend Server & AI Pipeline
* **Python & Flask Engine:** Light REST API request processor routing data vectors.
* **Flask-CORS:** Manages cross-origin security handshakes between ports.
* **OpenCV:** Runs Cascade classifiers for computer vision face and eye crops.
* **PyWavelets:** Implements 2D Wavelet Discrete transforms for structural feature extraction.
* **Scikit-Learn & Joblib:** Saves and runs the predictive pipeline model.

---

## 🚀 Installation & Local Setup

Follow these setup boundaries to run both backend pipelines and web views on your local desktop layout securely.

### 1. Initialize Python Flask Server
Open your terminal window console framework and point it at your server configuration layout folder:

```bash
# Navigate to the server folder
cd server

# Install the necessary library requirements
pip install flask flask-cors opencv-python PyWavelets joblib numpy

# Run the backend process engine
python server.py