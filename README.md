# IdentifyAI: Animal Object Detection with YOLOv5

![IdentifyAI Logo](static/UI/Logo.png)

## 📋 Overview

IdentifyAI is a powerful AI-based object detection application specifically designed for detecting and identifying various animal species. Built with YOLOv5 and Flask, this application allows users to detect and identify animals in images, videos, and through live webcam feeds.

## ✨ Features

- **Multi-format Detection**: Process images, videos, and live webcam feeds for object detection
- **Focused Animal Detection**: Specifically trained to identify animals including birds, dogs, cats, bears, elephants, giraffes, zebras, horses, cows, and sheep
- **Real-time Processing**: Live webcam feed with real-time animal detection
- **User-friendly Interface**: Clean and intuitive web-based interface
- **Confidence Threshold**: Shows only detections with confidence score > 0.25

## 🎮 Demo

The application provides three ways to detect animals:

1. **Scan Images**: Upload images containing animals
2. **Scan Videos**: Upload videos for animal detection
3. **Scan Live**: Use your webcam for real-time animal detection

## 🔧 Technologies Used

- **YOLOv5**: State-of-the-art object detection model
- **Flask**: Python web framework for the backend
- **OpenCV**: For image and video processing
- **PyTorch**: Deep learning framework for running the YOLOv5 model
- **HTML/CSS/JavaScript**: Frontend interface

## 🚀 Installation

### Prerequisites

- Python 3.6+
- pip (Python package installer)
- Git

### Setup

1. Clone the repository:

   ```
   git clone https://github.com/yourusername/identifyai.git
   cd identifyai
   ```

2. Install the required packages:

   ```
   pip install flask torch opencv-python
   ```

3. Install YOLOv5 dependencies:
   ```
   pip install -r https://raw.githubusercontent.com/ultralytics/yolov5/master/requirements.txt
   ```

## 📝 Usage

1. Start the Flask application:

   ```
   python app.py
   ```

2. Open your web browser and go to:

   ```
   http://localhost:5000
   ```

3. Use the interface to upload an image/video or start the webcam feed

## 📂 Project Structure

- `app.py`: Main Flask application
- `best.pt`: Trained YOLOv5 model weights
- `templates/`: Contains HTML templates
  - `index.html`: Main application interface
- `static/`: Static files (CSS, JavaScript, images)
  - `UI/`: Interface assets
  - `results/`: Storage for processed images and videos
- `uploads/`: Temporary storage for uploaded files

## 🧠 Model Information

The application uses a custom-trained YOLOv5 model optimized for animal detection. The model has been trained to identify the following animal classes with high accuracy:

- Birds
- Dogs
- Cats
- Bears
- Elephants
- Giraffes
- Zebras
- Horses
- Cows
- Sheep

## 🙏 Acknowledgments

- YOLOv5 by Ultralytics
- Flask web framework
