# 🎯 ClipTracker: Open Vocabulary Object Detection & Tracking

A real-time computer vision system that performs object detection, segmentation, and multi-object tracking using natural language descriptions. Built with state-of-the-art deep learning models and optimized for live webcam inference.

## 🧠 Machine Learning Pipeline

### Architecture Overview
The system combines four neural networks in a sophisticated pipeline:

1. **YOLOv10 (Detection)** - Generates region proposals with bounding boxes
2. **YOLOv8-seg (Segmentation)** - Produces pixel-accurate segmentation masks  
3. **CLIP (Classification)** - Performs open-vocabulary classification using text-image similarity
4. **SORT Tracker (Multi-Object Tracking)** - Maintains object identity across frames

### Technical Implementation
- **Object Proposals**: YOLOv10 provides objectness scores while ignoring class predictions
- **Semantic Understanding**: CLIP embeddings enable detection of arbitrary objects described in natural language
- **Real-time Optimization**: Frame processing adapts to hardware capabilities with no buffering
- **Performance Tuning**: Configurable confidence thresholds, NMS, and mask IoU filtering

## 🎯 Multi-Object Tracking System

### Tracking Architecture
The system implements **SORT (Simple Online and Realtime Tracking)** with enhancements:

- **Kalman Filter**: Predictive state estimation with velocity modeling
- **Hungarian Algorithm**: Optimal assignment between detections and tracks
- **Feature Similarity**: CLIP embeddings for robust re-identification
- **Track Management**: Confirmation, deletion, and merging logic

### Advanced Tracking Features
- **Fast Motion Handling**: Displacement-based association with velocity scaling
- **Track Persistence**: Extended lifetime for high-velocity objects
- **Similar Track Merging**: Automatic consolidation of fragmented tracks
- **Adaptive Parameters**: Dynamic thresholds based on motion characteristics

## 🎬 Demo Video

![Tracking Demo](examples/horse_track.gif)

*Multi-object tracking demonstration showing persistent object IDs across frames, even during fast motion and temporary occlusions.*

## 🛠️ Technology Stack

**Deep Learning & Computer Vision**
- PyTorch, torchvision
- Ultralytics YOLOv8/v10
- OpenCLIP
- OpenCV
- PIL (Pillow)
- NumPy, SciPy

**Web Application & API**
- FastAPI
- WebSockets
- Uvicorn
- HTML5 Canvas
- JavaScript (ES6+)

**Performance & Utilities**
- Asyncio (async/await)
- Base64 encoding
- Real-time video streaming
- GPU acceleration (CUDA)

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Launch application
python start_app.py
```

Open browser to `http://localhost:8000` and start detecting objects with natural language!

## 🎮 Features

- **Open Vocabulary**: Detect any object by describing it in text
- **Multi-Object Tracking**: Persistent object IDs across frames
- **Dual Mode**: Real-time detection (bounding boxes) or segmentation (pixel masks)
- **Live Performance**: Adaptive frame rate matching processing speed
- **Dynamic Prompts**: Add/remove detection targets while running
- **Web Interface**: Professional browser-based UI with real-time visualization
- **Tracking Metrics**: Real-time performance statistics and analytics

## 📁 Project Structure

```
├── start_app.py          # Application launcher
├── ml/                   # Machine learning models
│   ├── clip_detection.py
│   ├── clip_segmentation.py
│   └── tracker.py        # Multi-object tracking system
├── app/                  # Web application
│   ├── app_server.py     # FastAPI backend
│   └── app_frontend.html # Frontend interface
├── output_videos/        # Demo videos
│   └── dog_track.mp4     # Tracking demonstration
└── requirements.txt      # Dependencies
```

## 🏆 Technical Achievements

- **Zero-shot Detection**: No training required for new object classes
- **Real-time Processing**: Optimized inference pipeline for live video
- **Robust Tracking**: Handles fast motion, occlusion, and blur
- **Scalable Architecture**: Modular design with async processing
- **Cross-platform**: Runs on CPU/GPU with automatic device detection
- **Production-ready**: Professional web interface with error handling

---

*Built with modern ML engineering practices and optimized for real-world deployment.* 