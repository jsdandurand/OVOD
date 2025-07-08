"""
Real-time ClipTracker Server for Webcam Object Detection/Segmentation
Hosts the ClipTracker models and provides REST API for frame processing.
"""

import asyncio
import base64
import io
import json
import time
from typing import List, Dict, Any, Optional

import numpy as np
from PIL import Image
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
import uvicorn

from ml.clip_detection import ClipDetector
from ml.tracker import ClipTracker


class ClipTrackerServer:
    """Server class that manages ClipTracker models and processing."""
    
    def __init__(self):
        self.detector = None
        self.tracker = None
        self.current_prompts: List[str] = []
        self.mode = "detection"  # Only detection mode supported for real-time
        self.processing = False
        self.tracking_enabled = True
        
        # Settings
        self.confidence_threshold = 0.1
        self.similarity_threshold = 0.25
        self.apply_filtering = True
        
    async def initialize_models(self):
        """Initialize ClipTracker models (async to avoid blocking)."""
        print("Initializing ClipTracker models...")
        # Initialize in separate thread to avoid blocking
        loop = asyncio.get_event_loop()
        
        # Initialize detector
        self.detector = await loop.run_in_executor(None, ClipDetector)
        print("Detection model loaded!")
        
        # Initialize tracker
        self.tracker = await loop.run_in_executor(None, ClipTracker)
        print("Tracking system loaded!")
        
    def set_prompts(self, prompts: List[str]):
        """Update the current text prompts."""
        self.current_prompts = [p.strip() for p in prompts if p.strip()]
        
    def set_mode(self, mode: str):
        """Set detection mode (segmentation removed for real-time performance)."""
        if mode == "detection":
            self.mode = mode
            
    def set_settings(self, settings: Dict[str, Any]):
        """Update detection settings."""
        if "confidence_threshold" in settings:
            self.confidence_threshold = max(0.01, min(0.99, settings["confidence_threshold"]))
        if "similarity_threshold" in settings:
            self.similarity_threshold = max(0.01, min(0.99, settings["similarity_threshold"]))
        if "apply_filtering" in settings:
            self.apply_filtering = bool(settings["apply_filtering"])
        if "tracking_enabled" in settings:
            self.tracking_enabled = bool(settings["tracking_enabled"])
    
    async def process_frame(self, image: Image.Image) -> Dict[str, Any]:
        """Process a single frame and return results."""
        if self.processing or not self.current_prompts:
            return {"results": [], "timing": {}, "frame_info": {}}
            
        self.processing = True
        start_time = time.time()
        
        try:
            # Run in executor to avoid blocking event loop
            loop = asyncio.get_event_loop()
            
            # Run detection with optional tracking
            if self.tracking_enabled:
                result = await loop.run_in_executor(
                    None, self._run_detection_with_tracking, image
                )
            else:
                result = await loop.run_in_executor(
                    None, self._run_detection, image
                )
                
            # Add frame info
            result["frame_info"] = {
                "size": image.size,
                "mode": self.mode,
                "prompts": self.current_prompts,
                "total_time": time.time() - start_time
            }
            
            return result
            
        finally:
            self.processing = False
    
    def _run_detection(self, image: Image.Image) -> Dict[str, Any]:
        """Run detection on image (blocking call)."""
        timing_dict = {}
        
        boxes, scores, similarities, classes = self.detector.detect(
            image, self.current_prompts,
            confidence_threshold=self.confidence_threshold,
            similarity_threshold=self.similarity_threshold,
            iou_threshold=0.001,
            apply_nms=self.apply_filtering,
            verbose=False,
            timing_dict=timing_dict
        )
        
        # Convert results to JSON-serializable format
        results = []
        for i, (box, sim, cls) in enumerate(zip(boxes, similarities, classes)):
            # Validate bbox data
            if box is not None and len(box) == 4:
                results.append({
                    "id": i,
                    "class": cls,
                    "confidence": float(sim),
                    "bbox": [float(x) for x in box],  # [x1, y1, x2, y2]
                    "type": "detection"
                })
            
        return {"results": results, "timing": timing_dict}
    
    def _run_detection_with_tracking(self, image: Image.Image) -> Dict[str, Any]:
        """Run detection with tracking (blocking call)."""
        timing_dict = {}
        
        # Run detection
        boxes, scores, similarities, classes = self.detector.detect(
            image, self.current_prompts,
            confidence_threshold=self.confidence_threshold,
            similarity_threshold=self.similarity_threshold,
            iou_threshold=0.001,
            apply_nms=self.apply_filtering,
            verbose=False,
            timing_dict=timing_dict
        )
        
        # Convert to tracker format
        detections = []
        for i, (box, sim, cls) in enumerate(zip(boxes, similarities, classes)):
            # Validate bbox data
            if box is not None and len(box) == 4:
                detections.append({
                    'bbox': box.tolist(),
                    'class': cls,
                    'confidence': float(sim)
                })
        
        # Update tracker
        tracked_objects = self.tracker.update(detections)
        
        # Get tracking metrics
        tracking_metrics = self.tracker.get_metrics()
        tracking_history = self.tracker.get_all_tracking_history()
        
        # Convert tracked objects to result format
        results = []
        for obj in tracked_objects:
            # Validate bbox data
            if 'bbox' in obj and obj['bbox'] is not None and len(obj['bbox']) == 4:
                results.append({
                    "id": obj['id'],
                    "class": obj['class'],
                    "confidence": obj['confidence'],
                    "bbox": obj['bbox'],
                    "type": "detection",
                    "track_id": obj['id'],
                    "track_age": obj['age'],
                    "track_hits": obj['hits']
                })
        
        # Add tracking info to timing dict
        timing_dict['tracking'] = tracking_metrics
        
        return {"results": results, "timing": timing_dict, "tracking_history": tracking_history}
    



# Global server instance
cliptracker_server = ClipTrackerServer()

# FastAPI app
app = FastAPI(title="ClipTracker Real-time Server", version="1.0.0")

# CORS middleware for web frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files not needed - serving frontend directly from root route


@app.on_event("startup")
async def startup_event():
    """Initialize models on startup."""
    await cliptracker_server.initialize_models()


@app.get("/")
async def read_root():
    """Serve the main application page."""
    import os
    frontend_path = os.path.join(os.path.dirname(__file__), "app_frontend.html")
    return HTMLResponse(content=open(frontend_path).read(), status_code=200)


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "models_loaded": cliptracker_server.detector is not None and cliptracker_server.tracker is not None,
        "processing": cliptracker_server.processing
    }


@app.post("/prompts")
async def update_prompts(data: Dict[str, List[str]]):
    """Update text prompts."""
    prompts = data.get("prompts", [])
    cliptracker_server.set_prompts(prompts)
    return {"status": "success", "prompts": cliptracker_server.current_prompts}


@app.post("/mode")
async def update_mode(data: Dict[str, str]):
    """Switch between detection and segmentation mode."""
    mode = data.get("mode", "detection")
    cliptracker_server.set_mode(mode)
    return {"status": "success", "mode": cliptracker_server.mode}


@app.post("/settings")
async def update_settings(data: Dict[str, Any]):
    """Update detection settings."""
    cliptracker_server.set_settings(data)
    return {
        "status": "success", 
        "settings": {
            "confidence_threshold": cliptracker_server.confidence_threshold,
            "similarity_threshold": cliptracker_server.similarity_threshold,
            "apply_filtering": cliptracker_server.apply_filtering,
            "tracking_enabled": cliptracker_server.tracking_enabled
        }
    }


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time frame processing."""
    await websocket.accept()
    
    try:
        while True:
            # Receive frame data
            data = await websocket.receive_text()
            frame_data = json.loads(data)
            
            if frame_data["type"] == "frame":
                # Decode base64 image
                image_data = base64.b64decode(frame_data["image"])
                image = Image.open(io.BytesIO(image_data)).convert('RGB')
                
                # Process frame
                results = await cliptracker_server.process_frame(image)
                
                # Send results back
                await websocket.send_text(json.dumps({
                    "type": "results",
                    "data": results
                }))
                
            elif frame_data["type"] == "ping":
                # Respond to ping
                await websocket.send_text(json.dumps({
                    "type": "pong"
                }))
                
    except WebSocketDisconnect:
        print("Client disconnected")
    except Exception as e:
        print(f"WebSocket error: {e}")
        import traceback
        traceback.print_exc()
        await websocket.close()


if __name__ == "__main__":
    print("Starting ClipTracker Real-time Server...")
    print("Open http://localhost:8000 in your browser")
    
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000,
        log_level="info"
    ) 