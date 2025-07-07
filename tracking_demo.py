"""
Tracking Demo Script for OVOD
Processes video files with object detection and tracking, showing track IDs and motion.
"""

import cv2
import numpy as np
from PIL import Image
import os
import time
import argparse
from tqdm import tqdm
try:
    from .ml.ovod_detection import OVODDetector
    from .ml.tracker import OVODTracker
except ImportError:
    # Fallback for direct script execution
    from ml.ovod_detection import OVODDetector
    from ml.tracker import OVODTracker


def process_video_with_tracking(video_path: str, text_queries, output_path: str):
    """Process video for object detection with tracking and save as video."""
    # Convert single query to list for uniform handling
    if isinstance(text_queries, str):
        text_queries = [text_queries]
    
    # Initialize detector and tracker once
    print("Initializing OVOD Detector and Tracker...")
    detector = OVODDetector()
    tracker = OVODTracker(
        max_age=30,
        min_hits=3,
        iou_threshold=0.3,
        max_displacement=1000.0,
        velocity_scale=1.2
    )
    
    # Open input video
    cap = cv2.VideoCapture(video_path)
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Create output directory if needed
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Define the codec and create VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    print(f"Processing video: {video_path}")
    print(f"Looking for: {text_queries}")
    print(f"Output video: {output_path}")
    print(f"FPS: {fps}")
    print(f"Resolution: {width}x{height}")
    
    # Initialize timing tracking
    timing_stats = {
        'detection_time': 0.0,
        'tracking_time': 0.0,
        'total_frames': 0
    }
    
    # Create progress bar
    with tqdm(total=total_frames, desc="Processing", unit="frame") as pbar:
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert BGR to RGB and then to PIL
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
            
            # Run detection on every frame
            detection_start = time.time()
            boxes, scores, similarities, classes = detector.detect(
                pil_image, text_queries, 
                confidence_threshold=0.3,
                similarity_threshold=0.23,
                verbose=False,
                iou_threshold=0.2
            )
            detection_time = time.time() - detection_start
            
            # Convert detections to tracker format
            detections = []
            for i, (box, sim, cls) in enumerate(zip(boxes, similarities, classes)):
                detections.append({
                    'bbox': box.tolist(),
                    'class': cls,
                    'confidence': float(sim)
                })
            
            # Update tracker
            tracking_start = time.time()
            tracked_objects = tracker.update(detections)
            tracking_time = time.time() - tracking_start
            
            # Create visualization
            vis_frame = visualize_tracking(frame, tracked_objects, text_queries)
            
            # Write frame to output video
            out.write(vis_frame)
            frame_count += 1
            
            # Update timing stats
            timing_stats['detection_time'] += detection_time
            timing_stats['tracking_time'] += tracking_time
            timing_stats['total_frames'] += 1
            
            # Update progress bar
            avg_detection = timing_stats['detection_time'] / timing_stats['total_frames']
            avg_tracking = timing_stats['tracking_time'] / timing_stats['total_frames']
            avg_total = avg_detection + avg_tracking
            
            pbar.set_postfix({
                'Det': f'{avg_detection*1000:.1f}ms',
                'Track': f'{avg_tracking*1000:.1f}ms',
                'Total': f'{avg_total*1000:.1f}ms',
                'Tracks': len(tracked_objects)
            })
            pbar.update(1)
    
    # Release everything
    cap.release()
    out.release()
    
    # Print final statistics
    print(f"\nTracking Demo Complete!")
    print(f"Processed {frame_count} frames")
    print(f"Average detection time: {timing_stats['detection_time']/frame_count*1000:.1f}ms")
    print(f"Average tracking time: {timing_stats['tracking_time']/frame_count*1000:.1f}ms")
    print(f"Average total time: {(timing_stats['detection_time']+timing_stats['tracking_time'])/frame_count*1000:.1f}ms")
    print(f"Output saved to: {output_path}")
    
    # Print tracking metrics
    metrics = tracker.get_metrics()
    print(f"\nTracking Metrics:")
    print(f"Total tracks created: {metrics.get('total_tracks', 0)}")
    print(f"Active tracks: {metrics.get('active_tracks', 0)}")
    print(f"Average track lifetime: {metrics.get('avg_track_lifetime', 0):.1f} frames")


def visualize_tracking(frame, tracked_objects, text_queries):
    """Visualize tracking results on frame."""
    vis_frame = frame.copy()
    
    # Define colors for different tracks
    colors = [
        (255, 107, 107),  # Red
        (78, 205, 196),   # Cyan
        (69, 183, 209),   # Blue
        (150, 206, 180),  # Green
        (255, 234, 167),  # Yellow
        (221, 160, 221),  # Plum
        (152, 216, 200),  # Mint
        (255, 165, 0),    # Orange
        (138, 43, 226),   # Blue Violet
        (255, 20, 147)    # Deep Pink
    ]
    
    for obj in tracked_objects:
        track_id = obj['id']
        bbox = obj['bbox']
        class_name = obj['class']
        confidence = obj['confidence']
        age = obj['age']
        hits = obj['hits']
        
        # Get color for this track
        color = colors[track_id % len(colors)]
        
        # Draw bounding box
        x1, y1, x2, y2 = map(int, bbox)
        cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, 3)
        
        # Draw track ID and info
        label = f"ID:{track_id} {class_name}"
        if confidence > 0:
            label += f" ({confidence:.2f})"
        
        # Calculate text size
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2
        (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, thickness)
        
        # Draw background rectangle for text
        cv2.rectangle(vis_frame, 
                     (x1, y1 - text_height - 10), 
                     (x1 + text_width + 10, y1), 
                     color, -1)
        
        # Draw text
        cv2.putText(vis_frame, label, (x1 + 5, y1 - 5), 
                   font, font_scale, (255, 255, 255), thickness)
        
        # Draw age and hits info
        info_label = f"Age:{age} Hits:{hits}"
        cv2.putText(vis_frame, info_label, (x1, y2 + 20), 
                   font, 0.5, color, 1)
    
    # Draw frame info
    info_text = f"Tracks: {len(tracked_objects)} | Queries: {', '.join(text_queries)}"
    cv2.putText(vis_frame, info_text, (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    return vis_frame


def main():
    """Main function to run tracking demo."""
    parser = argparse.ArgumentParser(description="OVOD Tracking Demo")
    parser.add_argument("video_path", help="Path to input video file")
    parser.add_argument("text_queries", nargs="+", help="Text queries to search for")
    parser.add_argument("--output", "-o", default="output_videos/tracking_demo.mp4", 
                       help="Output video path")
    
    args = parser.parse_args()
    
    # Check if input video exists
    if not os.path.exists(args.video_path):
        print(f"Error: Input video file '{args.video_path}' not found.")
        return
    
    # Process video with tracking
    process_video_with_tracking(args.video_path, args.text_queries, args.output)


if __name__ == "__main__":
    main() 