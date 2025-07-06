"""
Temporal OVOD: Combined temporal tracking and open vocabulary segmentation for videos.
Reduces flickering and improves temporal consistency in video object detection/segmentation.
"""

import cv2
import numpy as np
from PIL import Image
from typing import List, Dict, Optional
import os
import time
from tqdm import tqdm

from ovod_segmentation import OVODSegmenter
from temporal_tracker import TemporalTracker, TrackerConfig


class TemporalOVODSegmenter:
    """Combined temporal tracking and OVOD segmentation for consistent video results."""
    
    def __init__(self, tracker_config: TrackerConfig = None, device: str = None):
        """Initialize the temporal OVOD segmenter."""
        self.segmenter = OVODSegmenter(device=device)
        self.tracker = TemporalTracker(tracker_config or TrackerConfig())
        self.class_to_id = {}  # Map class names to IDs
        self.next_class_id = 0
    
    def _get_class_id(self, class_name: str) -> int:
        """Get or create class ID for a class name."""
        if class_name not in self.class_to_id:
            self.class_to_id[class_name] = self.next_class_id
            self.next_class_id += 1
        return self.class_to_id[class_name]
    
    def _segmenter_to_tracker_format(self, boxes: np.ndarray, scores: np.ndarray, 
                                   masks: np.ndarray, similarities: np.ndarray, 
                                   classes: List[str]) -> List[Dict]:
        """Convert segmenter output to tracker input format."""
        detections = []
        for i, (box, score, mask, similarity, class_name) in enumerate(zip(boxes, scores, masks, similarities, classes)):
            detection = {
                'box': box.tolist(),
                'mask': mask[0] if mask.ndim == 3 else mask,  # Ensure 2D mask
                'similarity': float(similarity),
                'class_id': self._get_class_id(class_name),
                'class_name': class_name,
                'score': float(score)
            }
            detections.append(detection)
        return detections
    
    def _tracker_to_display_format(self, stable_detections: List[Dict]) -> tuple:
        """Convert tracker output to display format."""
        if not stable_detections:
            return np.array([]), np.array([]), np.array([]), np.array([]), []
        
        boxes = np.array([det['box'] for det in stable_detections])
        masks = np.array([det['mask'] for det in stable_detections])
        similarities = np.array([det['similarity'] for det in stable_detections])
        classes = [det['class_name'] for det in stable_detections]
        scores = np.array([det.get('score', det['similarity']) for det in stable_detections])
        
        # Ensure masks have the right shape for visualization
        if masks.ndim == 3 and masks.shape[0] > 0:
            # Add channel dimension if needed
            masks = masks[:, np.newaxis, :, :]
        elif masks.ndim == 2:
            # Single mask case
            masks = masks[np.newaxis, np.newaxis, :, :]
        
        return boxes, scores, masks, similarities, classes
    
    def process_frame(self, image: Image.Image, text_queries: List[str], frame_number: int,
                     confidence_threshold: float = 0.2,
                     similarity_threshold: float = 0.25,
                     verbose: bool = False,
                     debug: bool = False) -> tuple:
        """Process a single frame with temporal tracking."""
        # Run segmentation
        boxes, scores, masks, similarities, classes = self.segmenter.segment(
            image=image,
            text_queries=text_queries,
            confidence_threshold=confidence_threshold,
            similarity_threshold=similarity_threshold,
            verbose=verbose,
            debug=debug,
            iou_threshold=0.01,
            merge_overlaps=True,  # Use merge for better temporal consistency
            use_mask_iou=True     # Use mask IoU for better tracking
        )
        
        # Convert to tracker format
        detections = self._segmenter_to_tracker_format(boxes, scores, masks, similarities, classes)
        
        # Update tracker
        stable_detections = self.tracker.update(detections, frame_number)
        
        # Convert back to display format
        return self._tracker_to_display_format(stable_detections)
    
    def visualize_frame(self, image: Image.Image, boxes: np.ndarray, masks: np.ndarray, 
                       similarities: np.ndarray, classes: List[str], text_queries: List[str]) -> Image.Image:
        """Visualize a frame with tracking results."""
        return self.segmenter.visualize_segmentations_to_pil(
            image, boxes, masks, similarities, classes, text_queries
        )
    
    def get_tracking_stats(self) -> Dict:
        """Get current tracking statistics."""
        active_tracks = len(self.tracker.tracks)
        track_info = []
        for track in self.tracker.tracks:
            track_info.append({
                'id': track.track_id,
                'class': track.class_name,
                'confidence': track.smoothed_confidence,
                'frames_since_detection': self.tracker.current_frame - track.last_detection_frame,
                'displayed': track.currently_displayed,
                'confirmed': track.is_confirmed,
                'consecutive_detections': track.consecutive_detections
            })
        
        return {
            'active_tracks': active_tracks,
            'tracks': track_info,
            'cache_info': self.segmenter._get_text_embedding.cache_info()._asdict()
        }


def process_video_with_tracking(video_path: str, text_queries: List[str], output_path: str,
                              confidence_threshold: float = 0.2,
                              similarity_threshold: float = 0.25,
                              tracker_config: TrackerConfig = None,
                              device: str = None,
                              verbose: bool = True,
                              debug: bool = False):
    """Process video with temporal tracking for consistent segmentation."""
    
    # Convert single query to list for uniform handling
    if isinstance(text_queries, str):
        text_queries = [text_queries]
    
    print("Initializing Temporal OVOD Segmenter...")
    temporal_segmenter = TemporalOVODSegmenter(tracker_config=tracker_config, device=device)
    
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
    print(f"Total frames: {total_frames}")
    print("Processing every frame with temporal tracking...")
    
    # Initialize timing
    start_time = time.time()
    total_detections = 0
    
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
            
            # Process frame with temporal tracking
            boxes, scores, masks, similarities, classes = temporal_segmenter.process_frame(
                image=pil_image,
                text_queries=text_queries,
                frame_number=frame_count,
                confidence_threshold=confidence_threshold,
                similarity_threshold=similarity_threshold,
                verbose=False,  # Reduce output for video processing
                debug=debug and frame_count % 30 == 0  # Debug only occasionally
            )
            
            total_detections += len(boxes)
            
            # Create visualization
            if len(boxes) > 0:
                vis_image = temporal_segmenter.visualize_frame(
                    pil_image, boxes, masks, similarities, classes, text_queries
                )
                # Convert back to OpenCV format
                vis_frame = cv2.cvtColor(np.array(vis_image), cv2.COLOR_RGB2BGR)
            else:
                # No detections, use original frame
                vis_frame = frame
            
            # Write frame to output video
            out.write(vis_frame)
            frame_count += 1
            
            # Update progress bar
            elapsed_time = time.time() - start_time
            current_fps = frame_count / elapsed_time if elapsed_time > 0 else 0
            
            # Get tracking stats
            stats = temporal_segmenter.get_tracking_stats()
            cache_info = stats['cache_info']
            hit_rate = (cache_info['hits'] / (cache_info['hits'] + cache_info['misses'])) * 100 if (cache_info['hits'] + cache_info['misses']) > 0 else 0
            
            pbar.set_postfix({
                'FPS': f'{current_fps:.1f}',
                'Tracks': stats['active_tracks'],
                'Cache': f"{hit_rate:.1f}%",
                'Total Detections': total_detections
            })
            pbar.update(1)
    
    # Release everything
    cap.release()
    out.release()
    
    # Final statistics
    final_stats = temporal_segmenter.get_tracking_stats()
    
    print(f"\nVideo processing complete!")
    print(f"Processed {frame_count} frames")
    print(f"Total detections across all frames: {total_detections}")
    print(f"Average detections per frame: {total_detections/frame_count:.2f}")
    print(f"Active tracks at end: {final_stats['active_tracks']}")
    print(f"Cache hit rate: {hit_rate:.1f}%")
    print(f"Output saved to: {output_path}")
    
    if verbose and final_stats['tracks']:
        print("\nFinal track information:")
        for track in final_stats['tracks']:
            status = "confirmed" if track['confirmed'] else f"unconfirmed ({track['consecutive_detections']}/3)"
            print(f"  Track {track['id']}: {track['class']} ({status}, confidence: {track['confidence']:.3f}, "
                  f"frames since detection: {track['frames_since_detection']})")


def main():
    """Example usage of temporal OVOD segmentation."""
    
    # Example 1: Single class temporal segmentation
    print("Example 1: Single class temporal segmentation")
    
    # You can customize the tracker configuration
    custom_config = TrackerConfig(
        max_interpolation_frames=10,      # Track objects for up to 8 frames without detection
        confidence_decay_rate=0.99,      # Slower confidence decay (more stable)
        confirmation_frames_required=5,  # Require 3 consecutive detections before showing object
        appear_threshold=0.25,           # Lower threshold to appear
        disappear_threshold=0.1,        # Lower threshold to disappear (more sticky)
        max_predicted_movement=75        # Allow larger movements per frame
    )
    
#    if os.path.exists("data/object_detection/kitchen.mp4"):
#         process_video_with_tracking(
#             video_path="data/object_detection/kitchen.mp4",
#             text_queries="banana",  # Single class
#             output_path="output_videos/kitchen_banana_temporal.mp4",
#             confidence_threshold=0.3,
#             similarity_threshold=0.20,
#             tracker_config=custom_config,
#             verbose=True,
#             debug=True
#         )
#     else:
#         print("Kitchen video not found, skipping single class example")
    
#     print("\n" + "="*60 + "\n") 
    
    # Example 2: Multi-class temporal segmentation
    print("Example 2: Multi-class temporal segmentation")
    
    if os.path.exists("data/object_detection/kitchen.mp4"):
        process_video_with_tracking(
            video_path="data/object_detection/kitchen.mp4",
            text_queries=["banana", "knife", "computer", "paper towel", "tomato"],  # Multiple classes
            output_path="output_videos/kitchen_temporal.mp4",
            confidence_threshold=0.1,
            similarity_threshold=0.22,
            tracker_config=custom_config,
            verbose=True,
            debug=False
        )
    else:
        print("Kitchen video not found, skipping multi-class example")
    
    print("\n" + "="*60 + "\n")
    
    # Example 3: Different video with default config
    # print("Example 3: Dog video with default tracking config")
    
    # default_config = TrackerConfig()  # Use default settings
    
    # if os.path.exists("data/object_detection/dog.MOV"):
    #     process_video_with_tracking(
    #         video_path="data/object_detection/dog.MOV",
    #         text_queries=["dog", "person"],
    #         output_path="output_videos/dog_temporal.mp4",
    #         confidence_threshold=0.3,
    #         similarity_threshold=0.25,
    #         tracker_config=default_config,
    #         verbose=True
    #     )
    # else:
    #     print("Dog video not found, skipping dog example")
    
    print("\nTemporal OVOD processing examples complete!")
    print("Key benefits of temporal tracking:")
    print("- Reduced flickering in detections")
    print("- Confirmation requirement (3 consecutive frames) prevents false positive flickering")
    print("- Smooth object trajectories")
    print("- Hysteresis prevents rapid appearing/disappearing")
    print("- Motion prediction fills in missing detections")
    print("- Per-class tracking maintains object identities")


if __name__ == "__main__":
    main() 