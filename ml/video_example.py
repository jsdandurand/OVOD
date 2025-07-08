"""
Example of using ClipTracker classes for video processing.
This demonstrates the benefit of the class-based approach where models are loaded once
and reused across multiple frames.
"""

import cv2
import numpy as np
from PIL import Image
from .clip_detection import ClipDetector
from .clip_segmentation import ClipSegmenter
import os
import time
from tqdm import tqdm


def process_video_detection(video_path: str, text_queries, output_path: str, 
                           sample_rate: int = 30):
    """Process video for object detection using ClipTracker and save as video."""
    # Convert single query to list for uniform handling
    if isinstance(text_queries, str):
        text_queries = [text_queries]
    
    # Initialize detector once (models loaded here)
    print("Initializing ClipTracker Detector...")
    detector = ClipDetector()
    
    # Open input video
    cap = cv2.VideoCapture(video_path)
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Create output directory if needed
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Define the codec and create VideoWriter - same FPS as input
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    processed_count = 0
    
    # Cache for last detection results
    last_boxes = []
    last_similarities = []
    last_classes = []
    
    print(f"Processing video: {video_path}")
    if len(text_queries) == 1:
        print(f"Looking for: {text_queries[0]}")
    else:
        print(f"Looking for: {text_queries}")
    print(f"Output video: {output_path}")
    print(f"FPS: {fps} (same as input)")
    print(f"Detection sample rate: every {sample_rate} frames")
    
    # Initialize timing tracking
    start_time = time.time()
    timing_stats = {
        'yolo_inference': 0.0,
        'clip_image': 0.0, 
        'clip_text': 0.0,
        'other': 0.0,
        'total_inference_time': 0.0
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
            
            # Check if we should run detection on this frame
            if frame_count % sample_rate == 0:
                # Track timing for this detection
                timing_dict = {}
                
                # Run detection on this frame
                boxes, scores, similarities, classes = detector.detect(
                    pil_image, text_queries, 
                    confidence_threshold=0.3,
                    similarity_threshold=0.23,
                    verbose=False,  # Reduce output for video processing
                    iou_threshold=0.2,
                    timing_dict=timing_dict
                )
                
                # Accumulate timing statistics
                if timing_dict:
                    timing_stats['yolo_inference'] += timing_dict.get('yolo_inference', 0)
                    timing_stats['clip_image'] += timing_dict.get('clip_image', 0)
                    timing_stats['clip_text'] += timing_dict.get('clip_text', 0)
                    timing_stats['other'] += timing_dict.get('other', 0)
                    timing_stats['total_inference_time'] += sum(timing_dict.values())
                
                # Cache the results
                last_boxes = boxes
                last_similarities = similarities
                last_classes = classes
                processed_count += 1
            
            # Create visualization using current or cached results
            if len(last_boxes) > 0:
                # Use cached detection results
                vis_image = detector.visualize_detections_to_pil(
                    pil_image, last_boxes, last_similarities, last_classes, text_queries
                )
                # Convert back to OpenCV format
                vis_frame = cv2.cvtColor(np.array(vis_image), cv2.COLOR_RGB2BGR)
            else:
                # No detections available, use original frame
                vis_frame = frame
            
            # Write frame to output video
            out.write(vis_frame)
            frame_count += 1
            
            # Update progress bar with timing statistics
            if timing_stats['total_inference_time'] > 0:
                yolo_pct = (timing_stats['yolo_inference'] / timing_stats['total_inference_time']) * 100
                clip_img_pct = (timing_stats['clip_image'] / timing_stats['total_inference_time']) * 100
                clip_txt_pct = (timing_stats['clip_text'] / timing_stats['total_inference_time']) * 100
                other_pct = (timing_stats['other'] / timing_stats['total_inference_time']) * 100
                
                pbar.set_postfix({
                    'Y': f'{yolo_pct:.1f}%',
                    'CI': f'{clip_img_pct:.1f}%', 
                    'CT': f'{clip_txt_pct:.1f}%',
                    'O': f'{other_pct:.1f}%',
                    'Dets': processed_count
                })
            else:
                pbar.set_postfix({'Dets': processed_count})
            pbar.update(1)
    
    # Release everything
    cap.release()
    out.release()
    
    print(f"Video processing complete!")
    print(f"Processed {processed_count} frames with detections out of {frame_count} total frames")
    print(f"Output saved to: {output_path}")


def process_video_segmentation(video_path: str, text_queries, output_path: str, 
                              sample_rate: int = 30, apply_filtering: bool = True):
    """Process video for instance segmentation using ClipTracker and save as video.
    
    Args:
        apply_filtering: Whether to apply full filtering (NMS, merging, top-k) or just similarity threshold
    """
    # Convert single query to list for uniform handling
    if isinstance(text_queries, str):
        text_queries = [text_queries]
    
    # Initialize segmenter once (models loaded here)
    print("Initializing ClipTracker Segmenter...")
    segmenter = ClipSegmenter()
    
    # Open input video
    cap = cv2.VideoCapture(video_path)
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Create output directory if needed
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Define the codec and create VideoWriter - same FPS as input
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    processed_count = 0
    
    # Cache for last segmentation results
    last_boxes = []
    last_masks = []
    last_similarities = []
    last_classes = []
    
    print(f"Processing video: {video_path}")
    if len(text_queries) == 1:
        print(f"Looking for: {text_queries[0]}")
    else:
        print(f"Looking for: {text_queries}")
    print(f"Output video: {output_path}")
    print(f"FPS: {fps} (same as input)")
    print(f"Segmentation sample rate: every {sample_rate} frames")
    
    # Initialize timing tracking
    start_time = time.time()
    timing_stats = {
        'yolo_inference': 0.0,
        'clip_image': 0.0, 
        'clip_text': 0.0,
        'other': 0.0,
        'total_inference_time': 0.0
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
            
            # Check if we should run segmentation on this frame
            if frame_count % sample_rate == 0:
                # Track timing for this segmentation
                timing_dict = {}
                
                # Run segmentation on this frame
                boxes, scores, masks, similarities, classes = segmenter.segment(
                    pil_image, text_queries, 
                    confidence_threshold=0.1,
                    similarity_threshold=0.20,
                    merge_overlaps=True,  # Use merge_overlaps
                    verbose=False, # Reduce output for video processing
                    iou_threshold=0.2,
                    apply_filtering=apply_filtering,
                    timing_dict=timing_dict
                )
                
                # Accumulate timing statistics
                if timing_dict:
                    timing_stats['yolo_inference'] += timing_dict.get('yolo_inference', 0)
                    timing_stats['clip_image'] += timing_dict.get('clip_image', 0)
                    timing_stats['clip_text'] += timing_dict.get('clip_text', 0)
                    timing_stats['other'] += timing_dict.get('other', 0)
                    timing_stats['total_inference_time'] += sum(timing_dict.values())
                
                # Cache the results
                last_boxes = boxes
                last_masks = masks
                last_similarities = similarities
                last_classes = classes
                processed_count += 1
            
            # Create visualization using current or cached results
            if len(last_boxes) > 0:
                # Use cached segmentation results
                vis_image = segmenter.visualize_segmentations_to_pil(
                    pil_image, last_boxes, last_masks, last_similarities, last_classes, text_queries
                )
                # Convert back to OpenCV format
                vis_frame = cv2.cvtColor(np.array(vis_image), cv2.COLOR_RGB2BGR)
            else:
                # No segmentations available, use original frame
                vis_frame = frame
            
            # Write frame to output video
            out.write(vis_frame)
            frame_count += 1
            
            # Update progress bar with timing statistics
            if timing_stats['total_inference_time'] > 0:
                yolo_pct = (timing_stats['yolo_inference'] / timing_stats['total_inference_time']) * 100
                clip_img_pct = (timing_stats['clip_image'] / timing_stats['total_inference_time']) * 100
                clip_txt_pct = (timing_stats['clip_text'] / timing_stats['total_inference_time']) * 100
                other_pct = (timing_stats['other'] / timing_stats['total_inference_time']) * 100
                
                pbar.set_postfix({
                    'Y': f'{yolo_pct:.1f}%',
                    'CI': f'{clip_img_pct:.1f}%', 
                    'CT': f'{clip_txt_pct:.1f}%',
                    'O': f'{other_pct:.1f}%',
                    'Segs': processed_count
                })
            else:
                pbar.set_postfix({'Segs': processed_count})
            pbar.update(1)
    
    # Release everything
    cap.release()
    out.release()
    
    print(f"Video processing complete!")
    print(f"Processed {processed_count} frames with segmentations out of {frame_count} total frames")
    print(f"Output saved to: {output_path}")


def batch_process_images(image_dir: str, text_queries, output_dir: str, mode: str = "detection"):
    """Process a batch of images efficiently using ClipTracker classes."""
    # Convert single query to list for uniform handling
    if isinstance(text_queries, str):
        text_queries = [text_queries]
    
    if mode == "detection":
        processor = ClipDetector()
        process_func = processor.detect
        viz_func = processor.visualize_detections
        ext = ".jpg"
    else:  # segmentation
        processor = ClipSegmenter()
        process_func = processor.segment
        viz_func = processor.visualize_segmentations
        ext = ".png"
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all image files
    image_files = [f for f in os.listdir(image_dir) 
                   if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
    
    if len(text_queries) == 1:
        print(f"Processing {len(image_files)} images for '{text_queries[0]}' using {mode}")
    else:
        print(f"Processing {len(image_files)} images for {text_queries} using {mode}")
    
    for i, image_file in enumerate(image_files):
        image_path = os.path.join(image_dir, image_file)
        image = Image.open(image_path).convert('RGB')
        
        # Process image
        if mode == "detection":
            boxes, scores, similarities, classes = process_func(
                image, text_queries, verbose=False
            )
            if len(boxes) > 0:
                output_path = os.path.join(output_dir, f"{os.path.splitext(image_file)[0]}_det{ext}")
                viz_func(image, boxes, similarities, classes, text_queries, output_path)
        else:
            boxes, scores, masks, similarities, classes = process_func(
                image, text_queries, use_mask_iou=True, verbose=False
            )
            if len(boxes) > 0:
                output_path = os.path.join(output_dir, f"{os.path.splitext(image_file)[0]}_seg{ext}")
                viz_func(image, boxes, masks, similarities, classes, text_queries, output_path)
        
        if (i + 1) % 10 == 0:
            print(f"Processed {i + 1}/{len(image_files)} images")
    
    print("Batch processing complete!")


def main():
    """Example usage for video and batch processing."""
    
    # Example 1: Single-class Video Detection
    # Uncomment and modify paths as needed

    # process_video_detection(
    #     video_path="data/object_detection/dog.MOV",
    #     text_queries="dog",  # Single class
    #     output_path="output_videos/dog_detections.mp4",
    #     sample_rate=2 # Process every 2 frames
    # )

    
    # Example 2: Multi-class Video Detection
    # Uncomment and modify paths as needed

    process_video_segmentation(
        video_path="data/object_detection/dog.MOV",
        text_queries=["dog"],  # Single class segmentation
        output_path="output_videos/dog_segmentations.mp4",
        sample_rate=1,
        apply_filtering=False  # Test without filtering to check performance
    )
    
    
    # Example 3: Single-class Video Segmentation
    # Uncomment and modify paths as needed

    # process_video_segmentation(
    #     video_path="data/object_detection/dog.MOV",
    #     text_queries="dog",  # Single class
    #     output_path="output_videos/dog_segmentations.mp4",
    #     sample_rate=1  # Process every frame
    # )

    
    # Example 4: Multi-class Video Segmentation
    # Uncomment and modify paths as needed
    """
    process_video_segmentation(
        video_path="data/object_detection/dog.MOV",
        text_queries=["dog", "person", "grass"],  # Multiple classes
        output_path="output_videos/multiclass_segmentations.mp4",
        sample_rate=2
    )
    """
    
    # Example 5: Batch Image Processing (still outputs individual images)
    # Uncomment and modify paths as needed
    """
    batch_process_images(
        image_dir="path/to/your/images",
        text_queries=["cat", "dog", "person"],  # Multiple classes
        output_dir="batch_detections",
        mode="detection"  # or "segmentation"
    )
    """
    
    print("Multi-class video processing examples ready!")
    print("Video outputs will be saved as MP4 files with visualizations.")
    print("Output videos maintain same FPS as input - sample_rate controls detection frequency.")
    print("Bounding boxes from last detection are reused on non-sampled frames.")
    print("Supports both single-class and multi-class detection/segmentation.")
    print("Each class gets its own color for visualization.")
    print("Uncomment the relevant sections and update paths to use.")
    
    # # Single and Multi-class image examples that should work
    # if os.path.exists("data/object_detection/cat.jpg"):
    #     print("\nRunning single and multi-class image examples...")
        
    #     # Single-class detection example
    #     detector = ClipDetector()
    #     image = Image.open("data/object_detection/cat.jpg").convert('RGB')
    #     boxes, scores, similarities, classes = detector.detect(image, "cat")
    #     print(f"Single-class detection found {len(boxes)} objects")
        
    #     # Multi-class detection example
    #     boxes, scores, similarities, classes = detector.detect(image, ["cat", "grass", "plant"])
    #     print(f"Multi-class detection found {len(boxes)} objects across {len(set(classes))} classes")
        
    #     # Single-class segmentation example
    #     segmenter = ClipSegmenter()
    #     boxes, scores, masks, similarities, classes = segmenter.segment(image, "cat", use_mask_iou=True)
    #     print(f"Single-class segmentation found {len(boxes)} objects")
        
    #     # Multi-class segmentation example
    #     boxes, scores, masks, similarities, classes = segmenter.segment(image, ["cat", "grass", "plant"], use_mask_iou=True)
    #     print(f"Multi-class segmentation found {len(boxes)} objects across {len(set(classes))} classes")


if __name__ == "__main__":
    main() 