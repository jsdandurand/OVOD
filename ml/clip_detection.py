import torch
import torchvision
from torchvision import transforms
import open_clip
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import matplotlib.pyplot as plt
import os
import time
from functools import lru_cache
from typing import List, Tuple, Optional
from ultralytics import YOLO


class ClipDetector:
    """Open Vocabulary Object Detection using YOLOv10 for detection and CLIP for classification."""
    
    def __init__(self, device: str = None):
        """Initialize the ClipTracker detector with models."""
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Initializing ClipTracker Detector on {self.device}...")
        
        # Load YOLOv10 for region proposals
        self.yolo_model = YOLO('ml/models/yolov10l.pt')  # You can use yolov10s.pt, yolov10m.pt, yolov10l.pt, or yolov10x.pt
        self.yolo_model.to(self.device)
        
        # Load OpenCLIP
        self.clip_model, _, self.clip_preprocess = open_clip.create_model_and_transforms(
            'ViT-B-32', pretrained='laion2b_s34b_b79k'
        )
        self.clip_model.eval()
        self.clip_model.to(self.device)
        
        # Get feature dimension
        self.feature_dim = self.clip_model.visual.output_dim
        
        # Hardcoded list of background prompts for weighted scoring
        self.background_prompts = ["background"]
        
        # Pre-compute embeddings for background prompts
        print("Pre-computing embeddings for background prompts...")
        self.background_embeddings = []
        for prompt in self.background_prompts:
            embedding = self._get_text_embedding(prompt)
            self.background_embeddings.append(embedding)
        
        print("Models loaded successfully!")

    @lru_cache(maxsize=100)
    def _get_text_embedding(self, text_query: str):
        """Get text embedding from cache or compute if not cached."""
        # Compute text embedding
        text_tokens = open_clip.tokenize([text_query]).to(self.device)
        with torch.no_grad():
            text_features = self.clip_model.encode_text(text_tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        return text_features

    def generate_proposals(self, image: Image.Image, confidence_threshold: float = 0.1, 
                          max_proposals: int = 300, timing_dict: dict = None):
        """Generate region proposals using YOLOv10 (ignoring its class predictions)."""
        
        # Run YOLOv10 detection with low confidence to get all possible objects
        # We use YOLO's objectness scores, not its class predictions
        yolo_start = time.time()
        results = self.yolo_model(image, conf=0.01, verbose=False)  # Very low threshold to get all detections
        yolo_inference_time = time.time() - yolo_start
        
        # Store timing if dict provided
        if timing_dict is not None:
            timing_dict['yolo_inference'] = yolo_inference_time
        
        if len(results[0].boxes) == 0:
            return np.array([]).reshape(0, 4), np.array([])
        
        # Extract boxes and confidence scores 
        yolo_boxes = results[0].boxes.xyxy.cpu().numpy()  # [x1, y1, x2, y2]
        yolo_scores = results[0].boxes.conf.cpu().numpy()  # Use confidence as "objectness" score
        
        # Filter by objectness threshold (how confident we are something IS an object)
        valid_indices = yolo_scores > confidence_threshold
        
        if not np.any(valid_indices):
            return np.array([]).reshape(0, 4), np.array([])
        
        filtered_boxes = yolo_boxes[valid_indices]
        filtered_scores = yolo_scores[valid_indices]
        
        # Remove very small boxes
        areas = (filtered_boxes[:, 2] - filtered_boxes[:, 0]) * (filtered_boxes[:, 3] - filtered_boxes[:, 1])
        valid_size = areas > 32 * 32  # Minimum area
        
        if np.any(valid_size):
            filtered_boxes = filtered_boxes[valid_size]
            filtered_scores = filtered_scores[valid_size]
        
        # Sort by objectness score and limit
        if len(filtered_boxes) > max_proposals:
            sorted_indices = np.argsort(filtered_scores)[::-1][:max_proposals]
            final_boxes = filtered_boxes[sorted_indices]
            final_scores = filtered_scores[sorted_indices]
        else:
            final_boxes = filtered_boxes
            final_scores = filtered_scores
        
        return final_boxes, final_scores


    def crop_regions(self, image: Image.Image, boxes: np.ndarray) -> List[Image.Image]:
        """Crop regions from image based on bounding boxes."""
        crops = []
        for box in boxes:
            x1, y1, x2, y2 = box.astype(int)
            # Ensure coordinates are within image bounds
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(image.width, x2)
            y2 = min(image.height, y2)
            
            crop = image.crop((x1, y1, x2, y2))
            crops.append(crop)
        
        return crops


    def compute_clip_embeddings(self, crops: List[Image.Image], text_queries: List[str], timing_dict: dict = None):
        """Compute CLIP embeddings for image crops and text queries."""
        
        # Time image processing
        clip_image_start = time.time()
        image_features = []
        for crop in crops:
            if crop.size[0] > 0 and crop.size[1] > 0:  # Valid crop
                preprocessed = self.clip_preprocess(crop).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    features = self.clip_model.encode_image(preprocessed)
                    features = features / features.norm(dim=-1, keepdim=True)
                    image_features.append(features)
            else:
                # Handle invalid crops
                image_features.append(torch.zeros(1, self.feature_dim).to(self.device))
        
        image_features = torch.cat(image_features, dim=0)
        clip_image_time = time.time() - clip_image_start
        
        # Time text processing
        clip_text_start = time.time()
        text_features = []
        for text_query in text_queries:
            text_feat = self._get_text_embedding(text_query)
            text_features.append(text_feat)
        
        text_features = torch.cat(text_features, dim=0)
        clip_text_time = time.time() - clip_text_start
        
        # Store timing if dict provided
        if timing_dict is not None:
            timing_dict['clip_image'] = clip_image_time
            timing_dict['clip_text'] = clip_text_time
        
        return image_features, text_features


    def compute_weighted_similarities(self, image_features: torch.Tensor, text_features: torch.Tensor,
                                    target_weight: float = 1.0, background_weight: float = -0.4):
        """Compute weighted similarities considering target prompts and background prompts.
        
        Args:
            image_features: Image feature embeddings
            text_features: Target text feature embeddings
            target_weight: Weight for target prompt similarities (positive)
            background_weight: Weight for background prompt similarities (negative)
            
        Returns:
            np.ndarray: Weighted similarity matrix of shape (num_boxes, num_queries)
        """
        # Compute similarities with target prompts
        target_similarities = (image_features @ text_features.T).cpu().numpy()
        
        # Compute similarities with background prompts and take average
        background_similarities = []
        for bg_embedding in self.background_embeddings:
            bg_sim = (image_features @ bg_embedding.T).cpu().numpy()
            background_similarities.append(bg_sim)
        
        # mean similarity across all background prompts
        mean_background_similarities = np.mean(background_similarities, axis=0)
        
        # Apply weighted scoring: high target similarity + low background similarity
        weighted_similarities = (target_weight * target_similarities + 
                               background_weight * mean_background_similarities)
        
        return weighted_similarities

    def compute_similarities(self, image_features: torch.Tensor, text_features: torch.Tensor):
        """Compute cosine similarities between image and text features.
        
        Returns:
            np.ndarray: Similarity matrix of shape (num_boxes, num_queries)
        """
        similarities = (image_features @ text_features.T)
        return similarities.cpu().numpy()


    def compute_iou(self, box1: np.ndarray, box2: np.ndarray) -> float:
        """Compute Intersection over Union (IoU) of two bounding boxes."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        # No intersection
        if x2 <= x1 or y2 <= y1:
            return 0.0
        
        # Intersection area
        intersection = (x2 - x1) * (y2 - y1)
        
        # Union area
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0

    def classify_boxes(self, similarities: np.ndarray, text_queries: List[str], similarity_threshold: float = 0.25):
        """Classify boxes based on highest similarity with text queries.
        
        Args:
            similarities: Similarity matrix of shape (num_boxes, num_queries)
            text_queries: List of text queries
            similarity_threshold: Minimum similarity threshold
            
        Returns:
            dict: For each query, returns (boxes_indices, similarities) that belong to that class
        """
        # Find the query with highest similarity for each box
        max_similarities = np.max(similarities, axis=1)
        best_query_indices = np.argmax(similarities, axis=1)
        
        # Filter by similarity threshold
        valid_boxes = max_similarities > similarity_threshold
        
        # Group boxes by their assigned class
        results = {}
        for i, query in enumerate(text_queries):
            # Find boxes assigned to this query that meet threshold
            query_boxes = valid_boxes & (best_query_indices == i)
            box_indices = np.where(query_boxes)[0]
            box_similarities = similarities[box_indices, i] if len(box_indices) > 0 else np.array([])
            
            results[query] = (box_indices, box_similarities)
        
        return results
    
    def apply_nms(self, boxes: np.ndarray, scores: np.ndarray, similarities: np.ndarray,
                  iou_threshold: float = 0.5):
        """Apply Non-Maximum Suppression to remove overlapping detections using weighted scoring.
        
        Args:
            boxes: Bounding boxes
            scores: Objectness scores
            similarities: Weighted similarity scores (high target similarity + low background similarity)
            iou_threshold: IoU threshold for suppression
        """
        if len(boxes) == 0:
            return boxes, scores, similarities
        
        # Sort by weighted similarity scores (highest first)
        # Higher scores = high similarity with target + low similarity with background
        sorted_indices = np.argsort(similarities)[::-1]
        
        keep_indices = []
        
        while len(sorted_indices) > 0:
            # Take the detection with highest weighted similarity
            current_idx = sorted_indices[0]
            keep_indices.append(current_idx)
            
            if len(sorted_indices) == 1:
                break
                
            # Compute IoU with remaining detections
            current_box = boxes[current_idx]
            remaining_indices = sorted_indices[1:]
            
            ious = np.array([
                self.compute_iou(current_box, boxes[idx]) 
                for idx in remaining_indices
            ])
            
            # Keep only detections with IoU below threshold
            keep_mask = ious < iou_threshold
            sorted_indices = remaining_indices[keep_mask]
        
        keep_indices = np.array(keep_indices)
        return boxes[keep_indices], scores[keep_indices], similarities[keep_indices]

    def filter_detections(self, boxes: np.ndarray, scores: np.ndarray, similarities: np.ndarray,
                         text_queries: List[str], similarity_threshold: float = 0.25, top_k: int = 5, 
                         iou_threshold: float = 0.5, apply_nms: bool = True):
        """Filter detections based on similarity threshold and keep top-k per class."""
        # Classify boxes by their best matching query
        class_results = self.classify_boxes(similarities, text_queries, similarity_threshold)
        
        # Collect all final results
        all_final_boxes = []
        all_final_scores = []
        all_final_similarities = []
        all_final_classes = []
        
        # Process each class separately
        for query, (box_indices, box_similarities) in class_results.items():
            if len(box_indices) == 0:
                continue
                
            # Get boxes and scores for this class
            class_boxes = boxes[box_indices]
            class_scores = scores[box_indices]
            class_similarities = box_similarities
            
            # Apply Non-Maximum Suppression within this class only
            if apply_nms and len(class_boxes) > 1:
                class_boxes, class_scores, class_similarities = self.apply_nms(
                    class_boxes, class_scores, class_similarities, iou_threshold
                )
            
            # Sort by similarity and take top-k for this class
            if len(class_boxes) > top_k:
                sorted_indices = np.argsort(class_similarities)[::-1][:top_k]
                class_boxes = class_boxes[sorted_indices]
                class_scores = class_scores[sorted_indices]
                class_similarities = class_similarities[sorted_indices]
            
            # Add to final results with class labels
            if len(class_boxes) > 0:
                all_final_boxes.append(class_boxes)
                all_final_scores.append(class_scores)
                all_final_similarities.append(class_similarities)
                all_final_classes.extend([query] * len(class_boxes))
        
        # Combine all classes
        if len(all_final_boxes) > 0:
            final_boxes = np.vstack(all_final_boxes)
            final_scores = np.concatenate(all_final_scores)
            final_similarities = np.concatenate(all_final_similarities)
            final_classes = all_final_classes
        else:
            final_boxes = np.array([]).reshape(0, 4)
            final_scores = np.array([])
            final_similarities = np.array([])
            final_classes = []
        
        return final_boxes, final_scores, final_similarities, final_classes


    def visualize_detections_to_pil(self, image: Image.Image, boxes: np.ndarray, similarities: np.ndarray,
                                   classes: List[str], text_queries):
        """Visualize detections and return PIL image (for video processing)."""
        # Convert single query to list for uniform handling
        if isinstance(text_queries, str):
            text_queries = [text_queries]
        
        # Create a copy for drawing
        vis_image = image.copy()
        draw = ImageDraw.Draw(vis_image)
        
        # Try to load a font, fallback to default if not available
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
        except:
            font = ImageFont.load_default()
        
        # Colors for different classes
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'cyan', 'magenta', 'yellow']
        class_colors = {}
        for i, query in enumerate(text_queries):
            class_colors[query] = colors[i % len(colors)]
        
        for i, (box, similarity, cls) in enumerate(zip(boxes, similarities, classes)):
            x1, y1, x2, y2 = box.astype(int)
            color = class_colors.get(cls, 'red')
            
            # Draw bounding box
            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
            
            # Draw class and similarity score
            label = f"{cls}: {similarity:.3f}"
            bbox = draw.textbbox((x1, y1-25), label, font=font)
            draw.rectangle(bbox, fill=color)
            draw.text((x1, y1-25), label, fill='white', font=font)
        
        # Add title
        if len(text_queries) == 1:
            title = f"Query: '{text_queries[0]}' | Detections: {len(boxes)}"
        else:
            title = f"Queries: {len(text_queries)} | Detections: {len(boxes)}"
        draw.text((10, 10), title, fill='white', font=font)
        
        return vis_image

    def visualize_detections(self, image: Image.Image, boxes: np.ndarray, similarities: np.ndarray,
                            classes: List[str], text_queries, output_path: str):
        """Visualize detections with bounding boxes and similarity scores."""
        vis_image = self.visualize_detections_to_pil(image, boxes, similarities, classes, text_queries)
        
        # Save the result
        vis_image.save(output_path)
        print(f"Visualization saved to: {output_path}")
        
        return vis_image


    def detect(self, image: Image.Image, text_queries, 
               confidence_threshold: float = 0.1,
               similarity_threshold: float = 0.25,
               max_proposals: int = 300,
               top_k: int = 5,
               iou_threshold: float = 0.5,
               apply_nms: bool = True,
               use_weighted_scoring: bool = True,
               target_weight: float = 1.0,
               background_weight: float = -0.4,
               verbose: bool = True,
               debug: bool = False,
               timing_dict: dict = None):
        """Perform open-vocabulary object detection on an image.
        
        Args:
            image: PIL Image to process
            text_queries: Text descriptions of objects to find (list or single string)
            confidence_threshold: Minimum confidence for region proposals
            similarity_threshold: Minimum CLIP similarity for final detections
            max_proposals: Maximum number of region proposals to consider
            top_k: Maximum number of final detections to return per class
            iou_threshold: IoU threshold for NMS (higher = more aggressive removal)
            apply_nms: Whether to apply Non-Maximum Suppression to remove overlaps
            use_weighted_scoring: Whether to use weighted scoring considering background prompts
            target_weight: Weight for target prompt similarities (positive)
            background_weight: Weight for background prompt similarities (negative)
            verbose: Whether to print progress messages
            debug: Whether to print timing breakdown information
            
        Returns:
            tuple: (boxes, scores, similarities, classes) arrays for detected objects
        """
        # Convert single query to list for uniform handling
        if isinstance(text_queries, str):
            text_queries = [text_queries]
        
        start_time = time.time()
        # Use external timing_dict if provided, otherwise create internal one
        internal_timing = timing_dict if timing_dict is not None else {}
        
        # YOLOv10 inference + post-processing (done once for all queries)
        if verbose:
            print("Generating region proposals...")
        yolo_total_start = time.time()
        boxes, proposal_scores = self.generate_proposals(image, confidence_threshold, max_proposals, internal_timing)
        yolo_total_time = time.time() - yolo_total_start
        
        if verbose:
            print(f"Generated {len(boxes)} region proposals")
        
        if len(boxes) == 0:
            if verbose:
                print("No region proposals found!")
            return np.array([]), np.array([]), np.array([]), []
        
        # Other processing start time (everything except model inference)
        other_start = time.time()
        
        # Crop regions
        if verbose:
            print("Cropping regions...")
        crops = self.crop_regions(image, boxes)
        
        # CLIP inference (compute similarities with all queries)
        if verbose:
            print(f"Computing CLIP embeddings for {len(text_queries)} queries...")
        image_features, text_features = self.compute_clip_embeddings(crops, text_queries, internal_timing)
        
        # Compute similarities (matrix: boxes x queries)
        if verbose:
            print("Computing similarities...")
        
        if use_weighted_scoring:
            if verbose:
                print(f"Using weighted similarity scoring with background prompts: {self.background_prompts}")
            similarities = self.compute_weighted_similarities(
                image_features, text_features, target_weight, background_weight
            )
        else:
            similarities = self.compute_similarities(image_features, text_features)
        
        # Filter detections (per class)
        if verbose:
            print("Filtering detections per class...")
        final_boxes, final_scores, final_similarities, final_classes = self.filter_detections(
            boxes, proposal_scores, similarities, text_queries, similarity_threshold, top_k, 
            iou_threshold, apply_nms
        )
        
        # Calculate timing components
        total_time = time.time() - start_time
        yolo_inference_time = internal_timing.get('yolo_inference', 0)
        clip_image_time = internal_timing.get('clip_image', 0)
        clip_text_time = internal_timing.get('clip_text', 0)
        
        # Other time includes: YOLO post-processing + cropping + similarities + filtering
        yolo_postprocess_time = yolo_total_time - yolo_inference_time
        other_time = time.time() - other_start - clip_image_time - clip_text_time + yolo_postprocess_time
        
        # Store "other" timing in the timing dict
        internal_timing['other'] = other_time
        
        # Print timing breakdown
        if debug:
            yolo_pct = (yolo_inference_time / total_time) * 100
            clip_img_pct = (clip_image_time / total_time) * 100
            clip_txt_pct = (clip_text_time / total_time) * 100
            other_pct = (other_time / total_time) * 100
            print(f"Timing: YOLO {yolo_pct:.1f}%, CLIP-img {clip_img_pct:.1f}%, CLIP-txt {clip_txt_pct:.1f}%, Other {other_pct:.1f}% ({total_time:.3f}s total)")
        
        if verbose:
            print(f"Found {len(final_boxes)} matching detections")
            if len(final_boxes) > 0:
                for i, (box, sim, cls) in enumerate(zip(final_boxes, final_similarities, final_classes)):
                    x1, y1, x2, y2 = box
                    print(f"Detection {i+1}: '{cls}' Box=({x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}), Similarity={sim:.3f}")
        
        return final_boxes, final_scores, final_similarities, final_classes
    
    def detect_and_visualize(self, image_path: str, text_queries, 
                            confidence_threshold: float = 0.1,
                            similarity_threshold: float = 0.25,
                            max_proposals: int = 300,
                            top_k: int = 5,
                            iou_threshold: float = 0.5,
                            apply_nms: bool = True,
                            use_weighted_scoring: bool = True,
                            target_weight: float = 1.0,
                            background_weight: float = -0.4,
                            debug: bool = False,
                            output_path: Optional[str] = None):
        """Detect objects and save visualization."""
        # Convert single query to list for uniform handling
        if isinstance(text_queries, str):
            text_queries = [text_queries]
        
        # Load image
        print(f"Loading image: {image_path}")
        image = Image.open(image_path).convert('RGB')
        
        # Perform detection
        final_boxes, final_scores, final_similarities, final_classes = self.detect(
            image, text_queries, confidence_threshold, similarity_threshold, 
            max_proposals, top_k, iou_threshold, apply_nms, use_weighted_scoring,
            target_weight, background_weight, verbose=True, debug=debug
        )
        
        if len(final_boxes) > 0:
            # Visualize
            if output_path is None:
                base_name = os.path.splitext(os.path.basename(image_path))[0]
                if len(text_queries) == 1:
                    query_clean = text_queries[0].replace(' ', '_').replace('/', '_')
                    output_path = f"{base_name}_{query_clean}_detections.jpg"
                else:
                    output_path = f"{base_name}_multiclass_detections.jpg"
            
            print("Creating visualization...")
            self.visualize_detections(image, final_boxes, final_similarities, final_classes, text_queries, output_path)
        else:
            queries_str = "', '".join(text_queries)
            print(f"No objects matching '{queries_str}' found with similarity > {similarity_threshold}")
        
        return final_boxes, final_scores, final_similarities, final_classes


def main():
    """Example usage of the detection system."""
    # Example parameters - modify these for your use case
    image_path = "data/object_detection/cat.jpg"  # Change to your image path
    text_queries = ["cat"]  # Change to your query
    
    # Check if example image exists
    if not os.path.exists(image_path):
        print(f"Image file '{image_path}' not found!")
        print("Please provide a valid image path in the main() function.")
        return
    
    # Initialize detector
    detector = ClipDetector()
    
    # Run detection with visualization using weighted scoring
    print("Running detection with weighted scoring (considering background prompts)...")
    detector.detect_and_visualize(
        image_path=image_path,
        text_queries=text_queries,
        confidence_threshold=0.1,
        similarity_threshold=0.25,
        max_proposals=300,
        debug=True,
        top_k=5,
        use_weighted_scoring=True,
        target_weight=1.0,      # Positive weight for target prompt
        background_weight=-0.4   # Negative weight for background prompts
    )
    
    # You can also run without weighted scoring for comparison
    print("\nRunning detection without weighted scoring...")
    detector.detect_and_visualize(
        image_path=image_path,
        text_queries=text_queries,
        confidence_threshold=0.1,
        similarity_threshold=0.25,
        max_proposals=300,
        debug=True,
        top_k=5,
        use_weighted_scoring=False  # Use traditional similarity scoring
    )


if __name__ == "__main__":
    main() 