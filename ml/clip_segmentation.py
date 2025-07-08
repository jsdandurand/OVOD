import torch
import torchvision
from torchvision import transforms
import open_clip
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import to_rgba
import os
import time
from functools import lru_cache
from typing import List, Tuple, Optional
from ultralytics import YOLO
from scipy.ndimage import zoom


class ClipSegmenter:
    """Open Vocabulary Instance Segmentation using YOLOv8-seg for segmentation and CLIP for classification."""
    
    def __init__(self, device: str = None):
        """Initialize the ClipTracker segmenter with models."""
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Initializing ClipTracker Segmenter on {self.device}...")
        
        # Load YOLOv8-seg for region proposals and segmentation masks
        self.yolo_model = YOLO('ml/models/yolov8m-seg.pt')  # You can use yolov8s-seg.pt, yolov8m-seg.pt, yolov8l-seg.pt, or yolov8x-seg.pt
        self.yolo_model.to(self.device)
        
        # Load OpenCLIP
        self.clip_model, _, self.clip_preprocess = open_clip.create_model_and_transforms(
            'ViT-B-32', pretrained='laion2b_s34b_b79k'
        )
        self.clip_model.eval()
        self.clip_model.to(self.device)
        
        # Get feature dimension
        self.feature_dim = self.clip_model.visual.output_dim
        
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

    def generate_proposals_and_masks(self, image: Image.Image, confidence_threshold: float = 0.1, 
                                   max_proposals: int = 300, verbose: bool = True, timing_dict: dict = None):
        """Generate region proposals and segmentation masks using YOLOv8-seg."""
        
        # Run YOLOv8-seg with low confidence to get all possible objects with masks
        # We use YOLO's objectness scores, not its class predictions
        yolo_start = time.time()
        results = self.yolo_model(image, conf=0.01, verbose=False)  # Very low threshold to get all detections
        yolo_inference_time = time.time() - yolo_start
        
        # Store timing if dict provided
        if timing_dict is not None:
            timing_dict['yolo_inference'] = yolo_inference_time
        
        if len(results[0].boxes) == 0 or results[0].masks is None:
            return np.array([]).reshape(0, 4), np.array([]), np.array([]).reshape(0, 1, image.height, image.width)
        
        # Extract boxes, confidence scores, and masks
        yolo_boxes = results[0].boxes.xyxy.cpu().numpy()  # [x1, y1, x2, y2] - already at original image resolution
        yolo_scores = results[0].boxes.conf.cpu().numpy()  # Use confidence as "objectness" score
        yolo_masks = results[0].masks.data.cpu().numpy()  # Actual segmentation masks - at YOLO's internal resolution
        
        # Keep masks at YOLO's internal resolution for now - will resize final ones later
        
        if verbose:
            print(f"YOLOv8-seg generated {len(yolo_boxes)} raw detections with masks")
            # Only print detailed shape info on first run to avoid spam
            if hasattr(self, '_shape_info_printed') is False:
                print(f"Image size: {image.size} (W x H)")
                print(f"Mask shape at YOLO resolution: {yolo_masks.shape}")
                if len(yolo_masks) > 0:
                    print(f"Individual mask shape: {yolo_masks[0].shape} (will resize final masks later)")
                self._shape_info_printed = True
        
        # Filter by objectness threshold (how confident we are something IS an object)
        valid_indices = yolo_scores > confidence_threshold
        
        if not np.any(valid_indices):
            return np.array([]).reshape(0, 4), np.array([]), np.array([]).reshape(0, 1, image.height, image.width)
        
        filtered_boxes = yolo_boxes[valid_indices]
        filtered_scores = yolo_scores[valid_indices]
        filtered_masks = yolo_masks[valid_indices]
        
        # Remove very small boxes
        areas = (filtered_boxes[:, 2] - filtered_boxes[:, 0]) * (filtered_boxes[:, 3] - filtered_boxes[:, 1])
        valid_size = areas > 32 * 32  # Minimum area
        
        if np.any(valid_size):
            filtered_boxes = filtered_boxes[valid_size]
            filtered_scores = filtered_scores[valid_size]
            filtered_masks = filtered_masks[valid_size]
        
        # Sort by objectness score and limit
        if len(filtered_boxes) > max_proposals:
            sorted_indices = np.argsort(filtered_scores)[::-1][:max_proposals]
            final_boxes = filtered_boxes[sorted_indices]
            final_scores = filtered_scores[sorted_indices]
            final_masks = filtered_masks[sorted_indices]
        else:
            final_boxes = filtered_boxes
            final_scores = filtered_scores
            final_masks = filtered_masks
        
        # Add channel dimension to masks to match expected format [N, 1, H, W]
        final_masks = final_masks[:, np.newaxis, :, :]
        
        if verbose:
            print(f"Final proposal count: {len(final_boxes)}")
        
        return final_boxes, final_scores, final_masks



    def crop_regions(self, image: Image.Image, boxes: np.ndarray) -> List[Image.Image]:
        """Crop regions from image based on bounding boxes, expanded by 1.1x for better context."""
        crops = []
        for box in boxes:
            x1, y1, x2, y2 = box.astype(int)
            
            # Calculate original center and dimensions
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            width = x2 - x1
            height = y2 - y1
            
            # Expand by 1.1x while keeping center the same
            new_width = width * 1.2
            new_height = height * 1.2
            
            # Calculate new coordinates
            new_x1 = int(center_x - new_width / 2)
            new_y1 = int(center_y - new_height / 2)
            new_x2 = int(center_x + new_width / 2)
            new_y2 = int(center_y + new_height / 2)
            
            # Ensure coordinates are within image bounds
            new_x1 = max(0, new_x1)
            new_y1 = max(0, new_y1)
            new_x2 = min(image.width, new_x2)
            new_y2 = min(image.height, new_y2)
            
            crop = image.crop((new_x1, new_y1, new_x2, new_y2))
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

    def compute_mask_iou(self, mask1: np.ndarray, mask2: np.ndarray, threshold: float = 0.5) -> float:
        """Compute Intersection over Union (IoU) of two segmentation masks."""
        # Convert to binary masks
        binary_mask1 = mask1 > threshold
        binary_mask2 = mask2 > threshold
        
        # Compute intersection and union
        intersection = np.logical_and(binary_mask1, binary_mask2).sum()
        union = np.logical_or(binary_mask1, binary_mask2).sum()
        
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

    def merge_detections(self, boxes: np.ndarray, scores: np.ndarray, masks: np.ndarray, 
                        similarities: np.ndarray, iou_threshold: float = 0.5, 
                        use_mask_iou: bool = True):
        """Merge overlapping detections instead of removing them."""
        if len(boxes) == 0:
            return boxes, scores, masks, similarities
        
        # Keep track of which detections have been merged
        merged_flags = np.zeros(len(boxes), dtype=bool)
        merged_boxes = []
        merged_scores = []
        merged_masks = []
        merged_similarities = []
        
        for i in range(len(boxes)):
            if merged_flags[i]:
                continue
                
            # Start a new merged detection
            current_boxes = [boxes[i]]
            current_scores = [scores[i]]
            current_masks = [masks[i]]
            current_similarities = [similarities[i]]
            merged_flags[i] = True
            
            # Find overlapping detections to merge
            for j in range(i + 1, len(boxes)):
                if merged_flags[j]:
                    continue
                    
                # Compute IoU
                if use_mask_iou:
                    # Use mask IoU
                    if masks[i].ndim == 3:
                        mask_i = masks[i][0]  # Take first channel
                    else:
                        mask_i = masks[i]
                    if masks[j].ndim == 3:
                        mask_j = masks[j][0]  # Take first channel
                    else:
                        mask_j = masks[j]
                    iou = self.compute_mask_iou(mask_i, mask_j)
                else:
                    # Use bounding box IoU
                    iou = self.compute_iou(boxes[i], boxes[j])
                
                # If overlap is above threshold, merge
                if iou > iou_threshold:
                    current_boxes.append(boxes[j])
                    current_scores.append(scores[j])
                    current_masks.append(masks[j])
                    current_similarities.append(similarities[j])
                    merged_flags[j] = True
            
            # Merge the collected detections
            if len(current_boxes) == 1:
                # No merging needed
                merged_boxes.append(current_boxes[0])
                merged_scores.append(current_scores[0])
                merged_masks.append(current_masks[0])
                merged_similarities.append(current_similarities[0])
            else:
                # Merge multiple detections
                # Use highest similarity score
                best_idx = np.argmax(current_similarities)
                merged_score = current_scores[best_idx]
                merged_similarity = current_similarities[best_idx]
                
                # Create encompassing bounding box
                all_boxes = np.array(current_boxes)
                min_x = np.min(all_boxes[:, 0])
                min_y = np.min(all_boxes[:, 1])
                max_x = np.max(all_boxes[:, 2])
                max_y = np.max(all_boxes[:, 3])
                merged_box = np.array([min_x, min_y, max_x, max_y])
                
                # Union all masks
                merged_mask = np.zeros_like(current_masks[0])
                for mask in current_masks:
                    merged_mask = np.maximum(merged_mask, mask)
                
                merged_boxes.append(merged_box)
                merged_scores.append(merged_score)
                merged_masks.append(merged_mask)
                merged_similarities.append(merged_similarity)
        
        return (np.array(merged_boxes), np.array(merged_scores), 
                np.array(merged_masks), np.array(merged_similarities))
    
    def apply_nms(self, boxes: np.ndarray, scores: np.ndarray, masks: np.ndarray, 
                  similarities: np.ndarray, iou_threshold: float = 0.5, 
                  use_mask_iou: bool = True):
        """Apply Non-Maximum Suppression to remove overlapping detections."""
        if len(boxes) == 0:
            return boxes, scores, masks, similarities
        
        # Sort by similarity scores (highest first)
        sorted_indices = np.argsort(similarities)[::-1]
        
        keep_indices = []
        
        while len(sorted_indices) > 0:
            # Take the detection with highest similarity
            current_idx = sorted_indices[0]
            keep_indices.append(current_idx)
            
            if len(sorted_indices) == 1:
                break
                
            # Compute IoU with remaining detections
            current_box = boxes[current_idx]
            remaining_indices = sorted_indices[1:]
            
            if use_mask_iou:
                # Use mask IoU
                if masks[current_idx].ndim == 3:
                    current_mask = masks[current_idx][0]  # Take first channel
                else:
                    current_mask = masks[current_idx]
                
                ious = np.array([
                    self.compute_mask_iou(current_mask, 
                                        masks[idx][0] if masks[idx].ndim == 3 else masks[idx])
                    for idx in remaining_indices
                ])
            else:
                # Use bounding box IoU
                ious = np.array([
                    self.compute_iou(current_box, boxes[idx]) 
                    for idx in remaining_indices
                ])
            
            # Keep only detections with IoU below threshold
            keep_mask = ious < iou_threshold
            sorted_indices = remaining_indices[keep_mask]
        
        keep_indices = np.array(keep_indices)
        return boxes[keep_indices], scores[keep_indices], masks[keep_indices], similarities[keep_indices]

    def filter_detections(self, boxes: np.ndarray, scores: np.ndarray, masks: np.ndarray, similarities: np.ndarray,
                         text_queries: List[str], similarity_threshold: float = 0.25, top_k: int = 5, 
                         iou_threshold: float = 0.5, apply_nms: bool = True, 
                         merge_overlaps: bool = False, use_mask_iou: bool = True):
        """Filter detections based on similarity threshold and keep top-k per class."""
        # Classify boxes by their best matching query
        class_results = self.classify_boxes(similarities, text_queries, similarity_threshold)
        
        # Collect all final results
        all_final_boxes = []
        all_final_scores = []
        all_final_masks = []
        all_final_similarities = []
        all_final_classes = []
        
        # Process each class separately
        for query, (box_indices, box_similarities) in class_results.items():
            if len(box_indices) == 0:
                continue
                
            # Get boxes, scores, masks for this class
            class_boxes = boxes[box_indices]
            class_scores = scores[box_indices]
            class_masks = masks[box_indices]
            class_similarities = box_similarities
            
            # Handle overlapping detections within this class only
            if len(class_boxes) > 1:
                if merge_overlaps:
                    # Merge overlapping detections instead of removing them
                    class_boxes, class_scores, class_masks, class_similarities = self.merge_detections(
                        class_boxes, class_scores, class_masks, class_similarities, 
                        iou_threshold, use_mask_iou
                    )
                elif apply_nms:
                    # Apply Non-Maximum Suppression to remove overlapping detections
                    class_boxes, class_scores, class_masks, class_similarities = self.apply_nms(
                        class_boxes, class_scores, class_masks, class_similarities, 
                        iou_threshold, use_mask_iou
                    )
            
            # Sort by similarity and take top-k for this class
            if len(class_boxes) > top_k:
                sorted_indices = np.argsort(class_similarities)[::-1][:top_k]
                class_boxes = class_boxes[sorted_indices]
                class_scores = class_scores[sorted_indices]
                class_masks = class_masks[sorted_indices]
                class_similarities = class_similarities[sorted_indices]
            
            # Add to final results with class labels
            if len(class_boxes) > 0:
                all_final_boxes.append(class_boxes)
                all_final_scores.append(class_scores)
                all_final_masks.append(class_masks)
                all_final_similarities.append(class_similarities)
                all_final_classes.extend([query] * len(class_boxes))
        
        # Combine all classes
        if len(all_final_boxes) > 0:
            final_boxes = np.vstack(all_final_boxes)
            final_scores = np.concatenate(all_final_scores)
            final_masks = np.vstack(all_final_masks)
            final_similarities = np.concatenate(all_final_similarities)
            final_classes = all_final_classes
        else:
            final_boxes = np.array([]).reshape(0, 4)
            final_scores = np.array([])
            final_masks = np.array([]).reshape(0, 1, 1, 1)
            final_similarities = np.array([])
            final_classes = []
        
        return final_boxes, final_scores, final_masks, final_similarities, final_classes


    def visualize_segmentations_to_pil(self, image: Image.Image, boxes: np.ndarray, masks: np.ndarray, 
                                      similarities: np.ndarray, classes: List[str], text_queries):
        """Visualize segmentations and return PIL image (for video processing)."""
        # Convert single query to list for uniform handling
        if isinstance(text_queries, str):
            text_queries = [text_queries]
        
        # Convert PIL to numpy for matplotlib
        img_array = np.array(image)
        height, width = img_array.shape[:2]
        
        # Calculate figure size to maintain aspect ratio (scale to reasonable size)
        dpi = 100
        fig_width = width / dpi
        fig_height = height / dpi
        
        # Create figure with exact aspect ratio
        fig, ax = plt.subplots(1, 1, figsize=(fig_width, fig_height), dpi=dpi)
        ax.imshow(img_array)
        
        # Colors for different classes
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'cyan', 'magenta', 'yellow']
        class_colors = {}
        for i, query in enumerate(text_queries):
            class_colors[query] = colors[i % len(colors)]
        
        for i, (box, mask, similarity, cls) in enumerate(zip(boxes, masks, similarities, classes)):
            color = class_colors.get(cls, colors[i % len(colors)])
            
            # Get the mask (squeeze to 2D)
            if mask.ndim == 3:
                mask_2d = mask[0]  # Take first channel
            else:
                mask_2d = mask
            
            # Masks should be at the correct resolution after filtering and resizing
            if mask_2d.shape != (height, width):
                # This shouldn't happen if masks were properly resized after filtering
                scale_y = height / mask_2d.shape[0]
                scale_x = width / mask_2d.shape[1]
                mask_2d = zoom(mask_2d, (scale_y, scale_x), order=1)
            
            # Create colored mask overlay
            colored_mask = np.zeros((height, width, 4))
            mask_binary = mask_2d > 0.5
            colored_mask[mask_binary] = to_rgba(color, alpha=0.5)
            
            # Overlay the mask
            ax.imshow(colored_mask)
            
            # Draw bounding box
            x1, y1, x2, y2 = box.astype(int)
            rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, 
                                   linewidth=2, edgecolor=color, facecolor='none')
            ax.add_patch(rect)
            
            # Add class and similarity score
            label = f"{cls}: {similarity:.3f}"
            # Use smaller font size for smaller images
            font_size = max(8, min(16, width // 50))
            ax.text(x1, y1-10, label, fontsize=font_size, color=color, weight='bold',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
        
        # Add title
        if len(text_queries) == 1:
            title = f"Query: '{text_queries[0]}' | Segmentations: {len(boxes)}"
        else:
            title = f"Queries: {len(text_queries)} | Segmentations: {len(boxes)}"
        
        title_font_size = max(10, min(20, width // 40))
        ax.set_title(title, fontsize=title_font_size, weight='bold')
        ax.axis('off')
        
        # Remove all padding and margins to maintain exact size
        plt.subplots_adjust(left=0, bottom=0, right=1, top=0.95, wspace=0, hspace=0)
        
        # Convert matplotlib figure to PIL Image
        fig.canvas.draw()
        
        # Convert to numpy array and then to PIL
        buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        
        # Close the figure to free memory
        plt.close(fig)
        
        # Convert to PIL - should already be the right size
        vis_image = Image.fromarray(buf)
        
        return vis_image

    def visualize_segmentations(self, image: Image.Image, boxes: np.ndarray, masks: np.ndarray, 
                              similarities: np.ndarray, classes: List[str], text_queries, output_path: str):
        """Visualize segmentations with masks and similarity scores."""
        vis_image = self.visualize_segmentations_to_pil(image, boxes, masks, similarities, classes, text_queries)
        
        # Save the result
        vis_image.save(output_path, dpi=(150, 150))
        print(f"Segmentation visualization saved to: {output_path}")


    def segment(self, image: Image.Image, text_queries, 
               confidence_threshold: float = 0.1,
               similarity_threshold: float = 0.25,
               max_proposals: int = 300,
               top_k: int = 5,
               iou_threshold: float = 0.5,
               apply_nms: bool = True,
               merge_overlaps: bool = False,
               use_mask_iou: bool = True,
               apply_filtering: bool = True,
               verbose: bool = True,
               debug: bool = False,
               timing_dict: dict = None):
        """Perform open-vocabulary instance segmentation on an image.
        
        Args:
            image: PIL Image to process
            text_queries: Text descriptions of objects to find (list or single string)
            confidence_threshold: Minimum confidence for region proposals
            similarity_threshold: Minimum CLIP similarity for final detections
            max_proposals: Maximum number of region proposals to consider
            top_k: Maximum number of final detections to return per class
            iou_threshold: IoU threshold for NMS/merging (higher = more aggressive removal/merging)
            apply_nms: Whether to apply Non-Maximum Suppression to remove overlaps
            merge_overlaps: Whether to merge overlapping detections instead of removing them
            use_mask_iou: Whether to use mask IoU (True) or bounding box IoU (False)
            apply_filtering: Whether to apply full filtering (NMS, merging, top-k) or just similarity threshold
            verbose: Whether to print progress messages
            debug: Whether to print timing breakdown information
            timing_dict: Optional dict to store timing information
            
        Returns:
            tuple: (boxes, scores, masks, similarities, classes) arrays for segmented objects
        """
        # Convert single query to list for uniform handling
        if isinstance(text_queries, str):
            text_queries = [text_queries]
        
        start_time = time.time()
        # Use external timing_dict if provided, otherwise create internal one
        internal_timing = timing_dict if timing_dict is not None else {}
        
        # Generate region proposals and masks (YOLOv8-seg inference + post-processing)
        if verbose:
            print("Generating region proposals and masks...")
        yolo_total_start = time.time()
        boxes, proposal_scores, masks = self.generate_proposals_and_masks(
            image, confidence_threshold, max_proposals, verbose, internal_timing
        )
        yolo_total_time = time.time() - yolo_total_start
        
        if len(boxes) == 0:
            if verbose:
                print("No region proposals found!")
            return np.array([]), np.array([]), np.array([]), np.array([]), []
        
        # Other processing start time (everything except model inference)
        other_start = time.time()
        
        # Crop regions (using bounding boxes)
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
        similarities = self.compute_similarities(image_features, text_features)
        
        # Filter detections (per class) - optional for performance testing
        if apply_filtering:
            if verbose:
                print("Filtering segmentations per class...")
            final_boxes, final_scores, final_masks, final_similarities, final_classes = self.filter_detections(
                boxes, proposal_scores, masks, similarities, text_queries, similarity_threshold, top_k,
                iou_threshold, apply_nms, merge_overlaps, use_mask_iou
            )
        else:
            if verbose:
                print("Skipping filtering - returning raw detections...")
            # Simple filtering: just apply similarity threshold and assign best class
            max_similarities = np.max(similarities, axis=1)
            best_query_indices = np.argmax(similarities, axis=1)
            valid_detections = max_similarities > similarity_threshold
            
            final_boxes = boxes[valid_detections]
            final_scores = proposal_scores[valid_detections]
            final_masks = masks[valid_detections]
            final_similarities = max_similarities[valid_detections]
            final_classes = [text_queries[i] for i in best_query_indices[valid_detections]]

        # Resize only the final filtered masks to original image resolution
        if len(final_masks) > 0:
            target_height, target_width = image.height, image.width
            if final_masks[0].shape[-2:] != (target_height, target_width):
                if verbose and not hasattr(self, '_mask_resize_printed'):
                    print(f"Resizing {len(final_masks)} final masks from {final_masks[0].shape[-2:]} to {(target_height, target_width)}")
                    self._mask_resize_printed = True
                
                resized_masks = []
                for mask in final_masks:
                    if mask.ndim == 3:
                        mask_2d = mask[0]  # Take first channel
                    else:
                        mask_2d = mask
                    
                    scale_y = target_height / mask_2d.shape[0]
                    scale_x = target_width / mask_2d.shape[1]
                    resized_mask = zoom(mask_2d, (scale_y, scale_x), order=1)
                    # Add channel dimension back
                    resized_masks.append(resized_mask[np.newaxis, :, :])
                
                final_masks = np.array(resized_masks)
        
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
            print(f"Found {len(final_boxes)} matching segmentations")
            if len(final_boxes) > 0:
                for i, (box, sim, cls) in enumerate(zip(final_boxes, final_similarities, final_classes)):
                    x1, y1, x2, y2 = box
                    print(f"Segmentation {i+1}: '{cls}' Box=({x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}), Similarity={sim:.3f}")
        
        return final_boxes, final_scores, final_masks, final_similarities, final_classes
    
    def segment_and_visualize(self, image_path: str, text_queries, 
                             confidence_threshold: float = 0.1,
                             similarity_threshold: float = 0.25,
                             max_proposals: int = 300,
                             top_k: int = 5,
                             iou_threshold: float = 0.5,
                             apply_nms: bool = True,
                             merge_overlaps: bool = False,
                             use_mask_iou: bool = True,
                             debug: bool = False,
                             output_path: Optional[str] = None):
        """Segment objects and save visualization."""
        # Convert single query to list for uniform handling
        if isinstance(text_queries, str):
            text_queries = [text_queries]
        
        # Load image
        print(f"Loading image: {image_path}")
        image = Image.open(image_path).convert('RGB')
        
        # Perform segmentation
        final_boxes, final_scores, final_masks, final_similarities, final_classes = self.segment(
            image, text_queries, confidence_threshold, similarity_threshold, 
            max_proposals, top_k, iou_threshold, apply_nms, merge_overlaps, 
            use_mask_iou, verbose=True, debug=debug
        )
        
        if len(final_boxes) > 0:
            # Visualize
            if output_path is None:
                base_name = os.path.splitext(os.path.basename(image_path))[0]
                if len(text_queries) == 1:
                    query_clean = text_queries[0].replace(' ', '_').replace('/', '_')
                    output_path = f"{base_name}_{query_clean}_segmentations.png"
                else:
                    output_path = f"{base_name}_multiclass_segmentations.png"
            
            print("Creating visualization...")
            self.visualize_segmentations(image, final_boxes, final_masks, final_similarities, final_classes, text_queries, output_path)
        else:
            queries_str = "', '".join(text_queries)
            print(f"No objects matching '{queries_str}' found with similarity > {similarity_threshold}")
        
        return final_boxes, final_scores, final_masks, final_similarities, final_classes


def main():
    """Example usage of the segmentation system."""
    # Example parameters - modify these for your use case
    image_path = "data/object_detection/cat.jpg"  # Change to your image path
    text_query = ["cat"]  # Change to your query
    
    # Check if example image exists
    if not os.path.exists(image_path):
        print(f"Image file '{image_path}' not found!")
        print("Please provide a valid image path in the main() function.")
        return
    
    # Initialize segmenter
    segmenter = ClipSegmenter()
    
    # Run segmentation with visualization
    segmenter.segment_and_visualize(
        image_path=image_path,
        text_queries=text_query,
        confidence_threshold=0.1,
        similarity_threshold=0.20,
        apply_nms=True,
        iou_threshold=0.01,
        max_proposals=300,
        debug=True,
        top_k=5
    )


if __name__ == "__main__":
    main() 