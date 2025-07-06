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


class OVODSegmenter:
    """Open Vocabulary Instance Segmentation using Mask R-CNN and CLIP."""
    
    def __init__(self, device: str = None):
        """Initialize the OVOD segmenter with models."""
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Initializing OVOD Segmenter on {self.device}...")
        
        # Load Mask R-CNN
        self.mask_rcnn = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
        
        self.mask_rcnn.eval()
        self.mask_rcnn.to(self.device)
        
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

    def generate_proposals_and_masks(self, image: Image.Image, confidence_threshold: float = 0.5, 
                                   max_proposals: int = 300, verbose: bool = True):
        """Generate region proposals and masks using Mask R-CNN."""
        # Convert PIL to tensor
        transform = transforms.Compose([
            transforms.ToTensor()
        ])
        
        image_tensor = transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            predictions = self.mask_rcnn(image_tensor)
        
        # Get all proposals regardless of confidence for more coverage
        maskrcnn_scores = predictions[0]['scores'].cpu().numpy()
        maskrcnn_boxes = predictions[0]['boxes'].cpu().numpy()
        maskrcnn_masks = predictions[0]['masks'].cpu().numpy()
        
        if verbose:
            print(f"Mask R-CNN generated {len(maskrcnn_boxes)} raw proposals")
        
        # Filter by confidence threshold
        valid_indices = maskrcnn_scores > confidence_threshold
        
        if not np.any(valid_indices):
            return np.array([]).reshape(0, 4), np.array([]), np.array([]).reshape(0, 1, image.height, image.width)
        
        filtered_boxes = maskrcnn_boxes[valid_indices]
        filtered_scores = maskrcnn_scores[valid_indices]
        filtered_masks = maskrcnn_masks[valid_indices]
        
        # Remove very small boxes
        areas = (filtered_boxes[:, 2] - filtered_boxes[:, 0]) * (filtered_boxes[:, 3] - filtered_boxes[:, 1])
        valid_size = areas > 32 * 32  # Minimum area
        
        if np.any(valid_size):
            filtered_boxes = filtered_boxes[valid_size]
            filtered_scores = filtered_scores[valid_size]
            filtered_masks = filtered_masks[valid_size]
        
        # Sort by score and limit
        if len(filtered_boxes) > max_proposals:
            sorted_indices = np.argsort(filtered_scores)[::-1][:max_proposals]
            final_boxes = filtered_boxes[sorted_indices]
            final_scores = filtered_scores[sorted_indices]
            final_masks = filtered_masks[sorted_indices]
        else:
            final_boxes = filtered_boxes
            final_scores = filtered_scores
            final_masks = filtered_masks
        
        if verbose:
            print(f"Final proposal count: {len(final_boxes)}")
        return final_boxes, final_scores, final_masks


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


    def compute_clip_embeddings(self, crops: List[Image.Image], text_queries: List[str]):
        """Compute CLIP embeddings for image crops and text queries."""
        # Process image crops
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
        
        # Get text embeddings from cache for all queries
        text_features = []
        for text_query in text_queries:
            text_feat = self._get_text_embedding(text_query)
            text_features.append(text_feat)
        
        text_features = torch.cat(text_features, dim=0)
        
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
        
        # Create figure and axis
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
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
            
            # Create colored mask overlay
            colored_mask = np.zeros((*mask_2d.shape, 4))
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
            ax.text(x1, y1-10, label, fontsize=12, color=color, weight='bold',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
        
        # Add title
        if len(text_queries) == 1:
            title = f"Query: '{text_queries[0]}' | Segmentations: {len(boxes)}"
        else:
            title = f"Queries: {len(text_queries)} | Segmentations: {len(boxes)}"
        ax.set_title(title, fontsize=14, weight='bold')
        ax.axis('off')
        
        # Convert matplotlib figure to PIL Image
        plt.tight_layout()
        fig.canvas.draw()
        
        # Convert to numpy array and then to PIL
        buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        
        # Close the figure to free memory
        plt.close(fig)
        
        # Convert to PIL and resize to match original image dimensions
        vis_image = Image.fromarray(buf)
        vis_image = vis_image.resize(image.size, Image.Resampling.LANCZOS)
        
        return vis_image

    def visualize_segmentations(self, image: Image.Image, boxes: np.ndarray, masks: np.ndarray, 
                              similarities: np.ndarray, classes: List[str], text_queries, output_path: str):
        """Visualize segmentations with masks and similarity scores."""
        vis_image = self.visualize_segmentations_to_pil(image, boxes, masks, similarities, classes, text_queries)
        
        # Save the result
        vis_image.save(output_path, dpi=(150, 150))
        print(f"Segmentation visualization saved to: {output_path}")


    def segment(self, image: Image.Image, text_queries, 
               confidence_threshold: float = 0.5,
               similarity_threshold: float = 0.25,
               max_proposals: int = 300,
               top_k: int = 5,
               iou_threshold: float = 0.5,
               apply_nms: bool = True,
               merge_overlaps: bool = False,
               use_mask_iou: bool = True,
               verbose: bool = True,
               debug: bool = False):
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
            verbose: Whether to print progress messages
            debug: Whether to print timing breakdown information
            
        Returns:
            tuple: (boxes, scores, masks, similarities, classes) arrays for segmented objects
        """
        # Convert single query to list for uniform handling
        if isinstance(text_queries, str):
            text_queries = [text_queries]
        
        start_time = time.time()
        
        # Generate region proposals and masks (RCNN inference, done once for all queries)
        if verbose:
            print("Generating region proposals and masks...")
        rcnn_start = time.time()
        boxes, proposal_scores, masks = self.generate_proposals_and_masks(
            image, confidence_threshold, max_proposals, verbose
        )
        rcnn_time = time.time() - rcnn_start
        
        if len(boxes) == 0:
            if verbose:
                print("No region proposals found!")
            return np.array([]), np.array([]), np.array([]), np.array([]), []
        
        # Other processing (cropping, similarities, filtering)
        other_start = time.time()
        
        # Crop regions (using bounding boxes)
        if verbose:
            print("Cropping regions...")
        crops = self.crop_regions(image, boxes)
        
        # CLIP inference (compute similarities with all queries)
        if verbose:
            print(f"Computing CLIP embeddings for {len(text_queries)} queries...")
        clip_start = time.time()
        image_features, text_features = self.compute_clip_embeddings(crops, text_queries)
        clip_time = time.time() - clip_start
        
        # Compute similarities (matrix: boxes x queries)
        if verbose:
            print("Computing similarities...")
        similarities = self.compute_similarities(image_features, text_features)
        
        # Filter detections (per class)
        if verbose:
            print("Filtering segmentations per class...")
        final_boxes, final_scores, final_masks, final_similarities, final_classes = self.filter_detections(
            boxes, proposal_scores, masks, similarities, text_queries, similarity_threshold, top_k,
            iou_threshold, apply_nms, merge_overlaps, use_mask_iou
        )
        
        other_time = time.time() - other_start - clip_time
        total_time = time.time() - start_time
        
        # Print timing breakdown
        if debug:
            rcnn_pct = (rcnn_time / total_time) * 100
            clip_pct = (clip_time / total_time) * 100
            other_pct = (other_time / total_time) * 100
            print(f"Timing: RCNN {rcnn_pct:.1f}%, CLIP {clip_pct:.1f}%, Other {other_pct:.1f}% ({total_time:.3f}s total)")
        
        if verbose:
            print(f"Found {len(final_boxes)} matching segmentations")
            if len(final_boxes) > 0:
                for i, (box, sim, cls) in enumerate(zip(final_boxes, final_similarities, final_classes)):
                    x1, y1, x2, y2 = box
                    print(f"Segmentation {i+1}: '{cls}' Box=({x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}), Similarity={sim:.3f}")
        
        return final_boxes, final_scores, final_masks, final_similarities, final_classes
    
    def segment_and_visualize(self, image_path: str, text_queries, 
                             confidence_threshold: float = 0.5,
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
    text_query = ["cat", "not cat"]  # Change to your query
    
    # Check if example image exists
    if not os.path.exists(image_path):
        print(f"Image file '{image_path}' not found!")
        print("Please provide a valid image path in the main() function.")
        return
    
    # Initialize segmenter
    segmenter = OVODSegmenter()
    
    # Run segmentation with visualization
    segmenter.segment_and_visualize(
        image_path=image_path,
        text_queries=text_query,
        confidence_threshold=0.3,
        similarity_threshold=0.25,
        apply_nms=True,
        iou_threshold=0.01,
        max_proposals=300,
        debug=True,
        top_k=5
    )


if __name__ == "__main__":
    main() 