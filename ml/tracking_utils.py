"""
Utility functions for tracking visualization and analysis.
"""

import numpy as np
import cv2
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import to_rgba
import colorsys


def generate_track_colors(n_tracks: int) -> List[Tuple[int, int, int]]:
    """Generate distinct colors for tracking visualization."""
    colors = []
    for i in range(n_tracks):
        hue = i / n_tracks
        saturation = 0.8
        value = 0.9
        rgb = colorsys.hsv_to_rgb(hue, saturation, value)
        color = tuple(int(c * 255) for c in rgb)
        colors.append(color)
    return colors


def draw_tracking_visualization(image: np.ndarray, 
                              tracked_objects: List[Dict],
                              tracking_history: Dict = None,
                              show_trajectories: bool = True,
                              show_ids: bool = True) -> np.ndarray:
    """
    Draw tracking visualization on image.
    
    Args:
        image: Input image (BGR format)
        tracked_objects: List of tracked objects from tracker
        tracking_history: Optional tracking history for trajectories
        show_trajectories: Whether to show object trajectories
        show_ids: Whether to show track IDs
    
    Returns:
        Image with tracking visualization
    """
    vis_image = image.copy()
    
    if not tracked_objects:
        return vis_image
    
    # Generate colors for tracks
    colors = generate_track_colors(len(tracked_objects))
    
    for i, obj in enumerate(tracked_objects):
        track_id = obj['id']
        bbox = obj['bbox']
        class_name = obj['class']
        confidence = obj['confidence']
        age = obj['age']
        
        # Get color for this track
        color = colors[i % len(colors)]
        
        # Draw bounding box
        x1, y1, x2, y2 = map(int, bbox)
        cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, 2)
        
        # Draw track ID and info
        if show_ids:
            label = f"ID:{track_id} {class_name} ({confidence:.2f})"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            
            # Draw label background
            cv2.rectangle(vis_image, 
                         (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), 
                         color, -1)
            
            # Draw label text
            cv2.putText(vis_image, label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Draw trajectory if available
        if show_trajectories and tracking_history and track_id in tracking_history:
            trajectory = tracking_history[track_id]['positions']
            if len(trajectory) > 1:
                points = []
                for pos in trajectory:
                    points.append((int(pos['x']), int(pos['y'])))
                
                # Draw trajectory line
                for j in range(1, len(points)):
                    cv2.line(vis_image, points[j-1], points[j], color, 2)
                
                # Draw trajectory points
                for point in points:
                    cv2.circle(vis_image, point, 3, color, -1)
    
    return vis_image


def create_tracking_analytics(tracked_objects: List[Dict], 
                            tracking_history: Dict,
                            metrics: Dict) -> Dict:
    """
    Create analytics from tracking data.
    
    Args:
        tracked_objects: Current tracked objects
        tracking_history: Complete tracking history
        metrics: Tracker performance metrics
    
    Returns:
        Dictionary with analytics data
    """
    analytics = {
        'current_tracks': len(tracked_objects),
        'total_tracks_created': metrics.get('total_tracks', 0),
        'avg_track_lifetime': metrics.get('avg_track_lifetime', 0.0),
        'track_classes': {},
        'track_lifetimes': [],
        'track_distances': []
    }
    
    # Analyze current tracks
    for obj in tracked_objects:
        class_name = obj['class']
        if class_name not in analytics['track_classes']:
            analytics['track_classes'][class_name] = 0
        analytics['track_classes'][class_name] += 1
    
    # Analyze tracking history
    for track_id, history in tracking_history.items():
        if len(history['positions']) > 1:
            # Calculate track lifetime
            lifetime = history['last_seen'] - history['first_seen']
            analytics['track_lifetimes'].append(lifetime)
            
            # Calculate total distance traveled
            total_distance = 0
            positions = history['positions']
            for i in range(1, len(positions)):
                dx = positions[i]['x'] - positions[i-1]['x']
                dy = positions[i]['y'] - positions[i-1]['y']
                distance = np.sqrt(dx*dx + dy*dy)
                total_distance += distance
            
            analytics['track_distances'].append(total_distance)
    
    # Calculate statistics
    if analytics['track_lifetimes']:
        analytics['avg_lifetime'] = np.mean(analytics['track_lifetimes'])
        analytics['max_lifetime'] = np.max(analytics['track_lifetimes'])
    else:
        analytics['avg_lifetime'] = 0
        analytics['max_lifetime'] = 0
    
    if analytics['track_distances']:
        analytics['avg_distance'] = np.mean(analytics['track_distances'])
        analytics['max_distance'] = np.max(analytics['track_distances'])
    else:
        analytics['avg_distance'] = 0
        analytics['max_distance'] = 0
    
    return analytics


def plot_tracking_metrics(metrics_history: List[Dict], 
                         save_path: str = None) -> None:
    """
    Plot tracking performance metrics over time.
    
    Args:
        metrics_history: List of metrics dictionaries over time
        save_path: Optional path to save the plot
    """
    if not metrics_history:
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Extract data
    frames = list(range(len(metrics_history)))
    active_tracks = [m.get('active_tracks', 0) for m in metrics_history]
    total_tracks = [m.get('total_tracks', 0) for m in metrics_history]
    avg_lifetimes = [m.get('avg_track_lifetime', 0) for m in metrics_history]
    
    # Plot active tracks over time
    axes[0, 0].plot(frames, active_tracks, 'b-', linewidth=2)
    axes[0, 0].set_title('Active Tracks Over Time')
    axes[0, 0].set_xlabel('Frame')
    axes[0, 0].set_ylabel('Active Tracks')
    axes[0, 0].grid(True)
    
    # Plot total tracks created
    axes[0, 1].plot(frames, total_tracks, 'r-', linewidth=2)
    axes[0, 1].set_title('Total Tracks Created')
    axes[0, 1].set_xlabel('Frame')
    axes[0, 1].set_ylabel('Total Tracks')
    axes[0, 1].grid(True)
    
    # Plot average track lifetime
    axes[1, 0].plot(frames, avg_lifetimes, 'g-', linewidth=2)
    axes[1, 0].set_title('Average Track Lifetime')
    axes[1, 0].set_xlabel('Frame')
    axes[1, 0].set_ylabel('Lifetime (frames)')
    axes[1, 0].grid(True)
    
    # Plot track distribution by class (if available)
    if metrics_history and 'track_classes' in metrics_history[-1]:
        classes = list(metrics_history[-1]['track_classes'].keys())
        counts = list(metrics_history[-1]['track_classes'].values())
        
        axes[1, 1].bar(classes, counts, color=['red', 'blue', 'green', 'orange'])
        axes[1, 1].set_title('Current Tracks by Class')
        axes[1, 1].set_xlabel('Class')
        axes[1, 1].set_ylabel('Count')
        axes[1, 1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def calculate_tracking_accuracy(ground_truth: List[Dict], 
                              predictions: List[Dict],
                              iou_threshold: float = 0.5) -> Dict:
    """
    Calculate tracking accuracy metrics.
    
    Args:
        ground_truth: Ground truth detections
        predictions: Predicted detections
        iou_threshold: IoU threshold for matching
    
    Returns:
        Dictionary with accuracy metrics
    """
    if not ground_truth or not predictions:
        return {
            'precision': 0.0,
            'recall': 0.0,
            'f1_score': 0.0,
            'mota': 0.0
        }
    
    # Calculate IoU matrix
    iou_matrix = np.zeros((len(predictions), len(ground_truth)))
    
    for i, pred in enumerate(predictions):
        for j, gt in enumerate(ground_truth):
            iou_matrix[i, j] = calculate_iou(pred['bbox'], gt['bbox'])
    
    # Find matches
    from scipy.optimize import linear_sum_assignment
    pred_indices, gt_indices = linear_sum_assignment(-iou_matrix)
    
    # Count matches above threshold
    matches = 0
    for pred_idx, gt_idx in zip(pred_indices, gt_indices):
        if iou_matrix[pred_idx, gt_idx] >= iou_threshold:
            matches += 1
    
    # Calculate metrics
    precision = matches / len(predictions) if len(predictions) > 0 else 0
    recall = matches / len(ground_truth) if len(ground_truth) > 0 else 0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    # MOTA (Multiple Object Tracking Accuracy)
    mota = 1 - (len(predictions) - matches) / len(ground_truth) if len(ground_truth) > 0 else 0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'mota': mota,
        'matches': matches,
        'total_predictions': len(predictions),
        'total_ground_truth': len(ground_truth)
    }


def calculate_iou(bbox1: List[float], bbox2: List[float]) -> float:
    """Calculate Intersection over Union between two bounding boxes."""
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[2], bbox2[2])
    y2 = min(bbox1[3], bbox2[3])
    
    if x2 <= x1 or y2 <= y1:
        return 0.0
    
    intersection = (x2 - x1) * (y2 - y1)
    area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0


# Example usage
if __name__ == "__main__":
    # Test tracking visualization
    import numpy as np
    
    # Create test image
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Test tracked objects
    tracked_objects = [
        {'id': 1, 'bbox': [100, 100, 200, 200], 'class': 'person', 'confidence': 0.9, 'age': 5},
        {'id': 2, 'bbox': [300, 300, 400, 400], 'class': 'car', 'confidence': 0.8, 'age': 3}
    ]
    
    # Test tracking history
    tracking_history = {
        1: {
            'positions': [
                {'x': 100, 'y': 100, 'time': 0},
                {'x': 110, 'y': 105, 'time': 1},
                {'x': 120, 'y': 110, 'time': 2}
            ],
            'class': 'person',
            'first_seen': 0,
            'last_seen': 2
        }
    }
    
    # Draw visualization
    vis_image = draw_tracking_visualization(image, tracked_objects, tracking_history)
    
    # Create analytics
    metrics = {'total_tracks': 5, 'active_tracks': 2, 'avg_track_lifetime': 10.5}
    analytics = create_tracking_analytics(tracked_objects, tracking_history, metrics)
    
    print("Tracking Analytics:", analytics) 