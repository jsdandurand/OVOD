"""
Multi-Object Tracking System for OVOD
Implements SORT (Simple Online and Realtime Tracking) with enhancements
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
import time
from dataclasses import dataclass
from enum import Enum
import cv2


class TrackState(Enum):
    """Track state enumeration."""
    NEW = 0
    TRACKED = 1
    LOST = 2
    REMOVED = 3


@dataclass
class Track:
    """Represents a tracked object."""
    id: int
    bbox: np.ndarray  # [x1, y1, x2, y2]
    class_name: str
    confidence: float
    state: TrackState
    age: int
    hits: int
    time_since_update: int
    kalman_filter: Optional[object] = None
    features: Optional[np.ndarray] = None
    last_detection_time: float = 0.0


class KalmanFilter:
    """Kalman filter for object tracking."""
    
    def __init__(self):
        # State: [x1, y1, x2, y2, vx1, vy1, vx2, vy2]
        self.ndim = 8
        self.dt = 1.0
        
        # State transition matrix
        self.F = np.eye(self.ndim)
        for i in range(4):
            self.F[i, i + 4] = self.dt
        
        # Measurement matrix (we only observe position, not velocity)
        self.H = np.zeros((4, self.ndim))
        for i in range(4):
            self.H[i, i] = 1.0
        
        # Process noise covariance - increased for better velocity prediction
        self.Q = np.eye(self.ndim) * 0.3
        
        # Measurement noise covariance - increased for fast motion tolerance
        self.R = np.eye(4) * 2.0
        
        # Initial state covariance
        self.P = np.eye(self.ndim) * 10.0
        
        # State vector
        self.x = np.zeros(self.ndim)
    
    def predict(self):
        """Predict next state with velocity scaling for fast motion."""
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x[:4]  # Return position only
    
    def update(self, measurement):
        """Update with measurement."""
        # Kalman gain
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        
        # Update state
        y = measurement - self.H @ self.x
        self.x = self.x + K @ y
        
        # Update covariance
        I = np.eye(self.ndim)
        self.P = (I - K @ self.H) @ self.P
        
        return self.x[:4]
    
    def set_state(self, bbox):
        """Set initial state from bbox."""
        self.x[:4] = bbox
        self.x[4:] = 0  # Zero velocity initially


class SORTTracker:
    """SORT (Simple Online and Realtime Tracking) implementation."""
    
    def __init__(self, 
                 max_age: int = 30,
                 min_hits: int = 10,
                 iou_threshold: float = 0.3,
                 feature_similarity_threshold: float = 0.7,
                 max_displacement: float = 1000.0,
                 velocity_scale: float = 1.0):
        """
        Initialize SORT tracker.
        
        Args:
            max_age: Maximum number of frames to keep a track without updates
            min_hits: Minimum number of detections before track is confirmed
            iou_threshold: IoU threshold for association
            feature_similarity_threshold: Feature similarity threshold for association
            max_displacement: Maximum expected displacement between frames (pixels)
            velocity_scale: Scale factor for velocity prediction (higher = more aggressive)
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.feature_similarity_threshold = feature_similarity_threshold
        self.max_displacement = max_displacement
        self.velocity_scale = velocity_scale
        
        self.tracks: List[Track] = []
        self.frame_count = 0
        self.next_id = 1
        
        # Performance metrics
        self.metrics = {
            'total_tracks': 0,
            'active_tracks': 0,
            'track_switches': 0,
            'fragments': 0,
            'avg_track_lifetime': 0.0
        }
    
    def update(self, detections: List[Dict]) -> List[Dict]:
        """
        Update tracker with new detections.
        
        Args:
            detections: List of detection dictionaries with keys:
                       'bbox', 'class', 'confidence', 'features' (optional)
        
        Returns:
            List of tracked objects with additional tracking info
        """
        self.frame_count += 1
        
        # Convert detections to numpy arrays
        detection_bboxes = np.array([d['bbox'] for d in detections])
        detection_classes = [d['class'] for d in detections]
        detection_confidences = [d['confidence'] for d in detections]
        detection_features = [d.get('features', None) for d in detections]
        
        # Predict new locations for existing tracks
        self._predict_tracks()
        
        # Associate detections with tracks
        matched_tracks, matched_detections, unmatched_tracks, unmatched_detections = \
            self._associate_detections_to_tracks(detection_bboxes, detection_classes, detection_features)
        
        # Update matched tracks
        for track_idx, det_idx in zip(matched_tracks, matched_detections):
            track = self.tracks[track_idx]
            bbox = detection_bboxes[det_idx]
            class_name = detection_classes[det_idx]
            confidence = detection_confidences[det_idx]
            features = detection_features[det_idx]
            
            # Update track
            track.bbox = bbox
            track.class_name = class_name
            track.confidence = confidence
            track.age += 1
            track.hits += 1
            track.time_since_update = 0
            track.last_detection_time = time.time()
            
            # Update track state based on hits
            if track.hits >= self.min_hits:
                track.state = TrackState.TRACKED  # Confirmed track
            else:
                track.state = TrackState.NEW      # Tentative track
            
            if track.kalman_filter is not None:
                track.kalman_filter.update(bbox)
            
            if features is not None:
                track.features = features
        
        # Create new tracks for unmatched detections
        for det_idx in unmatched_detections:
            bbox = detection_bboxes[det_idx]
            class_name = detection_classes[det_idx]
            confidence = detection_confidences[det_idx]
            features = detection_features[det_idx]
            
            # Create Kalman filter
            kf = KalmanFilter()
            kf.set_state(bbox)
            
            # Create new track
            track = Track(
                id=self.next_id,
                bbox=bbox,
                class_name=class_name,
                confidence=confidence,
                state=TrackState.NEW,
                age=1,
                hits=1,
                time_since_update=0,
                kalman_filter=kf,
                features=features,
                last_detection_time=time.time()
            )
            
            self.tracks.append(track)
            self.next_id += 1
            self.metrics['total_tracks'] += 1
        
        # Update unmatched tracks
        for track_idx in unmatched_tracks:
            track = self.tracks[track_idx]
            track.state = TrackState.LOST
            track.time_since_update += 1
        
        # Remove old tracks
        self._remove_old_tracks()
        
        # Merge similar tracks (handle fast motion fragmentation)
        self._merge_similar_tracks()
        
        # Update metrics
        self._update_metrics()
        
        # Return active tracks
        return self._get_active_tracks()
    
    def _predict_tracks(self):
        """Predict new locations for all tracks with velocity scaling for fast motion."""
        for track in self.tracks:
            if track.kalman_filter is not None:
                predicted_bbox = track.kalman_filter.predict()
                
                # Apply velocity scaling for fast motion
                if track.age > 1:  # Only after first update
                    # Scale velocity components for better fast motion prediction
                    velocity = track.kalman_filter.x[4:] * self.velocity_scale
                    predicted_bbox[:2] += velocity[:2]  # x1, y1
                    predicted_bbox[2:] += velocity[2:]  # x2, y2
                
                track.bbox = predicted_bbox
            track.time_since_update += 1
    
    def _associate_detections_to_tracks(self, detection_bboxes, detection_classes, detection_features):
        """Associate detections to tracks using IoU, feature similarity, and displacement."""
        if len(self.tracks) == 0:
            return [], [], [], list(range(len(detection_bboxes)))
        
        if len(detection_bboxes) == 0:
            return [], [], list(range(len(self.tracks))), []
        
        # Calculate IoU matrix
        iou_matrix = self._calculate_iou_matrix(detection_bboxes)
        
        # Calculate displacement matrix for fast motion handling
        displacement_matrix = self._calculate_displacement_matrix(detection_bboxes)
        
        # Calculate feature similarity matrix if features are available
        feature_matrix = None
        if any(f is not None for f in detection_features):
            feature_matrix = self._calculate_feature_similarity_matrix(detection_features)
        
        # Combine IoU, displacement, and feature similarity
        similarity_matrix = iou_matrix.copy()
        
        # Apply displacement penalty for fast-moving objects
        # Higher displacement = lower similarity score
        displacement_penalty = np.exp(-displacement_matrix / self.max_displacement)
        similarity_matrix *= displacement_penalty
        
        if feature_matrix is not None:
            # Weighted combination: IoU * 0.5 + Feature Similarity * 0.5
            similarity_matrix = 0.8 * similarity_matrix + 0.2 * feature_matrix
        
        # Hungarian algorithm for optimal assignment
        from scipy.optimize import linear_sum_assignment
        track_indices, detection_indices = linear_sum_assignment(-similarity_matrix)
        
        # Filter assignments based on threshold
        matched_tracks = []
        matched_detections = []
        unmatched_tracks = []
        unmatched_detections = []
        
        for track_idx, det_idx in zip(track_indices, detection_indices):
            if similarity_matrix[track_idx, det_idx] >= self.iou_threshold:
                matched_tracks.append(track_idx)
                matched_detections.append(det_idx)
            else:
                unmatched_tracks.append(track_idx)
                unmatched_detections.append(det_idx)
        
        # Add unmatched tracks and detections
        for i in range(len(self.tracks)):
            if i not in matched_tracks:
                unmatched_tracks.append(i)
        
        for i in range(len(detection_bboxes)):
            if i not in matched_detections:
                unmatched_detections.append(i)
        
        return matched_tracks, matched_detections, unmatched_tracks, unmatched_detections
    
    def _calculate_iou_matrix(self, detection_bboxes):
        """Calculate IoU matrix between tracks and detections."""
        iou_matrix = np.zeros((len(self.tracks), len(detection_bboxes)))
        
        for i, track in enumerate(self.tracks):
            for j, det_bbox in enumerate(detection_bboxes):
                iou_matrix[i, j] = self._calculate_iou(track.bbox, det_bbox)
        
        return iou_matrix
    
    def _calculate_feature_similarity_matrix(self, detection_features):
        """Calculate feature similarity matrix between tracks and detections."""
        feature_matrix = np.zeros((len(self.tracks), len(detection_features)))
        
        for i, track in enumerate(self.tracks):
            if track.features is None:
                continue
            
            for j, det_features in enumerate(detection_features):
                if det_features is None:
                    continue
                
                # Cosine similarity
                similarity = np.dot(track.features, det_features) / (
                    np.linalg.norm(track.features) * np.linalg.norm(det_features)
                )
                feature_matrix[i, j] = max(0, similarity)  # Ensure non-negative
        
        return feature_matrix
    
    def _calculate_displacement_matrix(self, detection_bboxes):
        """Calculate displacement matrix between tracks and detections."""
        displacement_matrix = np.zeros((len(self.tracks), len(detection_bboxes)))
        
        for i, track in enumerate(self.tracks):
            for j, det_bbox in enumerate(detection_bboxes):
                # Calculate center points
                track_center = np.array([
                    (track.bbox[0] + track.bbox[2]) / 2,
                    (track.bbox[1] + track.bbox[3]) / 2
                ])
                det_center = np.array([
                    (det_bbox[0] + det_bbox[2]) / 2,
                    (det_bbox[1] + det_bbox[3]) / 2
                ])
                
                # Calculate Euclidean distance
                displacement = np.linalg.norm(track_center - det_center)
                displacement_matrix[i, j] = displacement
        
        return displacement_matrix
    
    def _calculate_iou(self, bbox1, bbox2):
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
    
    def _remove_old_tracks(self):
        """Remove tracks that haven't been updated for too long."""
        tracks_to_remove = []
        
        for i, track in enumerate(self.tracks):
            if track.time_since_update > self.max_age:
                tracks_to_remove.append(i)
        
        # Remove in reverse order to maintain indices
        for i in reversed(tracks_to_remove):
            del self.tracks[i]
    
    def _merge_similar_tracks(self):
        """Merge tracks that are likely the same object (fast motion handling)."""
        if len(self.tracks) < 2:
            return
        
        # Use a safer approach - process one merge at a time and re-evaluate indices
        merged = True
        while merged:
            merged = False
            
            # Find the first pair of tracks to merge
            for i in range(len(self.tracks)):
                for j in range(i + 1, len(self.tracks)):
                    track1 = self.tracks[i]
                    track2 = self.tracks[j]
                    
                    # Only merge tracks of same class
                    if track1.class_name != track2.class_name:
                        continue
                    
                    # Calculate center distance
                    center1 = np.array([(track1.bbox[0] + track1.bbox[2]) / 2, 
                                       (track1.bbox[1] + track1.bbox[3]) / 2])
                    center2 = np.array([(track2.bbox[0] + track2.bbox[2]) / 2, 
                                       (track2.bbox[1] + track2.bbox[3]) / 2])
                    
                    distance = np.linalg.norm(center1 - center2)
                    
                    # Merge if tracks are very close (likely same object)
                    if distance < 50:  # 50 pixels threshold
                        # Keep the track with the lower ID (earlier track), merge features if available
                        if track1.id <= track2.id:
                            # Keep track1 (lower ID), merge features if available
                            if track2.features is not None and track1.features is not None:
                                # Average features
                                track1.features = (track1.features + track2.features) / 2
                            # Also merge hits and age for better tracking
                            track1.hits = max(track1.hits, track2.hits)
                            track1.age = max(track1.age, track2.age)
                            del self.tracks[j]
                        else:
                            # Keep track2 (lower ID), merge features if available
                            if track1.features is not None and track2.features is not None:
                                # Average features
                                track2.features = (track1.features + track2.features) / 2
                            # Also merge hits and age for better tracking
                            track2.hits = max(track1.hits, track2.hits)
                            track2.age = max(track1.age, track2.age)
                            del self.tracks[i]
                        
                        merged = True
                        break  # Exit inner loop and restart from beginning
                
                if merged:
                    break  # Exit outer loop and restart from beginning
    
    def _update_metrics(self):
        """Update tracking performance metrics."""
        active_tracks = [t for t in self.tracks if t.state in [TrackState.TRACKED, TrackState.NEW]]
        self.metrics['active_tracks'] = len(active_tracks)
        
        if len(self.tracks) > 0:
            avg_lifetime = np.mean([t.age for t in self.tracks])
            self.metrics['avg_track_lifetime'] = avg_lifetime
    
    def _get_active_tracks(self) -> List[Dict]:
        """Get active tracks as dictionaries."""
        active_tracks = []
        
        for track in self.tracks:
            # Only return tracks that have been confirmed (enough hits)
            if track.state in [TrackState.TRACKED, TrackState.NEW] and track.hits >= self.min_hits:
                active_tracks.append({
                    'id': track.id,
                    'bbox': track.bbox.tolist(),
                    'class': track.class_name,
                    'confidence': track.confidence,
                    'state': track.state.value,
                    'age': track.age,
                    'hits': track.hits,
                    'time_since_update': track.time_since_update,
                    'track_lifetime': time.time() - track.last_detection_time
                })
        
        return active_tracks
    
    def get_metrics(self) -> Dict:
        """Get tracking performance metrics."""
        return self.metrics.copy()
    
    def reset(self):
        """Reset tracker state."""
        self.tracks = []
        self.frame_count = 0
        self.next_id = 1
        self.metrics = {
            'total_tracks': 0,
            'active_tracks': 0,
            'track_switches': 0,
            'fragments': 0,
            'avg_track_lifetime': 0.0
        }


class OVODTracker:
    """High-level tracker interface for OVOD system."""
    
    def __init__(self, 
                 max_age: int = 30,
                 min_hits: int = 3,
                 iou_threshold: float = 0.3,
                 max_displacement: float = 1000.0,
                 velocity_scale: float = 1.2):
        """
        Initialize OVOD tracker.
        
        Args:
            max_age: Maximum frames to keep track without updates
            min_hits: Minimum detections before track confirmation
            iou_threshold: IoU threshold for track-detection association
            max_displacement: Maximum expected displacement between frames (pixels)
            velocity_scale: Scale factor for velocity prediction (higher = more aggressive)
        """
        self.tracker = SORTTracker(
            max_age=max_age,
            min_hits=min_hits,
            iou_threshold=iou_threshold,
            max_displacement=max_displacement,
            velocity_scale=velocity_scale
        )
        self.frame_count = 0
        self.tracking_history = {}  # Track history for visualization
    
    def update(self, detections: List[Dict]) -> List[Dict]:
        """
        Update tracker with new detections.
        
        Args:
            detections: List of detection dictionaries from OVOD detector
        
        Returns:
            List of tracked objects with additional tracking information
        """
        self.frame_count += 1
        
        # Convert OVOD detections to tracker format
        tracker_detections = []
        for det in detections:
            tracker_det = {
                'bbox': np.array(det['bbox']),
                'class': det['class'],
                'confidence': det['confidence'],
                'features': det.get('features', None)  # CLIP features if available
            }
            tracker_detections.append(tracker_det)
        
        # Update tracker
        tracked_objects = self.tracker.update(tracker_detections)
        
        # Update tracking history for visualization
        self._update_tracking_history(tracked_objects)
        
        return tracked_objects
    
    def _update_tracking_history(self, tracked_objects):
        """Update tracking history for visualization."""
        current_time = time.time()
        
        for obj in tracked_objects:
            track_id = obj['id']
            
            if track_id not in self.tracking_history:
                self.tracking_history[track_id] = {
                    'positions': [],
                    'class': obj['class'],
                    'first_seen': current_time,
                    'last_seen': current_time
                }
            
            # Add current position to history
            bbox = obj['bbox']
            center_x = (bbox[0] + bbox[2]) / 2
            center_y = (bbox[1] + bbox[3]) / 2
            
            self.tracking_history[track_id]['positions'].append({
                'x': center_x,
                'y': center_y,
                'time': current_time,
                'bbox': bbox
            })
            
            # Keep only last 30 positions for performance
            if len(self.tracking_history[track_id]['positions']) > 30:
                self.tracking_history[track_id]['positions'].pop(0)
            
            self.tracking_history[track_id]['last_seen'] = current_time
        
        # Clean up old tracks
        cutoff_time = current_time - 60  # Remove tracks older than 60 seconds
        tracks_to_remove = []
        for track_id, history in self.tracking_history.items():
            if history['last_seen'] < cutoff_time:
                tracks_to_remove.append(track_id)
        
        for track_id in tracks_to_remove:
            del self.tracking_history[track_id]
    
    def get_tracking_history(self, track_id: int) -> Optional[List]:
        """Get tracking history for a specific track."""
        if track_id in self.tracking_history:
            return self.tracking_history[track_id]['positions']
        return None
    
    def get_all_tracking_history(self) -> Dict:
        """Get all tracking history."""
        return self.tracking_history.copy()
    
    def get_metrics(self) -> Dict:
        """Get tracking performance metrics."""
        return self.tracker.get_metrics()
    
    def reset(self):
        """Reset tracker state."""
        self.tracker.reset()
        self.frame_count = 0
        self.tracking_history = {}


# Example usage and testing
if __name__ == "__main__":
    # Test the tracker
    tracker = OVODTracker()
    
    # Simulate some detections
    detections = [
        {'bbox': [100, 100, 200, 200], 'class': 'person', 'confidence': 0.9},
        {'bbox': [300, 300, 400, 400], 'class': 'car', 'confidence': 0.8}
    ]
    
    # Update tracker
    tracked_objects = tracker.update(detections)
    print(f"Tracked objects: {tracked_objects}")
    
    # Get metrics
    metrics = tracker.get_metrics()
    print(f"Tracking metrics: {metrics}") 