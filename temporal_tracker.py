"""
Temporal tracker for OVOD segmentation to reduce flickering and improve consistency.
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
import cv2


@dataclass
class TrackerConfig:
    """Configuration for temporal tracking parameters."""
    # Matching thresholds
    min_mask_iou: float = 0.3          # Minimum IoU to consider same object
    max_centroid_distance: float = 100  # Max pixel distance for centroid matching
    min_similarity_consistency: float = 0.15  # Max allowed similarity change
    
    # Temporal parameters
    max_interpolation_frames: int = 5   # Max frames to interpolate missing detections
    confidence_decay_rate: float = 0.95  # How fast confidence decays per frame
    
    # Confirmation parameters (reduces false positive flickering)
    confirmation_frames_required: int = 3  # Consecutive frames needed to confirm object before showing
    
    # Hysteresis thresholds
    appear_threshold: float = 0.25       # Confidence needed to start showing object
    disappear_threshold: float = 0.2   # Confidence needed to keep showing object
    
    # Motion prediction
    velocity_smoothing: float = 0.7     # How much to smooth velocity estimates
    max_predicted_movement: float = 50  # Max pixels object can move per frame


class ObjectTrack:
    """Represents a single object being tracked over time."""
    
    def __init__(self, track_id: int, initial_detection: Dict, frame_number: int):
        self.track_id = track_id
        self.class_id = initial_detection['class_id']
        self.class_name = initial_detection['class_name']
        
        # Current state
        self.last_detection_frame = frame_number
        self.current_frame = frame_number
        self.is_active = True
        
        # Detection history
        self.detection_history = [initial_detection]
        self.position_history = [self._get_centroid(initial_detection['mask'])]
        
        # Temporal state
        self.smoothed_confidence = initial_detection['similarity']
        self.velocity = np.array([0.0, 0.0])  # pixels per frame
        self.currently_displayed = False
        
        # Confirmation state (for reducing false positive flickering)
        self.consecutive_detections = 1  # Start with 1 since we have initial detection
        self.is_confirmed = False  # Will be True once consecutive threshold is met
        
        # Current detection data
        self.current_box = initial_detection['box']
        self.current_mask = initial_detection['mask']
        self.current_similarity = initial_detection['similarity']
    
    def _get_centroid(self, mask: np.ndarray) -> np.ndarray:
        """Compute centroid of mask."""
        moments = cv2.moments(mask.astype(np.uint8))
        if moments['m00'] == 0:
            return np.array([0.0, 0.0])
        cx = moments['m10'] / moments['m00']
        cy = moments['m01'] / moments['m00']
        return np.array([cx, cy])
    
    def update_with_detection(self, detection: Dict, frame_number: int, config: TrackerConfig):
        """Update track with new detection."""
        previous_frame = self.current_frame
        self.last_detection_frame = frame_number
        self.current_frame = frame_number
        
        # Update confirmation status
        if not self.is_confirmed:
            # Check if this detection is consecutive (frame-by-frame)
            if frame_number == previous_frame + 1 or len(self.detection_history) == 1:
                # Consecutive frame or first detection
                self.consecutive_detections += 1
                if self.consecutive_detections >= config.confirmation_frames_required:
                    self.is_confirmed = True
            else:
                # Gap in detection, reset consecutive count but keep the track alive
                self.consecutive_detections = 1
        
        # Update position and velocity
        new_centroid = self._get_centroid(detection['mask'])
        if len(self.position_history) > 0:
            frame_diff = frame_number - (len(self.position_history) - 1)
            if frame_diff > 0:
                new_velocity = (new_centroid - self.position_history[-1]) / frame_diff
                self.velocity = (config.velocity_smoothing * self.velocity + 
                               (1 - config.velocity_smoothing) * new_velocity)
        
        self.position_history.append(new_centroid)
        self.detection_history.append(detection)
        
        # Update confidence with smoothing
        self.smoothed_confidence = (config.confidence_decay_rate * self.smoothed_confidence + 
                                  (1 - config.confidence_decay_rate) * detection['similarity'])
        
        # Update current state
        self.current_box = detection['box']
        self.current_mask = detection['mask']
        self.current_similarity = detection['similarity']
    
    def update_without_detection(self, frame_number: int, config: TrackerConfig):
        """Update track when no matching detection found."""
        self.current_frame = frame_number
        
        # Update confirmation status - reset consecutive count if not confirmed yet
        if not self.is_confirmed:
            self.consecutive_detections = 0  # Reset since we missed a detection
        
        # Decay confidence
        self.smoothed_confidence *= config.confidence_decay_rate
        
        # Check if should deactivate
        frames_since_detection = frame_number - self.last_detection_frame
        if frames_since_detection > config.max_interpolation_frames:
            self.is_active = False
    
    def get_interpolated_detection(self, frame_number: int, config: TrackerConfig) -> Optional[Dict]:
        """Get predicted detection for current frame."""
        frames_since_detection = frame_number - self.last_detection_frame
        
        if frames_since_detection == 0:
            # Have actual detection
            return {
                'box': self.current_box,
                'mask': self.current_mask,
                'similarity': self.current_similarity,
                'class_id': self.class_id,
                'class_name': self.class_name,
                'interpolated': False
            }
        elif frames_since_detection <= config.max_interpolation_frames:
            # Interpolate position
            predicted_centroid = (self.position_history[-1] + 
                                self.velocity * frames_since_detection)
            
            # Limit movement
            movement = np.linalg.norm(predicted_centroid - self.position_history[-1])
            if movement > config.max_predicted_movement * frames_since_detection:
                direction = (predicted_centroid - self.position_history[-1]) / movement
                predicted_centroid = (self.position_history[-1] + 
                                    direction * config.max_predicted_movement * frames_since_detection)
            
            # Create interpolated mask by translating last known mask
            translation = predicted_centroid - self.position_history[-1]
            interpolated_mask = self._translate_mask(self.current_mask, translation)
            interpolated_box = self._mask_to_box(interpolated_mask)
            
            return {
                'box': interpolated_box,
                'mask': interpolated_mask,
                'similarity': self.smoothed_confidence,
                'class_id': self.class_id,
                'class_name': self.class_name,
                'interpolated': True
            }
        
        return None
    
    def should_display(self, config: TrackerConfig) -> bool:
        """Determine if object should be displayed using confirmation and hysteresis."""
        # First check: object must be confirmed (detected for required consecutive frames)
        if not self.is_confirmed:
            return False
        
        # Second check: use hysteresis logic for confirmed objects
        if self.currently_displayed:
            return self.smoothed_confidence > config.disappear_threshold
        else:
            return self.smoothed_confidence > config.appear_threshold
    
    def _translate_mask(self, mask: np.ndarray, translation: np.ndarray) -> np.ndarray:
        """Translate mask by given offset."""
        h, w = mask.shape
        tx, ty = int(translation[0]), int(translation[1])
        
        # Create translation matrix
        M = np.float32([[1, 0, tx], [0, 1, ty]])
        translated_mask = cv2.warpAffine(mask.astype(np.uint8), M, (w, h))
        return translated_mask.astype(bool)
    
    def _mask_to_box(self, mask: np.ndarray) -> List[int]:
        """Convert mask to bounding box [x1, y1, x2, y2]."""
        coords = np.where(mask)
        if len(coords[0]) == 0:
            return [0, 0, 0, 0]
        y1, y2 = coords[0].min(), coords[0].max()
        x1, x2 = coords[1].min(), coords[1].max()
        return [x1, y1, x2, y2]


class TemporalTracker:
    """Main temporal tracker for managing multiple object tracks."""
    
    def __init__(self, config: TrackerConfig = None):
        self.config = config or TrackerConfig()
        self.tracks: List[ObjectTrack] = []
        self.next_track_id = 0
        self.current_frame = 0
    
    def update(self, detections: List[Dict], frame_number: int) -> List[Dict]:
        """
        Update tracker with new detections and return stable detections to display.
        
        detections: List of detection dicts with keys: 'box', 'mask', 'similarity', 'class_id', 'class_name'
        """
        self.current_frame = frame_number
        
        # Match detections to existing tracks
        matched_pairs, unmatched_detections, unmatched_tracks = self._match_detections(detections)
        
        # Update matched tracks
        for track, detection in matched_pairs:
            track.update_with_detection(detection, frame_number, self.config)
        
        # Update unmatched tracks (decay confidence)
        for track in unmatched_tracks:
            track.update_without_detection(frame_number, self.config)
        
        # Create new tracks for unmatched detections
        for detection in unmatched_detections:
            new_track = ObjectTrack(self.next_track_id, detection, frame_number)
            self.tracks.append(new_track)
            self.next_track_id += 1
        
        # Remove inactive tracks
        self.tracks = [track for track in self.tracks if track.is_active]
        
        # Generate stable detections for display
        stable_detections = []
        for track in self.tracks:
            if track.should_display(self.config):
                interpolated_detection = track.get_interpolated_detection(frame_number, self.config)
                if interpolated_detection:
                    stable_detections.append(interpolated_detection)
                    track.currently_displayed = True
            else:
                track.currently_displayed = False
        
        return stable_detections
    
    def _match_detections(self, detections: List[Dict]) -> Tuple[List[Tuple], List[Dict], List[ObjectTrack]]:
        """Match new detections to existing tracks using multiple criteria."""
        if not detections or not self.tracks:
            return [], detections, self.tracks
        
        # Compute matching scores between all detection-track pairs
        scores = np.zeros((len(detections), len(self.tracks)))
        
        for i, detection in enumerate(detections):
            for j, track in enumerate(self.tracks):
                if detection['class_id'] == track.class_id:
                    scores[i, j] = self._compute_match_score(detection, track)
        
        # Use Hungarian algorithm (simplified: greedy matching for now)
        matched_pairs = []
        used_detections = set()
        used_tracks = set()
        
        # Find best matches greedily
        while True:
            best_score = 0
            best_det_idx = -1
            best_track_idx = -1
            
            for i in range(len(detections)):
                if i in used_detections:
                    continue
                for j in range(len(self.tracks)):
                    if j in used_tracks:
                        continue
                    if scores[i, j] > best_score and scores[i, j] > 0.3:  # Minimum match threshold
                        best_score = scores[i, j]
                        best_det_idx = i
                        best_track_idx = j
            
            if best_det_idx == -1:
                break
            
            matched_pairs.append((self.tracks[best_track_idx], detections[best_det_idx]))
            used_detections.add(best_det_idx)
            used_tracks.add(best_track_idx)
        
        unmatched_detections = [det for i, det in enumerate(detections) if i not in used_detections]
        unmatched_tracks = [track for j, track in enumerate(self.tracks) if j not in used_tracks]
        
        return matched_pairs, unmatched_detections, unmatched_tracks
    
    def _compute_match_score(self, detection: Dict, track: ObjectTrack) -> float:
        """Compute matching score between detection and track."""
        # Mask IoU
        mask_iou = self._compute_mask_iou(detection['mask'], track.current_mask)
        if mask_iou < self.config.min_mask_iou:
            return 0.0
        
        # Centroid distance
        det_centroid = track._get_centroid(detection['mask'])
        track_centroid = track.position_history[-1] if track.position_history else np.array([0, 0])
        centroid_dist = np.linalg.norm(det_centroid - track_centroid)
        centroid_score = max(0, 1 - centroid_dist / self.config.max_centroid_distance)
        
        # Similarity consistency
        similarity_diff = abs(detection['similarity'] - track.current_similarity)
        similarity_score = max(0, 1 - similarity_diff / self.config.min_similarity_consistency)
        
        # Weighted combination
        total_score = 0.6 * mask_iou + 0.25 * centroid_score + 0.15 * similarity_score
        return total_score
    
    def _compute_mask_iou(self, mask1: np.ndarray, mask2: np.ndarray) -> float:
        """Compute IoU between two masks."""
        intersection = np.logical_and(mask1, mask2).sum()
        union = np.logical_or(mask1, mask2).sum()
        if union == 0:
            return 0.0
        return intersection / union 