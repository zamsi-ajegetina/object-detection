"""
Simple Online and Realtime Tracking (SORT) for Sprint 2.
Assigns consistent IDs across video frames using IoU matching.

This is a simplified SORT implementation without Kalman filtering,
using only IoU-based Hungarian assignment.
"""
import numpy as np
from scipy.optimize import linear_sum_assignment


class Track:
    """Represents a single tracked object."""
    _id_counter = 0

    def __init__(self, bbox, class_id, score):
        Track._id_counter += 1
        self.id = Track._id_counter
        self.bbox = bbox        # [cx, cy, w, h]
        self.class_id = class_id
        self.score = score
        self.age = 0            # frames since last update
        self.hits = 1           # total successful matches

    def update(self, bbox, score):
        self.bbox = bbox
        self.score = score
        self.age = 0
        self.hits += 1


def iou_batch(bb_a, bb_b):
    """
    Compute IoU between two sets of bboxes [cx, cy, w, h].
    Returns: (len(bb_a), len(bb_b)) matrix.
    """
    if len(bb_a) == 0 or len(bb_b) == 0:
        return np.zeros((len(bb_a), len(bb_b)))

    bb_a = np.array(bb_a)
    bb_b = np.array(bb_b)

    # Convert to [x1, y1, x2, y2]
    a_x1 = bb_a[:, 0] - bb_a[:, 2] / 2
    a_y1 = bb_a[:, 1] - bb_a[:, 3] / 2
    a_x2 = bb_a[:, 0] + bb_a[:, 2] / 2
    a_y2 = bb_a[:, 1] + bb_a[:, 3] / 2

    b_x1 = bb_b[:, 0] - bb_b[:, 2] / 2
    b_y1 = bb_b[:, 1] - bb_b[:, 3] / 2
    b_x2 = bb_b[:, 0] + bb_b[:, 2] / 2
    b_y2 = bb_b[:, 1] + bb_b[:, 3] / 2

    iou_matrix = np.zeros((len(bb_a), len(bb_b)))

    for i in range(len(bb_a)):
        inter_x1 = np.maximum(a_x1[i], b_x1)
        inter_y1 = np.maximum(a_y1[i], b_y1)
        inter_x2 = np.minimum(a_x2[i], b_x2)
        inter_y2 = np.minimum(a_y2[i], b_y2)

        inter_area = np.maximum(0, inter_x2 - inter_x1) * np.maximum(0, inter_y2 - inter_y1)
        a_area = (a_x2[i] - a_x1[i]) * (a_y2[i] - a_y1[i])
        b_area = (b_x2 - b_x1) * (b_y2 - b_y1)
        union = a_area + b_area - inter_area + 1e-6

        iou_matrix[i] = inter_area / union

    return iou_matrix


class SimpleTracker:
    """
    Simple IoU-based tracker (SORT without Kalman).

    Usage:
        tracker = SimpleTracker()
        for frame_detections in video:
            tracked = tracker.update(boxes, scores, class_ids)
            # tracked is a list of (track_id, bbox, class_id, score)
    """
    def __init__(self, iou_threshold=0.3, max_age=5, min_hits=2):
        self.iou_threshold = iou_threshold
        self.max_age = max_age      # frames before track is deleted
        self.min_hits = min_hits    # minimum hits before track is confirmed
        self.tracks = []
        Track._id_counter = 0

    def update(self, boxes, scores, class_ids):
        """
        Update tracker with new detections.

        Args:
            boxes: list of [cx, cy, w, h]
            scores: list of confidence scores
            class_ids: list of class indices

        Returns:
            List of (track_id, bbox, class_id, score) for confirmed tracks.
        """
        # Age all existing tracks
        for track in self.tracks:
            track.age += 1

        if len(boxes) == 0 and len(self.tracks) == 0:
            return []

        if len(boxes) == 0:
            # No new detections, just prune old tracks
            self.tracks = [t for t in self.tracks if t.age <= self.max_age]
            return [(t.id, t.bbox, t.class_id, t.score) for t in self.tracks if t.hits >= self.min_hits]

        if len(self.tracks) == 0:
            # No existing tracks, create new ones
            for box, score, cls in zip(boxes, scores, class_ids):
                self.tracks.append(Track(box, cls, score))
            return [(t.id, t.bbox, t.class_id, t.score) for t in self.tracks if t.hits >= self.min_hits]

        # Compute IoU cost matrix
        track_boxes = [t.bbox for t in self.tracks]
        iou_matrix = iou_batch(track_boxes, boxes)

        # Hungarian assignment (maximize IoU = minimize -IoU)
        cost_matrix = -iou_matrix
        row_indices, col_indices = linear_sum_assignment(cost_matrix)

        matched_tracks = set()
        matched_dets = set()

        for row, col in zip(row_indices, col_indices):
            if iou_matrix[row, col] >= self.iou_threshold:
                self.tracks[row].update(boxes[col], scores[col])
                matched_tracks.add(row)
                matched_dets.add(col)

        # Create new tracks for unmatched detections
        for d in range(len(boxes)):
            if d not in matched_dets:
                self.tracks.append(Track(boxes[d], class_ids[d], scores[d]))

        # Prune old tracks
        self.tracks = [t for t in self.tracks if t.age <= self.max_age]

        # Return confirmed tracks
        return [(t.id, t.bbox, t.class_id, t.score) for t in self.tracks if t.hits >= self.min_hits]

    def reset(self):
        self.tracks = []
        Track._id_counter = 0
