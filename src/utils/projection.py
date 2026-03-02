"""
Ground-Plane Distance Projection for Sprint 2.
Estimates the real-world distance (in metres) of a detected object
from the ego-vehicle using the bottom edge of its bounding box.

Assumes:
  - Flat ground plane
  - Fixed dashboard camera height and tilt angle
  - Simple pinhole camera model with inverse perspective mapping (IPM)
"""
import numpy as np


# Default camera parameters (reasonable for a smartphone on a dashboard)
DEFAULT_CAMERA_HEIGHT = 1.2    # metres above ground
DEFAULT_CAMERA_TILT = 10.0     # degrees below horizon
DEFAULT_FOCAL_LENGTH_PX = 800  # focal length in pixels (approximate for 1080p smartphone)
DEFAULT_IMG_HEIGHT = 720       # image height in pixels


def estimate_distance(bbox_bottom_y, img_height=DEFAULT_IMG_HEIGHT,
                      camera_height=DEFAULT_CAMERA_HEIGHT,
                      camera_tilt_deg=DEFAULT_CAMERA_TILT,
                      focal_length_px=DEFAULT_FOCAL_LENGTH_PX):
    """
    Estimate the ground-plane distance of an object from the ego-vehicle.

    Uses the simple geometric relationship:
        distance = (camera_height * focal_length) / (y_pixel - vanishing_point_y)

    Args:
        bbox_bottom_y: The y-coordinate of the bottom edge of the bounding box (in pixels).
        img_height: Height of the image in pixels.
        camera_height: Camera height above the ground plane (metres).
        camera_tilt_deg: Camera tilt angle below the horizon (degrees).
        focal_length_px: Focal length in pixels.

    Returns:
        distance_m: Estimated ground-plane distance in metres.
    """
    # Vanishing point: where the horizon line meets the image
    tilt_rad = np.radians(camera_tilt_deg)
    vanishing_y = (img_height / 2.0) - focal_length_px * np.tan(tilt_rad)

    # Pixel offset below vanishing point
    delta_y = bbox_bottom_y - vanishing_y

    if delta_y <= 0:
        # Object is at or above the horizon — distance is effectively infinite
        return float('inf')

    # Ground-plane distance via similar triangles
    distance_m = (camera_height * focal_length_px) / delta_y

    return max(distance_m, 0.5)  # Clamp minimum to 0.5m


def estimate_lateral_offset(bbox_center_x, img_width, distance_m,
                            focal_length_px=DEFAULT_FOCAL_LENGTH_PX):
    """
    Estimate the lateral offset (left/right) of an object from the ego-vehicle centreline.

    Args:
        bbox_center_x: The x-coordinate of the center of the bounding box (pixels).
        img_width: Width of the image in pixels.
        distance_m: Ground-plane distance (from estimate_distance).
        focal_length_px: Focal length in pixels.

    Returns:
        offset_m: Lateral offset in metres (positive = right, negative = left).
    """
    # Pixel offset from image centre
    delta_x = bbox_center_x - (img_width / 2.0)

    # Convert pixel offset to real-world offset using similar triangles
    offset_m = (delta_x * distance_m) / focal_length_px

    return offset_m


def project_detection(bbox, img_width, img_height, **camera_kwargs):
    """
    Project a single detection bounding box to ground-plane coordinates.

    Args:
        bbox: [cx, cy, w, h] in pixel coordinates
        img_width, img_height: Image dimensions

    Returns:
        dict with 'distance_m' and 'offset_m'
    """
    cx, cy, w, h = bbox

    # Bottom edge of the bounding box (closest point to camera on ground)
    bottom_y = cy + h / 2.0

    distance = estimate_distance(bottom_y, img_height=img_height, **camera_kwargs)
    offset = estimate_lateral_offset(cx, img_width, distance, **camera_kwargs)

    return {
        'distance_m': round(distance, 2),
        'offset_m': round(offset, 2),
    }
