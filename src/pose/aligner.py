import numpy as np
from typing import Dict, List, Tuple, Union
from scipy.ndimage import gaussian_filter1d
import logging

logger = logging.getLogger(__name__)

class PoseAligner:
    """
    Aligns source poses to match a reference person's position and scale.
    """
    def __init__(self, target_size: Tuple[int, int] = (832, 480)):
        """
        Initialize the PoseAligner.
        
        Args:
            target_size (tuple[int, int]): The target dimensions (width, height).
        """
        self.target_size = target_size

    def compute_body_bbox(self, keypoints: Dict[str, Union[np.ndarray, list, float]]) -> Tuple[float, float, float, float]:
        """
        Get bounding box from body keypoints.
        
        Args:
            keypoints (dict): Pose dictionary.
            
        Returns:
            tuple: Bounding box as (min_x, min_y, max_x, max_y).
        """
        body_kp = keypoints.get("body_keypoints", np.zeros((18, 3)))
        valid = body_kp[:, 2] > 0.1
        
        if not np.any(valid):
            return (0.0, 0.0, 0.0, 0.0)
            
        valid_kp = body_kp[valid]
        min_x = float(np.min(valid_kp[:, 0]))
        min_y = float(np.min(valid_kp[:, 1]))
        max_x = float(np.max(valid_kp[:, 0]))
        max_y = float(np.max(valid_kp[:, 1]))
        
        return (min_x, min_y, max_x, max_y)

    def compute_scale_factor(
        self, 
        source_bbox: Tuple[float, float, float, float], 
        target_bbox: Tuple[float, float, float, float]
    ) -> float:
        """
        Compute scale factor to match body sizes.
        
        Args:
            source_bbox (tuple): Source bounding box.
            target_bbox (tuple): Target bounding box.
            
        Returns:
            float: Scale factor.
        """
        src_h = source_bbox[3] - source_bbox[1]
        tgt_h = target_bbox[3] - target_bbox[1]
        
        if src_h <= 0.01:
            return 1.0
            
        return tgt_h / src_h

    def align_pose(
        self, 
        source_keypoints: Dict[str, Union[np.ndarray, list, float]], 
        reference_keypoints: Dict[str, Union[np.ndarray, list, float]]
    ) -> Dict[str, Union[np.ndarray, list, float]]:
        """
        Aligns source pose to match reference person's position and scale.
        
        Args:
            source_keypoints (dict): Source pose to transform.
            reference_keypoints (dict): Reference pose dict providing the target scale and position.
            
        Returns:
            dict: Aligned pose keypoints.
        """
        src_bbox = self.compute_body_bbox(source_keypoints)
        ref_bbox = self.compute_body_bbox(reference_keypoints)
        
        scale = self.compute_scale_factor(src_bbox, ref_bbox)
        
        src_center_x = (src_bbox[0] + src_bbox[2]) / 2.0
        src_center_y = (src_bbox[1] + src_bbox[3]) / 2.0
        
        ref_center_x = (ref_bbox[0] + ref_bbox[2]) / 2.0
        ref_center_y = (ref_bbox[1] + ref_bbox[3]) / 2.0
        
        aligned_kps = {}
        
        # Apply transformation to body, hands, and face keypoints
        for key in ["body_keypoints", "hand_keypoints", "face_keypoints"]:
            kps = source_keypoints[key].copy()
            valid = kps[:, 2] > 0.1
            
            kps[valid, 0] = (kps[valid, 0] - src_center_x) * scale + ref_center_x
            kps[valid, 1] = (kps[valid, 1] - src_center_y) * scale + ref_center_y
            
            aligned_kps[key] = kps
            
        # Transform the bounding box itself
        aligned_kps["bbox"] = [
            (src_bbox[0] - src_center_x) * scale + ref_center_x,
            (src_bbox[1] - src_center_y) * scale + ref_center_y,
            (src_bbox[2] - src_center_x) * scale + ref_center_x,
            (src_bbox[3] - src_center_y) * scale + ref_center_y,
        ]
        
        # Preserve confidence scores
        aligned_kps["confidence"] = source_keypoints.get("confidence", 1.0)
        
        return aligned_kps

    def align_pose_sequence(
        self, 
        source_sequence: List[Dict[str, Union[np.ndarray, list, float]]], 
        reference_keypoints: Dict[str, Union[np.ndarray, list, float]]
    ) -> List[Dict[str, Union[np.ndarray, list, float]]]:
        """
        Align an entire sequence of poses and apply smooth filtering to reduce jitter.
        
        Args:
            source_sequence (list[dict]): List of source poses.
            reference_keypoints (dict): Reference pose.
            
        Returns:
            list[dict]: Aligned and smoothed pose sequence.
        """
        if not source_sequence:
            return []
            
        logger.info(f"Aligning pose sequence of length {len(source_sequence)}")
        aligned_seq = [self.align_pose(p, reference_keypoints) for p in source_sequence]
        
        # Helper: apply smooth filtering to aligned sequence to reduce jitter
        # Using scipy gaussian_filter1d on keypoint positions over the time dimension (axis 0)
        for key in ["body_keypoints", "hand_keypoints", "face_keypoints"]:
            # Stack into a shape of (T, N, 3) where T is time, N is number of keypoints
            arr = np.array([p[key] for p in aligned_seq])
            
            # Filter X and Y coordinates along the time dimension
            arr[:, :, 0] = gaussian_filter1d(arr[:, :, 0], sigma=1.0, axis=0)
            arr[:, :, 1] = gaussian_filter1d(arr[:, :, 1], sigma=1.0, axis=0)
            
            for t in range(len(aligned_seq)):
                aligned_seq[t][key] = arr[t]
                
        logger.info("Pose sequence alignment and smoothing completed.")
        return aligned_seq
