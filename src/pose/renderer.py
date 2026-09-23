import cv2
import numpy as np
from typing import Dict, List, Tuple, Union
import logging

logger = logging.getLogger(__name__)

class PoseRenderer:
    """
    Renders pose skeletons to images.
    """
    
    # 18-keypoint body model connections
    BODY_CONNECTIONS = [
        (1, 2, (255, 0, 0)), (1, 5, (255, 85, 0)), (2, 3, (255, 170, 0)), (3, 4, (255, 255, 0)),
        (5, 6, (170, 255, 0)), (6, 7, (85, 255, 0)), (1, 8, (0, 255, 0)), (8, 9, (0, 255, 85)),
        (9, 10, (0, 255, 170)), (1, 11, (0, 255, 255)), (11, 12, (0, 170, 255)), (12, 13, (0, 85, 255)),
        (1, 0, (0, 0, 255)), (0, 14, (85, 0, 255)), (14, 16, (170, 0, 255)), (0, 15, (255, 0, 255)),
        (15, 17, (255, 0, 170))
    ]
    
    # 21-keypoint hand model connections (repeated for left and right hands)
    HAND_CONNECTIONS = [
        (0, 1), (1, 2), (2, 3), (3, 4),        # Thumb
        (0, 5), (5, 6), (6, 7), (7, 8),        # Index finger
        (0, 9), (9, 10), (10, 11), (11, 12),   # Middle finger
        (0, 13), (13, 14), (14, 15), (15, 16), # Ring finger
        (0, 17), (17, 18), (18, 19), (19, 20)  # Pinky finger
    ]

    def __init__(self, canvas_size: Tuple[int, int] = (832, 480), line_width: int = 4):
        """
        Initialize the PoseRenderer.
        
        Args:
            canvas_size (tuple[int, int]): Size of the output canvas (width, height).
            line_width (int): Base line width for drawing skeletons.
        """
        self.canvas_size = canvas_size
        self.line_width = line_width

    def render_pose(self, keypoints: Dict[str, Union[np.ndarray, list, float]], canvas_size: Tuple[int, int] = None) -> np.ndarray:
        """
        Renders a single pose skeleton on a black background.
        
        Args:
            keypoints (dict): Pose dictionary.
            canvas_size (tuple, optional): Canvas size (width, height) to override the default.
            
        Returns:
            np.ndarray: Rendered image as a numpy array.
        """
        if canvas_size is None:
            canvas_size = self.canvas_size
            
        # Create black background
        canvas = np.zeros((canvas_size[1], canvas_size[0], 3), dtype=np.uint8)
        
        # Draw body skeleton with colored limbs
        body_kp = keypoints.get("body_keypoints", np.zeros((18, 3)))
        for i, j, color in self.BODY_CONNECTIONS:
            if i < len(body_kp) and j < len(body_kp):
                pt1 = body_kp[i]
                pt2 = body_kp[j]
                if pt1[2] > 0.1 and pt2[2] > 0.1:  # Check confidence
                    x1, y1 = int(pt1[0]), int(pt1[1])
                    x2, y2 = int(pt2[0]), int(pt2[1])
                    cv2.line(canvas, (x1, y1), (x2, y2), color, self.line_width)
                    cv2.circle(canvas, (x1, y1), self.line_width, color, -1)
                    cv2.circle(canvas, (x2, y2), self.line_width, color, -1)
                    
        # Draw hand skeleton if available
        hand_kp = keypoints.get("hand_keypoints", np.zeros((42, 3)))
        if len(hand_kp) == 42:
            left_hand = hand_kp[:21]
            right_hand = hand_kp[21:]
            
            # Left hand blue, right hand red
            for hand, color in [(left_hand, (255, 0, 0)), (right_hand, (0, 0, 255))]:
                for i, j in self.HAND_CONNECTIONS:
                    if i < len(hand) and j < len(hand):
                        pt1 = hand[i]
                        pt2 = hand[j]
                        if pt1[2] > 0.1 and pt2[2] > 0.1:
                            x1, y1 = int(pt1[0]), int(pt1[1])
                            x2, y2 = int(pt2[0]), int(pt2[1])
                            cv2.line(canvas, (x1, y1), (x2, y2), color, max(1, self.line_width - 2))

        # Draw face outline if available (using simple white dots for simplicity)
        face_kp = keypoints.get("face_keypoints", np.zeros((68, 3)))
        if len(face_kp) == 68:
            for pt in face_kp:
                if pt[2] > 0.1:
                    cv2.circle(canvas, (int(pt[0]), int(pt[1])), max(1, self.line_width - 3), (255, 255, 255), -1)

        return canvas

    def render_pose_sequence(self, keypoints_list: List[Dict[str, Union[np.ndarray, list, float]]]) -> List[np.ndarray]:
        """
        Render all frames in a sequence.
        
        Args:
            keypoints_list (list[dict]): List of pose dictionaries.
            
        Returns:
            list[np.ndarray]: List of rendered frames.
        """
        return [self.render_pose(kp) for kp in keypoints_list]

    def render_comparison(self, original_frame: np.ndarray, pose_image: np.ndarray) -> np.ndarray:
        """
        Render a side-by-side comparison of the original frame and the pose image.
        
        Args:
            original_frame (np.ndarray): Original image.
            pose_image (np.ndarray): Rendered pose image.
            
        Returns:
            np.ndarray: Side-by-side comparison image.
        """
        h1, w1 = original_frame.shape[:2]
        
        # Resize pose image to match original frame height and width
        pose_resized = cv2.resize(pose_image, (w1, h1))
        
        # Side by side concatenation
        return np.concatenate((original_frame, pose_resized), axis=1)

    def __call__(self, keypoints: Dict, canvas_size: Tuple[int, int] = None) -> np.ndarray:
        """Render a single pose on canvas."""
        return self.render_pose(keypoints, canvas_size)

    def render_poses(self, keypoints_list: List[Dict], canvas_size: Tuple[int, int] = None) -> List[np.ndarray]:
        """Render all frames in a sequence with optional canvas size override."""
        if canvas_size is not None and len(canvas_size) == 2:
            # If shape was passed as (H, W), convert to (W, H) if needed
            cs = (canvas_size[1], canvas_size[0]) if canvas_size[0] > canvas_size[1] and self.canvas_size[0] > self.canvas_size[1] else canvas_size
        else:
            cs = canvas_size
        return [self.render_pose(kp, cs) for kp in keypoints_list]

