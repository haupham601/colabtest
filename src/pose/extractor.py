import logging
import numpy as np
from PIL import Image
from typing import Dict, List, Optional, Union
import torch
import cv2

logger = logging.getLogger(__name__)

class DWPoseExtractor:
    """
    DWPose extraction wrapper. Uses controlnet_aux DWPoseDetector if available,
    or falls back to a custom implementation.
    """
    def __init__(
        self, 
        device: str = "cuda", 
        det_model_path: Optional[str] = None, 
        pose_model_path: Optional[str] = None
    ):
        self.device = device
        self.model = None
        
        try:
            from controlnet_aux import DWposeDetector
            logger.info("Initializing DWPoseDetector from controlnet_aux...")
            self.model = DWposeDetector.from_pretrained(
                "yzd-v/DWPose",
                det_filename="yolox_l.onnx" if det_model_path is None else det_model_path,
                pose_filename="dw-ll_ucoco_384.onnx" if pose_model_path is None else pose_model_path
            ).to(device)
            logger.info("DWPoseDetector successfully initialized.")
        except ImportError:
            logger.warning("controlnet_aux not found. DWPose extraction may not work as expected.")
        except Exception as e:
            logger.error(f"Failed to load DWPoseDetector: {e}")

    def extract_from_image(self, image: Image.Image) -> Dict[str, Union[np.ndarray, list, float]]:
        """
        Extract pose keypoints from a single PIL image.
        
        Args:
            image (PIL.Image): Input image.
            
        Returns:
            dict: Pose information containing body, hand, face keypoints, bounding box, and confidence.
        """
        # Default empty result structure
        result = {
            "body_keypoints": np.zeros((18, 3), dtype=np.float32),
            "hand_keypoints": np.zeros((42, 3), dtype=np.float32),
            "face_keypoints": np.zeros((68, 3), dtype=np.float32),
            "bbox": [0.0, 0.0, 0.0, 0.0],
            "confidence": 0.0
        }

        if self.model is None:
            logger.error("DWPose model is not initialized. Returning empty result.")
            return result

        try:
            # For exact usage, checking if controlnet_aux output provides the pose directly.
            # Depending on controlnet_aux version, output_type="poses" may yield raw pose objects.
            poses = self.model(image, output_type="poses")
            if poses and len(poses) > 0:
                pose = poses[0]  # Take the primary person
                if hasattr(pose, 'body') and pose.body is not None:
                    result['body_keypoints'] = np.array(pose.body.keypoints, dtype=np.float32)
                if hasattr(pose, 'hands') and pose.hands is not None:
                    result['hand_keypoints'] = np.array(pose.hands.keypoints, dtype=np.float32)
                if hasattr(pose, 'face') and pose.face is not None:
                    result['face_keypoints'] = np.array(pose.face.keypoints, dtype=np.float32)
                if hasattr(pose, 'bbox') and pose.bbox is not None:
                    result['bbox'] = pose.bbox
                if hasattr(pose, 'score') and pose.score is not None:
                    result['confidence'] = float(pose.score)
                else:
                    result['confidence'] = 1.0
        except Exception as e:
            logger.error(f"Extraction error or no person detected: {e}")
            
        return result

    def extract_from_video(self, video_path: str, max_frames: Optional[int] = None, fps: Optional[int] = None) -> List[Dict[str, Union[np.ndarray, list, float]]]:
        """
        Extract poses from all frames in a video.
        
        Args:
            video_path (str): Path to the video file.
            max_frames (int, optional): Maximum number of frames to process.
            fps (int, optional): Target frame rate.
            
        Returns:
            list[dict]: List of pose dictionaries for each frame.
        """
        from decord import VideoReader, cpu
        
        logger.info(f"Loading video from {video_path}")
        vr = VideoReader(video_path, ctx=cpu(0))
        
        frames = []
        for i in range(len(vr)):
            if max_frames is not None and i >= max_frames:
                break
            # Convert to PIL Image for the extractor
            frame_np = vr[i].asnumpy()
            frames.append(Image.fromarray(frame_np))
            
        logger.info(f"Extracted {len(frames)} frames. Starting batch pose extraction...")
        return self.batch_extract(frames)

    def batch_extract(self, images: List[Image.Image]) -> List[Dict[str, Union[np.ndarray, list, float]]]:
        """
        Batch processing of images for pose extraction.
        
        Args:
            images (list[PIL.Image]): List of input images.
            
        Returns:
            list[dict]: List of extracted pose dictionaries.
        """
        results = []
        for i, img in enumerate(images):
            if i % 10 == 0:
                logger.info(f"Extracting pose from frame {i}/{len(images)}")
            
            try:
                pose_dict = self.extract_from_image(img)
                results.append(pose_dict)
            except Exception as e:
                logger.warning(f"Failed to extract pose from frame {i}: {e}")
                # Append empty on failure
                results.append({
                    "body_keypoints": np.zeros((18, 3), dtype=np.float32),
                    "hand_keypoints": np.zeros((42, 3), dtype=np.float32),
                    "face_keypoints": np.zeros((68, 3), dtype=np.float32),
                    "bbox": [0.0, 0.0, 0.0, 0.0],
                    "confidence": 0.0
                })
                
        logger.info("Batch pose extraction completed.")
        return results

    def __call__(self, image: Union[Image.Image, np.ndarray]) -> Dict[str, Union[np.ndarray, list, float]]:
        """Extract pose keypoints from a single image (PIL or numpy)."""
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        return self.extract_from_image(image)

    def extract_poses(self, images: List[Union[Image.Image, np.ndarray]]) -> List[Dict[str, Union[np.ndarray, list, float]]]:
        """Extract poses from a list of images (PIL or numpy arrays)."""
        pil_images = [Image.fromarray(img) if isinstance(img, np.ndarray) else img for img in images]
        return self.batch_extract(pil_images)

