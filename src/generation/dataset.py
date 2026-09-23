import os
import json
import random
import cv2
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from decord import VideoReader, cpu
import torchvision.transforms.functional as TF

class MotionTransferDataset(Dataset):
    """
    Dataset for Pose-Guided Human Video Generation.
    """
    def __init__(
        self, 
        data_dir: str, 
        resolution: tuple, 
        num_frames: int, 
        fps: int, 
        pose_type: str = 'dwpose', 
        transform=None
    ):
        self.data_dir = data_dir
        self.resolution = resolution  # (width, height)
        self.num_frames = num_frames
        self.fps = fps
        self.pose_type = pose_type
        self.transform = transform
        
        # Expected directory structure setup
        self.metadata_path = os.path.join(data_dir, 'metadata.csv')
        self.videos_dir = os.path.join(data_dir, 'videos')
        self.poses_dir = os.path.join(data_dir, 'poses')
        
        # Load metadata
        if os.path.exists(self.metadata_path):
            self.metadata = pd.read_csv(self.metadata_path)
        else:
            self.metadata = pd.DataFrame(columns=['video_id', 'duration', 'num_frames', 'quality_score'])
            
    def __len__(self):
        return len(self.metadata)
        
    def render_pose(self, keypoints: np.ndarray, width: int, height: int) -> np.ndarray:
        """Render a skeleton image from keypoints."""
        # Simple point rendering for skeleton approximation
        img = np.zeros((height, width, 3), dtype=np.uint8)
        for pt in keypoints:
            x, y = int(pt[0]), int(pt[1])
            if 0 <= x < width and 0 <= y < height:
                cv2.circle(img, (x, y), max(1, width // 64), (0, 255, 0), -1)
        return img
        
    def __getitem__(self, idx: int) -> dict:
        row = self.metadata.iloc[idx]
        video_id = row['video_id']
        
        video_path = os.path.join(self.videos_dir, f"{video_id}.mp4")
        pose_folder = os.path.join(self.poses_dir, video_id)
        
        # Load Video using Decord
        vr = VideoReader(video_path, ctx=cpu(0))
        total_frames = len(vr)
        
        # Random temporal sampling
        if total_frames > self.num_frames:
            start_frame = random.randint(0, total_frames - self.num_frames)
        else:
            start_frame = 0
            
        frame_indices = [min(start_frame + i, total_frames - 1) for i in range(self.num_frames)]
        frames = vr.get_batch(frame_indices).asnumpy()  # Shape: [T, H, W, C]
        
        # Random reference frame from the entire video
        reference_idx = random.randint(0, total_frames - 1)
        ref_image = vr[reference_idx].asnumpy()
        
        # Extract and render poses
        pose_images = []
        keypoints_list = []
        confidences_list = []
        
        for f_idx in frame_indices:
            pose_file = os.path.join(pose_folder, f"frame_{f_idx:03d}.json")
            if os.path.exists(pose_file):
                with open(pose_file, 'r') as f:
                    pose_data = json.load(f)
                    # Example format fallback
                    kpts = np.array(pose_data.get('keypoints', np.zeros((17, 3))))
                    conf = np.array(pose_data.get('confidence', np.ones(17)))
            else:
                kpts = np.zeros((17, 3))
                conf = np.zeros(17)
                
            pose_img = self.render_pose(kpts, self.resolution[0], self.resolution[1])
            pose_images.append(pose_img)
            keypoints_list.append(kpts)
            confidences_list.append(conf)
            
        # Resize inputs to match requested resolution
        # Usually requires resizing the raw numpy arrays, using simple reshape here for structure
        
        # Convert all to tensors
        # frames: [T, H, W, C] -> [T, C, H, W]
        video_tensor = torch.from_numpy(frames).permute(0, 3, 1, 2).float() / 255.0
        ref_tensor = torch.from_numpy(ref_image).permute(2, 0, 1).float() / 255.0
        pose_tensor = torch.from_numpy(np.stack(pose_images)).permute(0, 3, 1, 2).float() / 255.0
        
        kpts_tensor = torch.from_numpy(np.stack(keypoints_list)).float()
        conf_tensor = torch.from_numpy(np.stack(confidences_list)).float()
        
        # Data Augmentations
        # Apply identical spatial transforms to video, ref, and pose
        if random.random() > 0.5:
            video_tensor = TF.hflip(video_tensor)
            ref_tensor = TF.hflip(ref_tensor)
            pose_tensor = TF.hflip(pose_tensor)
            # Flip keypoints X coordinates
            kpts_tensor[:, :, 0] = self.resolution[0] - kpts_tensor[:, :, 0]
            
        # Color jitter on reference only
        if random.random() > 0.5:
            ref_tensor = TF.adjust_brightness(ref_tensor, brightness_factor=random.uniform(0.8, 1.2))
            
        return {
            'reference_image': ref_tensor,
            'video_frames': video_tensor,
            'pose_images': pose_tensor,
            'pose_keypoints': kpts_tensor,
            'confidence_scores': conf_tensor
        }
