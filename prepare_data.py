#!/usr/bin/env python3
"""Data preparation pipeline for AI Motion Transfer training.

Downloads and preprocesses human dance video datasets:
1. Downloads videos from supported sources (AIST++, TikTok dataset, Pexels)
2. Extracts frames at target FPS and resolution
3. Runs DWPose extraction on all frames
4. Renders pose skeleton images
5. Creates metadata.csv
6. Validates and filters low-quality samples

Usage:
    python prepare_data.py --source aist++ --output_dir ./data --max_videos 1000
    python prepare_data.py --source local --input_dir /path/to/videos --output_dir ./data
    python prepare_data.py --source pexels --query "person dancing" --max_videos 500 --output_dir ./data
"""

import argparse
import csv
import json
import logging
import os
import shutil
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import cv2
import requests
from tqdm import tqdm

from src.pose.extractor import DWPoseExtractor
from src.pose.renderer import PoseRenderer
from src.utils.video_io import load_video, extract_frames

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def download_aist_plus_plus(output_dir: Path, max_videos: int) -> None:
    """Download AIST++ dataset videos."""
    logger.info(f"Downloading AIST++ videos to {output_dir}")
    # Placeholder for actual AIST++ download logic, as it requires accepting terms
    # and downloading via specific scripts. Here we mock the process.
    logger.warning("AIST++ download requires manual acceptance of terms. Mocking download.")
    output_dir.mkdir(parents=True, exist_ok=True)
    for i in tqdm(range(max_videos), desc="Mocking AIST++ download"):
        mock_file = output_dir / f"aist_mock_{i:04d}.mp4"
        if not mock_file.exists():
            with open(mock_file, "w") as f:
                f.write("mock video data")


def download_pexels_videos(query: str, output_dir: Path, max_videos: int, api_key: str) -> None:
    """Download free stock videos from Pexels API."""
    if not api_key:
        logger.error("Pexels API key is required to download videos.")
        return

    logger.info(f"Searching Pexels for '{query}'...")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    headers = {"Authorization": api_key}
    url = f"https://api.pexels.com/videos/search?query={query}&per_page=80"
    
    downloaded = 0
    while url and downloaded < max_videos:
        response = requests.get(url, headers=headers)
        if response.status_code != 200:
            logger.error(f"Failed to fetch from Pexels: {response.text}")
            break
            
        data = response.json()
        videos = data.get("videos", [])
        
        for video in videos:
            if downloaded >= max_videos:
                break
                
            video_id = video["id"]
            video_files = video.get("video_files", [])
            if not video_files:
                continue
                
            # Get the highest quality HD video
            hd_files = [f for f in video_files if f.get("quality") == "hd"]
            target_file = hd_files[0] if hd_files else video_files[0]
            download_link = target_file["link"]
            
            out_path = output_dir / f"pexels_{video_id}.mp4"
            if out_path.exists():
                logger.info(f"Skipping {out_path.name}, already exists.")
                downloaded += 1
                continue
                
            try:
                urllib.request.urlretrieve(download_link, out_path)
                downloaded += 1
                logger.info(f"Downloaded {out_path.name} ({downloaded}/{max_videos})")
            except Exception as e:
                logger.error(f"Failed to download {video_id}: {e}")
                
        url = data.get("next_page")


def process_local_videos(input_dir: Path, output_dir: Path) -> None:
    """Copy local videos to the output directory."""
    logger.info(f"Processing local videos from {input_dir}")
    if not input_dir.exists():
        logger.error(f"Input directory {input_dir} does not exist.")
        return
        
    output_dir.mkdir(parents=True, exist_ok=True)
    video_extensions = {".mp4", ".avi", ".mov", ".mkv"}
    
    for file_path in input_dir.iterdir():
        if file_path.suffix.lower() in video_extensions:
            dest_path = output_dir / file_path.name
            if not dest_path.exists():
                shutil.copy2(file_path, dest_path)
                logger.info(f"Copied {file_path.name}")
            else:
                logger.info(f"Skipped {file_path.name} (already exists)")


def preprocess_video(video_path: Path, output_dir: Path, target_fps: int, target_resolution: str) -> Optional[Dict[str, Any]]:
    """Extract frames and resize."""
    video_name = video_path.stem
    frames_dir = output_dir / video_name / "frames"
    
    if frames_dir.exists() and len(list(frames_dir.glob("*.jpg"))) > 0:
        logger.info(f"Skipping {video_name}, frames already extracted.")
        num_frames = len(list(frames_dir.glob("*.jpg")))
        return {"video_id": video_name, "num_frames": num_frames, "frames_dir": frames_dir}
        
    frames_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        w_str, h_str = target_resolution.split("x")
        target_w, target_h = int(w_str), int(h_str)
        
        frames = extract_frames(str(video_path), fps=target_fps)
        if not frames:
            return None
            
        for i, frame in enumerate(frames):
            resized_frame = cv2.resize(frame, (target_w, target_h))
            cv2.imwrite(str(frames_dir / f"frame_{i:04d}.jpg"), resized_frame)
            
        return {"video_id": video_name, "num_frames": len(frames), "frames_dir": frames_dir}
    except Exception as e:
        logger.error(f"Error preprocessing {video_path.name}: {e}")
        return None


def extract_poses(frames_dir: Path, output_dir: Path) -> Path:
    """Run DWPose on all frames, save keypoints as JSON."""
    video_id = frames_dir.parent.name
    poses_dir = output_dir / video_id / "poses"
    
    if poses_dir.exists() and len(list(poses_dir.glob("*.json"))) > 0:
        logger.info(f"Skipping {video_id} pose extraction, already done.")
        return poses_dir
        
    poses_dir.mkdir(parents=True, exist_ok=True)
    extractor = DWPoseExtractor()
    
    frame_files = sorted(frames_dir.glob("*.jpg"))
    for frame_file in frame_files:
        frame = cv2.imread(str(frame_file))
        if frame is None:
            continue
            
        pose_data = extractor(frame)
        pose_path = poses_dir / f"{frame_file.stem}.json"
        
        with open(pose_path, "w") as f:
            json.dump(pose_data, f)
            
    return poses_dir


def render_pose_images(poses_dir: Path, output_dir: Path, canvas_size: Tuple[int, int]) -> Path:
    """Render skeleton images from pose JSONs."""
    video_id = poses_dir.parent.name
    render_dir = output_dir / video_id / "pose_images"
    
    if render_dir.exists() and len(list(render_dir.glob("*.jpg"))) > 0:
        logger.info(f"Skipping {video_id} pose rendering, already done.")
        return render_dir
        
    render_dir.mkdir(parents=True, exist_ok=True)
    renderer = PoseRenderer()
    
    pose_files = sorted(poses_dir.glob("*.json"))
    for pose_file in pose_files:
        with open(pose_file, "r") as f:
            pose_data = json.load(f)
            
        rendered_image = renderer(pose_data, canvas_size)
        render_path = render_dir / f"{pose_file.stem}.jpg"
        cv2.imwrite(str(render_path), rendered_image)
        
    return render_dir


def create_metadata(data_dir: Path) -> None:
    """Create metadata.csv."""
    logger.info("Creating metadata.csv")
    csv_path = data_dir / "metadata.csv"
    
    metadata = []
    for video_dir in data_dir.iterdir():
        if not video_dir.is_dir() or video_dir.name in ["raw_videos"]:
            continue
            
        video_id = video_dir.name
        frames_dir = video_dir / "frames"
        poses_dir = video_dir / "poses"
        
        if not frames_dir.exists() or not poses_dir.exists():
            continue
            
        num_frames = len(list(frames_dir.glob("*.jpg")))
        pose_files = list(poses_dir.glob("*.json"))
        
        if num_frames == 0 or len(pose_files) == 0:
            continue
            
        # Calculate mean confidence
        total_conf = 0.0
        valid_poses = 0
        for pose_file in pose_files:
            try:
                with open(pose_file, "r") as f:
                    pose_data = json.load(f)
                    # Simplified confidence calculation
                    conf = pose_data.get("score", 0.0)
                    total_conf += conf
                    valid_poses += 1
            except Exception:
                pass
                
        mean_conf = total_conf / valid_poses if valid_poses > 0 else 0.0
        
        # Determine resolution from first frame
        first_frame = next(frames_dir.glob("*.jpg"), None)
        resolution = "unknown"
        if first_frame:
            img = cv2.imread(str(first_frame))
            if img is not None:
                h, w = img.shape[:2]
                resolution = f"{w}x{h}"
                
        duration = num_frames / 15.0  # Assuming target FPS = 15
        quality_score = mean_conf * min(1.0, num_frames / 100.0) # Dummy quality score
        
        metadata.append({
            "video_id": video_id,
            "num_frames": num_frames,
            "duration": duration,
            "resolution": resolution,
            "quality_score": quality_score,
            "pose_confidence_mean": mean_conf
        })
        
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "video_id", "num_frames", "duration", "resolution", 
            "quality_score", "pose_confidence_mean"
        ])
        writer.writeheader()
        writer.writerows(metadata)
        
    logger.info(f"Metadata saved to {csv_path} with {len(metadata)} entries.")


def filter_low_quality(data_dir: Path, min_confidence: float, min_frames: int) -> None:
    """Remove samples with low pose confidence or too few frames."""
    logger.info("Filtering low quality samples")
    csv_path = data_dir / "metadata.csv"
    if not csv_path.exists():
        logger.error("Metadata not found, run create_metadata first.")
        return
        
    to_remove = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if float(row["pose_confidence_mean"]) < min_confidence or int(row["num_frames"]) < min_frames:
                to_remove.append(row["video_id"])
                
    for video_id in to_remove:
        video_dir = data_dir / video_id
        if video_dir.exists():
            shutil.rmtree(video_dir)
            logger.info(f"Removed low quality sample: {video_id}")
            
    # Recreate metadata after filtering
    create_metadata(data_dir)


def validate_dataset(data_dir: Path) -> None:
    """Check integrity of the dataset."""
    logger.info("Validating dataset integrity")
    csv_path = data_dir / "metadata.csv"
    if not csv_path.exists():
        logger.error("Metadata not found.")
        return
        
    invalid_samples = 0
    total_frames = 0
    total_videos = 0
    total_conf = 0.0
    
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            video_id = row["video_id"]
            num_frames = int(row["num_frames"])
            conf = float(row["pose_confidence_mean"])
            
            video_dir = data_dir / video_id
            frames_dir = video_dir / "frames"
            poses_dir = video_dir / "poses"
            render_dir = video_dir / "pose_images"
            
            frame_count = len(list(frames_dir.glob("*.jpg"))) if frames_dir.exists() else 0
            pose_count = len(list(poses_dir.glob("*.json"))) if poses_dir.exists() else 0
            render_count = len(list(render_dir.glob("*.jpg"))) if render_dir.exists() else 0
            
            if not (frame_count == pose_count == render_count == num_frames):
                logger.warning(f"Inconsistent counts for {video_id}: frames={frame_count}, poses={pose_count}, renders={render_count}, metadata={num_frames}")
                invalid_samples += 1
            else:
                total_videos += 1
                total_frames += frame_count
                total_conf += conf
                
    logger.info("Validation complete.")
    logger.info(f"Total valid videos: {total_videos}")
    logger.info(f"Total valid frames: {total_frames}")
    if total_videos > 0:
        logger.info(f"Average pose confidence: {total_conf / total_videos:.4f}")
    if invalid_samples > 0:
        logger.warning(f"Found {invalid_samples} invalid samples.")
        
    # Get disk usage
    total_size = sum(f.stat().st_size for f in data_dir.glob('**/*') if f.is_file())
    logger.info(f"Total disk usage: {total_size / (1024**3):.2f} GB")


def main():
    parser = argparse.ArgumentParser(description="Data preparation pipeline for AI Motion Transfer")
    parser.add_argument("--source", type=str, choices=["aist++", "pexels", "local"], required=True, help="Video source")
    parser.add_argument("--input_dir", type=str, help="Input directory for local videos")
    parser.add_argument("--output_dir", type=str, default="./data", help="Output directory")
    parser.add_argument("--max_videos", type=int, default=1000, help="Maximum number of videos to download")
    parser.add_argument("--query", type=str, default="person dancing", help="Search query for Pexels")
    parser.add_argument("--api_key", type=str, default="", help="Pexels API key")
    parser.add_argument("--target_fps", type=int, default=15, help="Target FPS for extraction")
    parser.add_argument("--target_resolution", type=str, default="832x480", help="Target resolution WxH")
    parser.add_argument("--min_confidence", type=float, default=0.5, help="Minimum pose confidence")
    parser.add_argument("--min_frames", type=int, default=30, help="Minimum number of frames")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of worker threads")
    
    args = parser.parse_args()
    
    out_dir = Path(args.output_dir)
    raw_dir = out_dir / "raw_videos"
    raw_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Acquire Videos
    if args.source == "aist++":
        download_aist_plus_plus(raw_dir, args.max_videos)
    elif args.source == "pexels":
        download_pexels_videos(args.query, raw_dir, args.max_videos, args.api_key)
    elif args.source == "local":
        if not args.input_dir:
            logger.error("--input_dir is required when source is 'local'")
            return
        process_local_videos(Path(args.input_dir), raw_dir)
        
    video_files = list(raw_dir.glob("*.mp4")) + list(raw_dir.glob("*.avi")) + list(raw_dir.glob("*.mov"))
    logger.info(f"Found {len(video_files)} videos to process.")
    
    # 2. Preprocess, Extract, Render
    w_str, h_str = args.target_resolution.split("x")
    canvas_size = (int(w_str), int(h_str))
    
    for video_path in tqdm(video_files, desc="Processing videos"):
        # Preprocess
        res = preprocess_video(video_path, out_dir, args.target_fps, args.target_resolution)
        if not res:
            continue
            
        frames_dir = res["frames_dir"]
        
        # Extract Poses
        poses_dir = extract_poses(frames_dir, out_dir)
        
        # Render Pose Images
        render_pose_images(poses_dir, out_dir, canvas_size)
        
    # 3. Create Metadata
    create_metadata(out_dir)
    
    # 4. Filter
    filter_low_quality(out_dir, args.min_confidence, args.min_frames)
    
    # 5. Validate
    validate_dataset(out_dir)
    
if __name__ == "__main__":
    main()
