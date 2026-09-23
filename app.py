"""Gradio web application for AI Motion Transfer.

Supports:
- Motion Transfer: Standard (25 steps) & Distilled Few-Step (4-8 steps)
- Pose Preview: extraction & skeleton visualization
- Gallery: review previously generated animations
"""

import os
import json
import time
import shutil
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple, List, Dict, Any

import gradio as gr
import torch
import cv2
import numpy as np

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("app")

# Import pipeline and pose components
from src.pipeline import MotionTransferPipeline
from src.pose.extractor import DWPoseExtractor
from src.pose.renderer import PoseRenderer

# Global state for models
pipeline = None
pose_extractor = None
pose_renderer = None

OUTPUT_DIR = Path("./outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_models():
    """Loads all required models into GPU."""
    global pipeline, pose_extractor, pose_renderer
    logger.info("Initializing models...")
    try:
        if pipeline is None:
            pipeline = MotionTransferPipeline()

        if pose_extractor is None:
            pose_extractor = DWPoseExtractor()

        if pose_renderer is None:
            pose_renderer = PoseRenderer()

        logger.info("Models loaded successfully.")
        return "Models loaded successfully!"
    except Exception as e:
        logger.error(f"Failed to load models: {str(e)}")
        return f"Error loading models: {str(e)}"


def extract_and_visualize_pose(video_path: str) -> Tuple[Optional[str], str]:
    """Extracts poses from video and creates a visualization."""
    if not video_path:
        return None, "Vui lòng tải lên video."

    if pose_extractor is None or pose_renderer is None:
        load_models()

    try:
        logger.info(f"Extracting poses from {video_path}")
        cap = cv2.VideoCapture(video_path)
        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
        cap.release()

        if not frames:
            return None, "Không đọc được frame nào từ video."

        poses = pose_extractor.extract_poses(frames)
        rendered_frames = pose_renderer.render_poses(poses, (frames[0].shape[0], frames[0].shape[1]))

        out_path = OUTPUT_DIR / "temp_pose_preview.mp4"
        if out_path.exists():
            out_path.unlink()

        height, width, _ = rendered_frames[0].shape
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(str(out_path), fourcc, 30, (width, height))
        for frame in rendered_frames:
            out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        out.release()

        return str(out_path), f"Hoàn thành trích xuất {len(rendered_frames)} frames pose!"
    except Exception as e:
        logger.error(f"Error extracting pose: {str(e)}")
        return None, f"Lỗi: {str(e)}"


def generate_video(
    ref_image: str,
    driving_video: str,
    model_mode: str,
    num_steps: int,
    guidance_scale: float,
    seed: int,
    resolution: str,
    enable_face_restoration: bool,
    enable_teacache: bool,
    progress=gr.Progress(),
) -> Tuple[Optional[str], str]:
    """Generates a motion transfer video."""
    if not ref_image or not driving_video:
        return None, "Vui lòng cung cấp cả ảnh tham chiếu và video chuyển động."

    global pipeline
    if pipeline is None:
        load_models()

    try:
        progress(0, desc="Đang chuẩn bị...")
        width, height = map(int, resolution.split("x"))
        is_distilled = "Distilled" in model_mode or num_steps <= 8

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = OUTPUT_DIR / timestamp
        run_dir.mkdir(parents=True, exist_ok=True)

        shutil.copy(ref_image, run_dir / "reference.png")
        shutil.copy(driving_video, run_dir / "driving.mp4")

        config = {
            "model_mode": model_mode,
            "is_distilled": is_distilled,
            "num_steps": num_steps,
            "guidance_scale": guidance_scale,
            "seed": seed,
            "resolution": resolution,
            "enable_face_restoration": enable_face_restoration,
            "enable_teacache": enable_teacache,
            "timestamp": timestamp,
        }
        with open(run_dir / "config.json", "w", encoding="utf-8") as f:
            json.dump(config, f, indent=4)

        progress(0.2, desc="Đang trích xuất và căn chỉnh pose...")
        output_video_path = str(run_dir / "output.mp4")

        pipeline.run(
            ref_image=ref_image,
            driving_video=driving_video,
            output_path=output_video_path,
            num_steps=num_steps,
            guidance_scale=guidance_scale,
            seed=seed,
            width=width,
            height=height,
            enable_face_restoration=enable_face_restoration,
            enable_teacache=enable_teacache,
            is_distilled=is_distilled,
            progress_callback=lambda step, total: progress(
                0.2 + 0.7 * (step / total),
                desc=f"Generating frame ({step}/{total} steps - {'Distilled' if is_distilled else 'Standard'})...",
            ),
        )

        progress(1.0, desc="Hoàn thành!")
        mode_label = "Distilled Few-Step ⚡" if is_distilled else "Standard"
        return output_video_path, f"Tạo video thành công ({mode_label}, {num_steps} steps)!"

    except Exception as e:
        logger.error(f"Generation error: {str(e)}", exc_info=True)
        return None, f"Lỗi trong quá trình tạo: {str(e)}"


def load_gallery() -> List[List[Any]]:
    """Loads previous results for the gallery."""
    results = []
    if not OUTPUT_DIR.exists():
        return results

    for run_dir in sorted(OUTPUT_DIR.iterdir(), reverse=True):
        if not run_dir.is_dir():
            continue

        ref_path = run_dir / "reference.png"
        drive_path = run_dir / "driving.mp4"
        out_path = run_dir / "output.mp4"
        config_path = run_dir / "config.json"

        if ref_path.exists() and drive_path.exists() and out_path.exists():
            timestamp = run_dir.name
            if config_path.exists():
                try:
                    with open(config_path, "r", encoding="utf-8") as f:
                        conf = json.load(f)
                        timestamp = conf.get("timestamp", timestamp)
                except Exception:
                    pass

            results.append([
                str(ref_path),
                str(drive_path),
                str(out_path),
                timestamp,
                str(run_dir),
            ])

    return results


def delete_gallery_item(run_dir_str: str) -> List[List[Any]]:
    """Deletes a gallery item and refreshes the list."""
    try:
        run_dir = Path(run_dir_str)
        if run_dir.exists() and run_dir.is_dir():
            shutil.rmtree(run_dir)
            logger.info(f"Deleted run directory: {run_dir_str}")
    except Exception as e:
        logger.error(f"Failed to delete {run_dir_str}: {str(e)}")

    return load_gallery()


def build_ui():
    """Builds the Gradio interface."""
    css = """
    .gradio-container {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    footer {
        text-align: center;
        margin-top: 2rem;
        padding-top: 1rem;
        border-top: 1px solid var(--border-color-primary);
    }
    """

    theme = gr.themes.Default(
        primary_hue="blue",
        secondary_hue="indigo",
        neutral_hue="slate",
    ).set(
        body_background_fill="*neutral_950",
        body_background_fill_dark="*neutral_950",
        body_text_color="*neutral_100",
        background_fill_primary="*neutral_900",
        background_fill_secondary="*neutral_800",
        border_color_primary="*neutral_700",
    )

    with gr.Blocks(theme=theme, css=css, title="AI Motion Transfer - Pose-Guided Video Generation") as app:
        gr.Markdown("""
        # 🎬 AI Motion Transfer
        ## Pose-Guided Video Generation (Wan2.1-14B + LoRA + PoseEncoder)
        Copy chuyển động từ video nguồn sang ảnh tham chiếu. Hỗ trợ **Consistency Distillation (4-8 steps ⚡)** và **TeaCache** cho tốc độ sinh video siêu tốc.
        """)

        with gr.Tabs():
            # TAB 1: MOTION TRANSFER
            with gr.Tab("Motion Transfer"):
                with gr.Row():
                    with gr.Column():
                        ref_image = gr.Image(type="filepath", label="Ảnh tham chiếu (Reference Image)")
                        driving_video = gr.Video(label="Video chuyển động (Driving Video)")

                        model_mode = gr.Radio(
                            choices=[
                                "Distilled Few-Step (4-8 steps ⚡ ~15s)",
                                "Standard Multi-Step (20-25 steps 🎨 Full quality)",
                            ],
                            value="Distilled Few-Step (4-8 steps ⚡ ~15s)",
                            label="Chế độ sinh video (Sampling Mode)",
                        )

                        with gr.Accordion("Cài đặt thông số chi tiết (Parameters)", open=False):
                            num_steps = gr.Slider(minimum=4, maximum=50, value=8, step=1, label="Số bước Inference (Steps)")
                            guidance_scale = gr.Slider(minimum=1.0, maximum=5.0, value=2.0, step=0.1, label="Guidance Scale")
                            seed = gr.Number(value=-1, label="Seed (-1 để ngẫu nhiên)")
                            resolution = gr.Dropdown(choices=["512x512", "768x768", "832x480"], value="832x480", label="Độ phân giải (Resolution)")
                            enable_face_restoration = gr.Checkbox(value=False, label="Bật phục hồi khuôn mặt (Face Restoration)")
                            enable_teacache = gr.Checkbox(value=True, label="Bật TeaCache (+2-4x tốc độ)")

                        generate_btn = gr.Button("🚀 Tạo Video Ngay", variant="primary")
                        status_text = gr.Textbox(label="Trạng thái", interactive=False)

                    with gr.Column():
                        output_video = gr.Video(label="Video kết quả (Generated Video)")

                        with gr.Row():
                            gr.Markdown("### So sánh chuyển động")
                        with gr.Row():
                            comp_driving = gr.Video(label="Nguồn (Driving)", interactive=False)
                            comp_output = gr.Video(label="Kết quả (Generated)", interactive=False)

                # Dynamically adjust steps when switching modes
                def on_mode_change(mode):
                    if "Distilled" in mode:
                        return gr.update(value=8, maximum=16)
                    else:
                        return gr.update(value=25, maximum=50)

                model_mode.change(fn=on_mode_change, inputs=[model_mode], outputs=[num_steps])

                def on_generate(*args):
                    out_vid, status = generate_video(*args)
                    return out_vid, status, args[1], out_vid

                generate_btn.click(
                    fn=on_generate,
                    inputs=[
                        ref_image,
                        driving_video,
                        model_mode,
                        num_steps,
                        guidance_scale,
                        seed,
                        resolution,
                        enable_face_restoration,
                        enable_teacache,
                    ],
                    outputs=[output_video, status_text, comp_driving, comp_output],
                )

            # TAB 2: POSE PREVIEW
            with gr.Tab("Pose Preview"):
                with gr.Row():
                    with gr.Column():
                        pose_input_video = gr.Video(label="Video nguồn")
                        extract_btn = gr.Button("Trích xuất & Hiển thị Pose", variant="primary")
                        pose_status = gr.Textbox(label="Trạng thái", interactive=False)

                    with gr.Column():
                        pose_output_video = gr.Video(label="Pose Preview")

                extract_btn.click(
                    fn=extract_and_visualize_pose,
                    inputs=[pose_input_video],
                    outputs=[pose_output_video, pose_status],
                )

            # TAB 3: GALLERY
            with gr.Tab("Gallery"):
                gr.Markdown("Kết quả đã tạo trước đây (lưu tại `./outputs`)")
                refresh_btn = gr.Button("Làm mới (Refresh)")

                gallery_output = gr.Dataframe(
                    headers=["Ảnh Tham Chiếu", "Video Nguồn", "Video Kết Quả", "Thời Gian", "Đường dẫn"],
                    datatype=["str", "str", "str", "str", "str"],
                    col_count=(5, "fixed"),
                    interactive=False,
                )

                with gr.Row():
                    delete_path = gr.Textbox(label="Đường dẫn cần xóa (Copy từ bảng trên)")
                    delete_btn = gr.Button("Xóa thư mục (Delete)", variant="stop")

                refresh_btn.click(fn=load_gallery, inputs=[], outputs=[gallery_output])

                delete_btn.click(
                    fn=delete_gallery_item,
                    inputs=[delete_path],
                    outputs=[gallery_output],
                )

                app.load(fn=load_gallery, outputs=[gallery_output])

        gr.Markdown(
            "<footer style='text-align: center; margin-top: 20px;'>"
            "Powered by Wan2.1-14B + UniAnimate-DiT + Latent Consistency Distillation</footer>"
        )
        app.load(fn=load_models, outputs=[status_text])

    return app


if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    app = build_ui()
    app.launch(server_name="0.0.0.0", server_port=7860, share=True)
