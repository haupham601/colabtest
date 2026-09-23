#!/usr/bin/env python3
"""Consistency Distillation training script for AI Motion Transfer.

Distills a trained 25-step Wan2.1-14B LoRA + PoseEncoder teacher model
into a 4-8 step fast student model using Latent Consistency Distillation (LCD)
and Video Temporal Consistency Loss.

Usage:
    # Distill trained teacher checkpoint to 8 steps
    accelerate launch train_distill.py \
        --config config/default.yaml \
        --teacher_checkpoint ./checkpoints/final \
        --output_dir ./checkpoints_distill

    # Distill with custom target steps
    accelerate launch train_distill.py \
        --config config/default.yaml \
        --teacher_checkpoint ./checkpoints/final \
        --num_steps 4 \
        --output_dir ./checkpoints_distill_4step
"""

import os
import sys
import math
import copy
import argparse
import logging
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from omegaconf import OmegaConf
from accelerate import Accelerator
from accelerate.utils import set_seed
from peft import LoraConfig, get_peft_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("train_distill")

from src.generation.pose_encoder import PoseEncoder3D
from src.generation.dataset import MotionTransferDataset
from src.generation.distillation import (
    ConsistencyDistillationLoss,
    TemporalConsistencyLoss,
    DDIMSolver,
    EMAModel,
    get_predicted_x0,
    build_skip_schedule,
)
from train import load_wan_model, get_cosine_schedule_with_warmup


def copy_model_weights(source: nn.Module, target: nn.Module) -> None:
    """Copy weights from source model to target model with same architecture."""
    target.load_state_dict(source.state_dict(), strict=False)


def main():
    parser = argparse.ArgumentParser(description="Train Consistency Distillation for AI Motion Transfer")
    parser.add_argument("--config", type=str, default="config/default.yaml", help="Path to config file")
    parser.add_argument("--teacher_checkpoint", type=str, default=None, help="Path to teacher checkpoint directory")
    parser.add_argument("--output_dir", type=str, default="./checkpoints_distill", help="Output directory")
    parser.add_argument("--num_steps", type=int, default=None, help="Target distilled steps (e.g. 4 or 8)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    # Load configuration
    config = OmegaConf.load(args.config)
    os.makedirs(args.output_dir, exist_ok=True)
    OmegaConf.save(config, os.path.join(args.output_dir, "config.yaml"))

    distill_cfg = config.get("distillation", {})
    target_steps = args.num_steps or distill_cfg.get("num_steps", 8)
    teacher_ckpt = args.teacher_checkpoint or distill_cfg.get("teacher_checkpoint", "./checkpoints/final")
    skipping_steps = distill_cfg.get("skipping_steps", 2)
    huber_c = distill_cfg.get("huber_c", 0.001)
    temporal_weight = distill_cfg.get("temporal_weight", 0.2)
    ema_decay = distill_cfg.get("ema_decay", 0.999)
    learning_rate = distill_cfg.get("lr", 2e-5)
    max_steps = distill_cfg.get("max_steps", 5000)
    save_every = distill_cfg.get("save_every", 500)
    batch_size = distill_cfg.get("batch_size", 2)
    grad_accum = distill_cfg.get("grad_accum", 4)

    # Initialize accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps=grad_accum,
        mixed_precision=config.training.mixed_precision,
        log_with="wandb",
    )
    set_seed(args.seed)
    device = accelerator.device

    if accelerator.is_main_process:
        accelerator.init_trackers(
            project_name="AI-Motion-Transfer-Distill",
            config={
                **OmegaConf.to_container(config, resolve=True),
                "distill_target_steps": target_steps,
                "skipping_steps": skipping_steps,
            },
        )

    # 1. Load base models
    logger.info("Loading base Wan2.1 model and VAE...")
    dit_base, vae, text_encoder, noise_scheduler = load_wan_model(config, device)
    vae.requires_grad_(False)
    if text_encoder is not None:
        text_encoder.requires_grad_(False)
    dit_base.requires_grad_(False)

    # 2. Setup Teacher model (Frozen)
    logger.info("Configuring Teacher model...")
    teacher_lora_config = LoraConfig(
        r=config.lora.rank,
        lora_alpha=config.lora.alpha,
        target_modules=list(config.lora.target_modules),
        lora_dropout=0.0,
        bias="none",
    )
    teacher_dit = get_peft_model(copy.deepcopy(dit_base), teacher_lora_config)

    # Load teacher LoRA weights if available
    teacher_lora_path = os.path.join(teacher_ckpt, "lora")
    if os.path.exists(teacher_lora_path):
        logger.info(f"Loading trained teacher LoRA weights from {teacher_lora_path}")
        teacher_dit.load_adapter(teacher_lora_path, adapter_name="default")
    else:
        logger.warning(f"Teacher LoRA weights not found at {teacher_lora_path}. Initializing with base LoRA.")
    teacher_dit.requires_grad_(False)
    teacher_dit.eval()

    # 3. Setup Pose Encoder (Frozen from teacher)
    hidden_dim = 1280
    pose_encoder = PoseEncoder3D(
        in_channels=config.pose_encoder.in_channels,
        base_channels=config.pose_encoder.base_channels,
        num_blocks=config.pose_encoder.num_blocks,
        output_channels=hidden_dim,
    ).to(device)

    pose_encoder_path = os.path.join(teacher_ckpt, "pose_encoder.pt")
    if os.path.exists(pose_encoder_path):
        logger.info(f"Loading trained PoseEncoder from {pose_encoder_path}")
        pose_encoder.load_state_dict(torch.load(pose_encoder_path, map_location=device))
    else:
        logger.warning(f"PoseEncoder checkpoint not found at {pose_encoder_path}. Using initial weights.")
    pose_encoder.requires_grad_(False)
    pose_encoder.eval()

    # 4. Setup Student model (Trainable LoRA)
    logger.info(f"Configuring Student model for {target_steps}-step consistency distillation...")
    student_lora_config = LoraConfig(
        r=config.lora.rank,
        lora_alpha=config.lora.alpha,
        target_modules=list(config.lora.target_modules),
        lora_dropout=0.0,
        bias="none",
    )
    student_dit = get_peft_model(dit_base, student_lora_config)
    # Warm-start student with teacher weights
    copy_model_weights(teacher_dit, student_dit)
    student_dit.train()

    # Setup EMA model for stable student targets
    logger.info(f"Initializing EMA model tracking student weights (decay={ema_decay})...")
    ema_model = EMAModel(student_dit, decay=ema_decay, update_after_step=50)

    # 5. Distillation Solvers & Loss functions
    solver = DDIMSolver(num_train_timesteps=1000).to(device)
    cd_loss_fn = ConsistencyDistillationLoss(
        num_train_timesteps=1000,
        num_ddim_steps=distill_cfg.get("teacher_steps", 25),
        skipping_steps=skipping_steps,
        huber_c=huber_c,
        use_huber=True,
    ).to(device)
    temporal_loss_fn = TemporalConsistencyLoss().to(device)

    # 6. Optimizer & Scheduler
    optimizer = torch.optim.AdamW(
        [p for p in student_dit.parameters() if p.requires_grad],
        lr=learning_rate,
        betas=(0.9, 0.999),
        weight_decay=0.01,
        eps=1e-8,
    )
    num_warmup_steps = int(max_steps * 0.05)
    lr_scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, max_steps)

    # 7. Dataset & DataLoader
    logger.info(f"Loading dataset from {config.data.dataset_path}")
    dataset = MotionTransferDataset(
        data_dir=config.data.dataset_path,
        resolution=tuple(config.data.resolution),
        num_frames=config.data.num_frames,
        fps=config.data.fps,
        pose_type=config.data.pose_type,
    )

    if len(dataset) == 0:
        logger.warning("Dataset is empty. Using synthetic fallback samples for initialization test.")
        from torch.utils.data import TensorDataset

        num_samples = 32
        res = config.data.resolution
        dataset = TensorDataset(
            torch.randn(num_samples, 3, res[1], res[0]),
            torch.randn(num_samples, config.data.num_frames, 3, res[1], res[0]),
            torch.randn(num_samples, config.data.num_frames, 3, res[1], res[0]),
            torch.randn(num_samples, config.data.num_frames, 18, 3),
            torch.ones(num_samples, config.data.num_frames, 18),
        )

        class SyntheticWrapper:
            def __init__(self, ds):
                self.ds = ds
            def __len__(self):
                return len(self.ds)
            def __getitem__(self, idx):
                ref, vid, pose, kpts, conf = self.ds[idx]
                return {
                    "reference_image": ref,
                    "video_frames": vid,
                    "pose_images": pose,
                    "pose_keypoints": kpts,
                    "confidence_scores": conf,
                }

        dataset = SyntheticWrapper(dataset)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        drop_last=True,
    )

    # Prepare with accelerator
    student_dit, optimizer, dataloader, lr_scheduler = accelerator.prepare(
        student_dit, optimizer, dataloader, lr_scheduler
    )

    logger.info("=" * 70)
    logger.info(f"⚡ Starting Consistency Distillation to {target_steps} steps")
    logger.info(f"  Max Steps: {max_steps}")
    logger.info(f"  Learning Rate: {learning_rate}")
    logger.info(f"  Skipping Steps: {skipping_steps}")
    logger.info(f"  Temporal Weight: {temporal_weight}")
    logger.info("=" * 70)

    global_step = 0
    while global_step < max_steps:
        for batch in dataloader:
            if global_step >= max_steps:
                break

            with accelerator.accumulate(student_dit):
                video_frames = batch["video_frames"]
                pose_images = batch["pose_images"]
                b, t, c, h, w = video_frames.shape

                # 1. Encode video frames to latents with frozen VAE
                with torch.no_grad():
                    video_3d = video_frames.permute(0, 2, 1, 3, 4)
                    latents = vae.encode(video_3d).sample()
                    # Pose embedding from frozen PoseEncoder
                    pose_embedding = pose_encoder(pose_images)

                # 2. Sample ODE timestep pair (t, t - k)
                timesteps_t, timesteps_prev = cd_loss_fn.sample_timestep_pairs(b, device)

                # 3. Add noise to sample x_t
                noise = torch.randn_like(latents)
                noisy_latents = solver.add_noise(latents, noise, timesteps_t)

                # 4. Student forward pass at timestep t
                student_noise_pred = student_dit(
                    noisy_latents,
                    timestep=timesteps_t,
                    pose_embedding=pose_embedding,
                )
                student_pred_x0 = get_predicted_x0(
                    student_noise_pred,
                    noisy_latents,
                    timesteps_t,
                    solver.alphas_cumprod,
                    prediction_type="epsilon",
                )

                # 5. Teacher step: compute trajectory to x_{t-k}
                with torch.no_grad():
                    teacher_noise_pred = teacher_dit(
                        noisy_latents,
                        timestep=timesteps_t,
                        pose_embedding=pose_embedding,
                    )
                    # 1-step DDIM solver from t -> t-k
                    x_prev = solver.ddim_step(
                        model_output=teacher_noise_pred,
                        timestep=timesteps_t,
                        prev_timestep=timesteps_prev,
                        sample=noisy_latents,
                    )

                    # 6. Target evaluation with EMA student at timestep t-k
                    # Create temporary model with EMA weights to compute target
                    ema_student = copy.deepcopy(accelerator.unwrap_model(student_dit))
                    ema_model.apply_to(ema_student)
                    ema_student.eval()

                    ema_noise_pred = ema_student(
                        x_prev,
                        timestep=timesteps_prev,
                        pose_embedding=pose_embedding,
                    )
                    target_pred_x0 = get_predicted_x0(
                        ema_noise_pred,
                        x_prev,
                        timesteps_prev,
                        solver.alphas_cumprod,
                        prediction_type="epsilon",
                    )

                # 7. Consistency Distillation Loss (Pseudo-Huber)
                cd_loss = cd_loss_fn(student_pred_x0, target_pred_x0)

                # 8. Video Temporal Consistency Loss
                temp_loss = temporal_loss_fn(student_pred_x0, target_pred_x0)

                total_loss = cd_loss + temporal_weight * temp_loss

                # Backward and optimization
                accelerator.backward(total_loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(student_dit.parameters(), config.training.max_grad_norm)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                # Update EMA weights
                unwrapped_student = accelerator.unwrap_model(student_dit)
                ema_model.update(unwrapped_student)

            # Logging
            if global_step % 10 == 0 and accelerator.is_main_process:
                lr = optimizer.param_groups[0]["lr"]
                logger.info(
                    f"[Distill Step {global_step}/{max_steps}] "
                    f"Total: {total_loss.item():.4f} | "
                    f"CD (Huber): {cd_loss.item():.4f} | "
                    f"Temporal: {temp_loss.item():.4f} | "
                    f"LR: {lr:.2e}"
                )
                accelerator.log(
                    {
                        "distill_total_loss": total_loss.item(),
                        "consistency_loss": cd_loss.item(),
                        "temporal_loss": temp_loss.item(),
                        "lr": lr,
                    },
                    step=global_step,
                )

            # Checkpoint saving
            if (global_step + 1) % save_every == 0 and accelerator.is_main_process:
                save_dir = os.path.join(args.output_dir, f"distill-step-{global_step + 1}")
                os.makedirs(save_dir, exist_ok=True)

                unwrapped_student = accelerator.unwrap_model(student_dit)
                # Save student LoRA weights
                unwrapped_student.save_pretrained(os.path.join(save_dir, "lora"))

                # Save EMA LoRA weights
                ema_save_dir = os.path.join(save_dir, "ema_lora")
                os.makedirs(ema_save_dir, exist_ok=True)
                ema_model_copy = copy.deepcopy(unwrapped_student)
                ema_model.apply_to(ema_model_copy)
                ema_model_copy.save_pretrained(ema_save_dir)

                # Copy pose encoder for self-contained checkpoint
                torch.save(pose_encoder.state_dict(), os.path.join(save_dir, "pose_encoder.pt"))
                logger.info(f"💾 Distillation checkpoint saved: {save_dir}")

            global_step += 1

    # Save final distilled weights
    if accelerator.is_main_process:
        final_dir = os.path.join(args.output_dir, "final")
        os.makedirs(final_dir, exist_ok=True)

        unwrapped_student = accelerator.unwrap_model(student_dit)
        # Apply EMA to final model as EMA provides smoother results
        ema_model.apply_to(unwrapped_student)
        unwrapped_student.save_pretrained(os.path.join(final_dir, "distilled_lora"))

        # Save pose encoder
        torch.save(pose_encoder.state_dict(), os.path.join(final_dir, "pose_encoder.pt"))

        # Save metadata info
        info = {
            "distilled_steps": target_steps,
            "teacher_checkpoint": teacher_ckpt,
            "base_model": config.model.base_model,
            "max_steps": max_steps,
        }
        with open(os.path.join(final_dir, "distill_info.json"), "w") as f:
            import json
            json.dump(info, f, indent=2)

        logger.info(f"🎉 Consistency Distillation complete! Final {target_steps}-step model saved to {final_dir}")

    accelerator.end_training()


if __name__ == "__main__":
    main()

