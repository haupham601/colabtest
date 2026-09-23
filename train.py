#!/usr/bin/env python3
"""Training script for AI Motion Transfer.

Trains LoRA adapters + 3D Conv Pose Encoder on Wan2.1-14B-I2V
for pose-guided human video generation.

Usage:
    # Train from scratch
    python train.py --config config/default.yaml --output_dir ./checkpoints

    # Resume training
    python train.py --config config/default.yaml --resume ./checkpoints/step-5000

    # On Colab with accelerate
    accelerate launch train.py --config config/default.yaml
"""

import os
import sys
import json
import math
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
logger = logging.getLogger("train")

from src.generation.pose_encoder import PoseEncoder3D
from src.generation.losses import MotionTransferLoss
from src.generation.dataset import MotionTransferDataset


def load_wan_model(config: OmegaConf, device: torch.device):
    """Load Wan2.1-14B-I2V model components.

    Attempts to load via DiffSynth-Studio first, then falls back
    to diffusers, and finally to a lightweight placeholder for
    development and test runs.

    Returns:
        Tuple of (dit_model, vae, text_encoder, noise_scheduler)
    """
    model_id = config.model.base_model
    dtype = torch.bfloat16 if config.model.dtype == "bf16" else torch.float16

    # --- Strategy 1: DiffSynth-Studio (official UniAnimate-DiT path) ---
    try:
        from diffsynth import ModelManager, WanVideoPipeline
        logger.info(f"Attempting to load Wan2.1 via DiffSynth-Studio: {model_id}")

        model_manager = ModelManager(
            torch_dtype=dtype,
            device=device,
        )
        model_manager.load_models([model_id])

        pipeline = WanVideoPipeline.from_model_manager(model_manager)
        dit = pipeline.dit
        vae = pipeline.vae
        text_encoder = pipeline.text_encoder
        noise_scheduler = pipeline.scheduler

        logger.info("✅ Wan2.1 loaded via DiffSynth-Studio successfully.")
        return dit, vae, text_encoder, noise_scheduler

    except (ImportError, Exception) as e:
        logger.warning(f"DiffSynth-Studio loading skipped/unavailable: {e}")

    # --- Strategy 2: HuggingFace diffusers ---
    try:
        from diffusers import AutoencoderKLWan, WanTransformer3DModel, FlowMatchEulerDiscreteScheduler
        from transformers import UMT5EncoderModel

        logger.info(f"Attempting to load Wan2.1 via diffusers: {model_id}")

        vae = AutoencoderKLWan.from_pretrained(
            model_id, subfolder="vae", torch_dtype=dtype
        ).to(device)

        dit = WanTransformer3DModel.from_pretrained(
            model_id, subfolder="transformer", torch_dtype=dtype
        ).to(device)

        text_encoder = UMT5EncoderModel.from_pretrained(
            model_id, subfolder="text_encoder", torch_dtype=dtype
        ).to(device)

        noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            model_id, subfolder="scheduler"
        )

        logger.info("✅ Wan2.1 loaded via diffusers successfully.")
        return dit, vae, text_encoder, noise_scheduler

    except (ImportError, Exception) as e:
        logger.warning(f"diffusers loading skipped/unavailable: {e}")

    # --- Strategy 3: Lightweight placeholder for dev/testing ---
    logger.warning(
        "⚠️ Using PLACEHOLDER models for development/testing. "
        "Install DiffSynth-Studio or diffusers with Wan2.1 weights for real training."
    )

    class PlaceholderVAE(nn.Module):
        """Placeholder VAE that mimics encode/decode shapes."""
        def __init__(self):
            super().__init__()
            self.encoder = nn.Conv3d(3, 16, 1)

        def encode(self, x: torch.Tensor):
            class Dist:
                def __init__(self, tensor):
                    b, c, t, h, w = tensor.shape
                    self.latent = torch.randn(
                        b, 16, t, max(1, h // 8), max(1, w // 8),
                        device=tensor.device, dtype=tensor.dtype
                    )
                def sample(self):
                    return self.latent
            return Dist(x)

        def decode(self, z: torch.Tensor):
            b, c, t, h, w = z.shape
            return torch.randn(b, 3, t, h * 8, w * 8, device=z.device, dtype=z.dtype)

    class PlaceholderDiT(nn.Module):
        """Placeholder DiT with matching attention target modules for LoRA."""
        def __init__(self):
            super().__init__()
            hidden = 1280
            self.to_q = nn.Linear(hidden, hidden)
            self.to_k = nn.Linear(hidden, hidden)
            self.to_v = nn.Linear(hidden, hidden)
            self.to_out = nn.Linear(hidden, hidden)
            self.proj_in = nn.Linear(16, hidden)
            self.proj_out = nn.Linear(hidden, 16)

        def forward(self, x, timestep, encoder_hidden_states=None, pose_embedding=None):
            b, c, t, h, w = x.shape
            x_flat = x.permute(0, 2, 3, 4, 1).reshape(b, t * h * w, c)
            h_state = self.proj_in(x_flat)

            if pose_embedding is not None:
                if pose_embedding.ndim == 4:
                    bp, tp, np_, cp = pose_embedding.shape
                    pe = pose_embedding.reshape(bp, tp * np_, cp)
                else:
                    pe = pose_embedding

                seq_len = h_state.shape[1]
                if pe.shape[1] >= seq_len:
                    pe = pe[:, :seq_len, :]
                else:
                    pe = F.pad(pe, (0, 0, 0, seq_len - pe.shape[1]))
                h_state = h_state + pe

            q = self.to_q(h_state)
            k = self.to_k(h_state)
            v = self.to_v(h_state)
            attn = torch.bmm(
                F.softmax(torch.bmm(q, k.transpose(-1, -2)) / math.sqrt(q.shape[-1]), dim=-1),
                v
            )
            out = self.to_out(attn)
            out = self.proj_out(out)
            return out.reshape(b, t, h, w, c).permute(0, 4, 1, 2, 3)

    class PlaceholderScheduler:
        def __init__(self):
            self.num_train_timesteps = 1000

        def add_noise(self, original: torch.Tensor, noise: torch.Tensor, timesteps: torch.Tensor):
            alpha = (1.0 - timesteps.float() / self.num_train_timesteps).view(-1, 1, 1, 1, 1)
            return alpha * original + (1.0 - alpha) * noise

    return (
        PlaceholderDiT().to(device),
        PlaceholderVAE().to(device),
        None,
        PlaceholderScheduler(),
    )


def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps):
    """Cosine learning rate schedule with warmup."""
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def validate(
    dit: nn.Module,
    vae: nn.Module,
    pose_encoder: PoseEncoder3D,
    val_loader: DataLoader,
    noise_scheduler,
    device: torch.device,
    global_step: int,
    accelerator: Accelerator,
) -> dict:
    """Run validation and compute metrics."""
    dit.eval()
    pose_encoder.eval()

    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in val_loader:
            ref_image = batch["reference_image"].to(device)
            video_frames = batch["video_frames"].to(device)
            pose_images = batch["pose_images"].to(device)

            b, t, c, h, w = video_frames.shape
            video_3d = video_frames.permute(0, 2, 1, 3, 4)
            latents = vae.encode(video_3d).sample()

            pose_emb = pose_encoder(pose_images)

            noise = torch.randn_like(latents)
            timesteps = torch.randint(0, 1000, (b,), device=device).long()
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            pred = dit(noisy_latents, timestep=timesteps, pose_embedding=pose_emb)
            loss = F.mse_loss(pred, noise)

            total_loss += loss.item()
            num_batches += 1

            if num_batches >= 10:
                break

    avg_loss = total_loss / max(num_batches, 1)
    dit.train()
    pose_encoder.train()

    logger.info(f"[Validation] Step {global_step} | Val Loss: {avg_loss:.6f}")
    return {"val_loss": avg_loss}


def main():
    parser = argparse.ArgumentParser(description="Train AI Motion Transfer")
    parser.add_argument("--config", type=str, default="config/default.yaml")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="./checkpoints")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Load config
    config = OmegaConf.load(args.config)
    os.makedirs(args.output_dir, exist_ok=True)
    OmegaConf.save(config, os.path.join(args.output_dir, "config.yaml"))

    # Initialize accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps=config.training.grad_accum,
        mixed_precision=config.training.mixed_precision,
        log_with="wandb",
    )
    set_seed(args.seed)
    device = accelerator.device

    if accelerator.is_main_process:
        accelerator.init_trackers(
            project_name="AI-Motion-Transfer",
            config=OmegaConf.to_container(config, resolve=True),
        )

    # 1. Load base models
    logger.info("Loading models...")
    dit, vae, text_encoder, noise_scheduler = load_wan_model(config, device)

    vae.requires_grad_(False)
    if text_encoder is not None:
        text_encoder.requires_grad_(False)
    dit.requires_grad_(False)

    # 2. Add LoRA
    logger.info(f"Adding LoRA (rank={config.lora.rank}, alpha={config.lora.alpha})")
    lora_config = LoraConfig(
        r=config.lora.rank,
        lora_alpha=config.lora.alpha,
        target_modules=list(config.lora.target_modules),
        lora_dropout=0.05,
        bias="none",
    )
    dit = get_peft_model(dit, lora_config)
    dit.print_trainable_parameters()

    if config.training.gradient_checkpointing and hasattr(dit, "enable_gradient_checkpointing"):
        dit.enable_gradient_checkpointing()
        logger.info("Gradient checkpointing enabled on DiT")

    # 3. Pose Encoder
    hidden_dim = 1280
    pose_encoder = PoseEncoder3D(
        in_channels=config.pose_encoder.in_channels,
        base_channels=config.pose_encoder.base_channels,
        num_blocks=config.pose_encoder.num_blocks,
        output_channels=hidden_dim,
    ).to(device)

    # 4. Multi-objective loss
    loss_fn = MotionTransferLoss(device=str(device))

    # 5. Optimizer & Scheduler
    trainable_params = [
        {"params": [p for p in dit.parameters() if p.requires_grad], "lr": config.training.lr},
        {"params": pose_encoder.parameters(), "lr": config.training.lr * 2},
    ]
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=config.training.lr,
        betas=(0.9, 0.999),
        weight_decay=0.01,
        eps=1e-8,
    )

    num_warmup_steps = int(config.training.max_steps * config.training.warmup_ratio)
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps, config.training.max_steps
    )

    # 6. Dataset
    logger.info(f"Loading dataset from {config.data.dataset_path}")
    dataset = MotionTransferDataset(
        data_dir=config.data.dataset_path,
        resolution=tuple(config.data.resolution),
        num_frames=config.data.num_frames,
        fps=config.data.fps,
        pose_type=config.data.pose_type,
    )

    if len(dataset) == 0:
        logger.warning(
            "Dataset is empty! Using synthetic fallback samples for initialization test. "
            "Run prepare_data.py to populate real training data."
        )
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
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        drop_last=True,
    )

    # Prepare with accelerate
    dit, pose_encoder, optimizer, dataloader, lr_scheduler = accelerator.prepare(
        dit, pose_encoder, optimizer, dataloader, lr_scheduler
    )

    global_step = 0
    if args.resume:
        accelerator.load_state(args.resume)
        global_step = int(Path(args.resume).name.split("-")[-1])
        logger.info(f"Resumed from {args.resume} at step {global_step}")

    logger.info(f"Starting training loop up to {config.training.max_steps} steps...")
    dit.train()
    pose_encoder.train()

    while global_step < config.training.max_steps:
        for batch in dataloader:
            if global_step >= config.training.max_steps:
                break

            with accelerator.accumulate(dit, pose_encoder):
                ref_image = batch["reference_image"]
                video_frames = batch["video_frames"]
                pose_images = batch["pose_images"]
                keypoints = batch["pose_keypoints"]
                confidence = batch["confidence_scores"]

                b, t, c, h, w = video_frames.shape

                with torch.no_grad():
                    video_3d = video_frames.permute(0, 2, 1, 3, 4)
                    latents = vae.encode(video_3d).sample()

                pose_embedding = pose_encoder(pose_images)

                noise = torch.randn_like(latents)
                timesteps = torch.randint(
                    0, noise_scheduler.num_train_timesteps, (b,), device=device
                ).long()

                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                noise_pred = dit(
                    noisy_latents,
                    timestep=timesteps,
                    pose_embedding=pose_embedding,
                )

                loss_config = {
                    "recon": config.loss.recon_weight,
                    "perceptual": config.loss.perceptual_weight,
                    "identity": config.loss.identity_weight,
                    "regional": config.loss.regional_weight,
                    "pose_confidence": config.loss.pose_confidence_weight,
                }
                total_loss, loss_dict = loss_fn(
                    pred=noise_pred,
                    target=noise,
                    ref_image=ref_image,
                    keypoints=keypoints,
                    confidence=confidence,
                    config=loss_config,
                )

                accelerator.backward(total_loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(
                        list(dit.parameters()) + list(pose_encoder.parameters()),
                        config.training.max_grad_norm,
                    )

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if global_step % 10 == 0 and accelerator.is_main_process:
                lr = optimizer.param_groups[0]["lr"]
                logger.info(
                    f"Step {global_step}/{config.training.max_steps} | "
                    f"Loss: {total_loss.item():.4f} | "
                    f"Recon: {loss_dict['recon_loss']:.4f} | "
                    f"LR: {lr:.2e}"
                )
                accelerator.log(
                    {**loss_dict, "learning_rate": lr},
                    step=global_step,
                )

            if (global_step + 1) % config.training.save_every == 0:
                if accelerator.is_main_process:
                    save_dir = os.path.join(args.output_dir, f"step-{global_step + 1}")
                    accelerator.save_state(save_dir)

                    lora_dir = os.path.join(save_dir, "lora")
                    os.makedirs(lora_dir, exist_ok=True)
                    unwrapped_dit = accelerator.unwrap_model(dit)
                    unwrapped_dit.save_pretrained(lora_dir)

                    pe_path = os.path.join(save_dir, "pose_encoder.pt")
                    unwrapped_pe = accelerator.unwrap_model(pose_encoder)
                    torch.save(unwrapped_pe.state_dict(), pe_path)

                    logger.info(f"💾 Checkpoint saved: {save_dir}")

            if (global_step + 1) % config.training.val_every == 0:
                if accelerator.is_main_process:
                    val_metrics = validate(
                        dit, vae, pose_encoder, dataloader,
                        noise_scheduler, device, global_step, accelerator,
                    )
                    accelerator.log(val_metrics, step=global_step)

            global_step += 1

    if accelerator.is_main_process:
        final_dir = os.path.join(args.output_dir, "final")
        accelerator.save_state(final_dir)

        unwrapped_dit = accelerator.unwrap_model(dit)
        unwrapped_dit.save_pretrained(os.path.join(final_dir, "lora"))

        unwrapped_pe = accelerator.unwrap_model(pose_encoder)
        torch.save(unwrapped_pe.state_dict(), os.path.join(final_dir, "pose_encoder.pt"))

        logger.info(f"✅ Training complete! Final model saved to {final_dir}")

    accelerator.end_training()


if __name__ == "__main__":
    main()
