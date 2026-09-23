"""Consistency Distillation utilities for reducing inference steps.

Implements Latent Consistency Distillation (LCD) adapted for DiT video models.
Distills a 25-step teacher into a 4-8 step student while preserving quality.

Key concepts:
- Teacher: fully trained model (25 steps) - FROZEN
- Student: LoRA copy that learns to produce same output in fewer steps - TRAINABLE
- EMA: exponential moving average of student weights for stability
- The student learns to map any point on the ODE trajectory directly to the clean output.

References:
- Latent Consistency Models (Luo et al. 2023)
- Progressive Distillation for Fast Sampling (Salimans & Ho 2022)
- Dual-Expert Consistency Model for Video (DCM, 2025)
"""

import math
import copy
import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

logger = logging.getLogger(__name__)


class DDIMSolver:
    """DDIM ODE solver for computing teacher trajectories.

    Given a noisy sample x_t and the teacher's noise prediction,
    computes the denoised estimate x_{t-1} using DDIM update rule.
    """

    def __init__(
        self,
        num_train_timesteps: int = 1000,
        beta_start: float = 0.00085,
        beta_end: float = 0.012,
        beta_schedule: str = "scaled_linear",
    ):
        if beta_schedule == "scaled_linear":
            betas = torch.linspace(beta_start**0.5, beta_end**0.5, num_train_timesteps) ** 2
        elif beta_schedule == "linear":
            betas = torch.linspace(beta_start, beta_end, num_train_timesteps)
        else:
            raise ValueError(f"Unknown beta schedule: {beta_schedule}")

        self.alphas = 1.0 - betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.num_train_timesteps = num_train_timesteps

        # Pre-compute useful quantities
        self.sqrt_alphas_cumprod = self.alphas_cumprod.sqrt()
        self.sqrt_one_minus_alphas_cumprod = (1.0 - self.alphas_cumprod).sqrt()

    def to(self, device: torch.device) -> "DDIMSolver":
        """Move all tensors to device."""
        self.alphas_cumprod = self.alphas_cumprod.to(device)
        self.sqrt_alphas_cumprod = self.sqrt_alphas_cumprod.to(device)
        self.sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod.to(device)
        return self

    def ddim_step(
        self,
        model_output: torch.Tensor,
        timestep: torch.Tensor,
        prev_timestep: torch.Tensor,
        sample: torch.Tensor,
    ) -> torch.Tensor:
        """Perform one DDIM denoising step.

        Args:
            model_output: Predicted noise from the model.
            timestep: Current timestep t.
            prev_timestep: Target timestep t-1.
            sample: Current noisy sample x_t.

        Returns:
            Denoised sample x_{t-1}.
        """
        # Get alpha values for current and previous timesteps
        # Handle batch of timesteps
        alpha_prod_t = self._get_alpha(timestep)
        alpha_prod_t_prev = self._get_alpha(prev_timestep)

        # Reshape for broadcasting with [B, C, T, H, W]
        while alpha_prod_t.ndim < sample.ndim:
            alpha_prod_t = alpha_prod_t.unsqueeze(-1)
            alpha_prod_t_prev = alpha_prod_t_prev.unsqueeze(-1)

        # Compute predicted x_0
        pred_x0 = (sample - (1 - alpha_prod_t).sqrt() * model_output) / alpha_prod_t.sqrt()

        # Compute x_{t-1} using DDIM update (deterministic, eta=0)
        dir_xt = (1.0 - alpha_prod_t_prev).sqrt() * model_output
        prev_sample = alpha_prod_t_prev.sqrt() * pred_x0 + dir_xt

        return prev_sample

    def _get_alpha(self, timestep: torch.Tensor) -> torch.Tensor:
        """Get cumulative alpha for given timesteps."""
        timestep = timestep.clamp(0, self.num_train_timesteps - 1).long()
        return self.alphas_cumprod[timestep]

    def add_noise(
        self,
        original: torch.Tensor,
        noise: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        """Add noise at given timestep (forward diffusion)."""
        sqrt_alpha = self.sqrt_alphas_cumprod[timestep.long()]
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[timestep.long()]

        while sqrt_alpha.ndim < original.ndim:
            sqrt_alpha = sqrt_alpha.unsqueeze(-1)
            sqrt_one_minus_alpha = sqrt_one_minus_alpha.unsqueeze(-1)

        return sqrt_alpha * original + sqrt_one_minus_alpha * noise


class ConsistencyDistillationLoss(nn.Module):
    """Consistency Distillation loss for few-step generation.

    The core idea: force the student model to produce consistent outputs
    across different points on the same ODE trajectory.

    L = || f_student(x_t, t) - f_ema(x_{t-k}, t-k) ||^2

    Where x_{t-k} is obtained by running k steps of the teacher's DDIM solver.
    """

    def __init__(
        self,
        num_train_timesteps: int = 1000,
        num_ddim_steps: int = 50,
        skipping_steps: int = 1,
        huber_c: float = 0.001,
        use_huber: bool = True,
    ):
        """Initialize consistency distillation loss.

        Args:
            num_train_timesteps: Total noise schedule timesteps.
            num_ddim_steps: Number of DDIM steps for the teacher trajectory.
            skipping_steps: How many ODE steps to skip (k in the formula).
            huber_c: Huber loss threshold. Smaller = closer to L1.
            use_huber: Whether to use Pseudo-Huber loss instead of MSE.
        """
        super().__init__()
        self.num_train_timesteps = num_train_timesteps
        self.num_ddim_steps = num_ddim_steps
        self.skipping_steps = skipping_steps
        self.huber_c = huber_c
        self.use_huber = use_huber

        # Build DDIM timestep schedule (evenly spaced)
        step_ratio = num_train_timesteps // num_ddim_steps
        self.ddim_timesteps = (
            (torch.arange(0, num_ddim_steps) * step_ratio)
            .round()
            .long()
            .flip(0)  # descending: T, T-k, T-2k, ..., 0
        )

        self.solver = DDIMSolver(num_train_timesteps=num_train_timesteps)

    def to(self, device: torch.device) -> "ConsistencyDistillationLoss":
        """Move to device."""
        self.ddim_timesteps = self.ddim_timesteps.to(device)
        self.solver = self.solver.to(device)
        return self

    def pseudo_huber_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Pseudo-Huber loss - smoother than MSE, more robust to outliers.

        L = sqrt((pred - target)^2 + c^2) - c
        """
        diff = pred - target
        return (torch.sqrt(diff**2 + self.huber_c**2) - self.huber_c).mean()

    def forward(
        self,
        student_output: torch.Tensor,
        ema_output: torch.Tensor,
    ) -> torch.Tensor:
        """Compute consistency distillation loss.

        Args:
            student_output: Student's predicted x_0 at timestep t.
            ema_output: EMA model's predicted x_0 at timestep t-k (from teacher trajectory).

        Returns:
            Scalar loss value.
        """
        if self.use_huber:
            return self.pseudo_huber_loss(student_output, ema_output.detach())
        else:
            return F.mse_loss(student_output, ema_output.detach())

    def sample_timestep_pairs(
        self,
        batch_size: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample (t, t-k) timestep pairs for training.

        Returns:
            Tuple of (timesteps_t, timesteps_t_minus_k), each shape [B].
        """
        num_steps = len(self.ddim_timesteps)
        max_idx = num_steps - self.skipping_steps

        # Random indices into DDIM schedule
        indices = torch.randint(0, max_idx, (batch_size,), device=device)
        indices_prev = indices + self.skipping_steps

        timesteps_t = self.ddim_timesteps[indices]
        timesteps_t_prev = self.ddim_timesteps[indices_prev.clamp(max=num_steps - 1)]

        return timesteps_t, timesteps_t_prev


class TemporalConsistencyLoss(nn.Module):
    """Temporal consistency loss for video distillation.

    Ensures that the distilled model maintains smooth temporal transitions.
    Without this, naive distillation causes inter-frame flickering.

    Computes both:
    1. Frame-level difference consistency (student vs teacher diffs should match)
    2. Feature-level flow consistency (optical flow correlation)
    """

    def __init__(self, flow_weight: float = 0.5, diff_weight: float = 0.5):
        super().__init__()
        self.flow_weight = flow_weight
        self.diff_weight = diff_weight

    def forward(
        self,
        student_frames: torch.Tensor,
        teacher_frames: torch.Tensor,
    ) -> torch.Tensor:
        """Compute temporal consistency loss.

        Args:
            student_frames: Student predictions [B, C, T, H, W].
            teacher_frames: Teacher predictions [B, C, T, H, W].

        Returns:
            Scalar temporal consistency loss.
        """
        if student_frames.size(2) < 2:
            return torch.tensor(0.0, device=student_frames.device)

        # 1. Frame difference consistency
        # Student's inter-frame diffs should match teacher's
        student_diffs = student_frames[:, :, 1:] - student_frames[:, :, :-1]
        teacher_diffs = teacher_frames[:, :, 1:] - teacher_frames[:, :, :-1]
        diff_loss = F.mse_loss(student_diffs, teacher_diffs.detach())

        # 2. Feature correlation consistency
        # Flatten spatial dims and compute cosine similarity between adjacent frames
        b, c, t, h, w = student_frames.shape
        s_flat = rearrange(student_frames, "b c t h w -> (b t) (c h w)")
        t_flat = rearrange(teacher_frames, "b c t h w -> (b t) (c h w)")

        # Cosine similarity between adjacent frames
        s_sim = F.cosine_similarity(s_flat[:-b], s_flat[b:], dim=-1)
        t_sim = F.cosine_similarity(t_flat[:-b], t_flat[b:], dim=-1)
        flow_loss = F.mse_loss(s_sim, t_sim.detach())

        total = self.diff_weight * diff_loss + self.flow_weight * flow_loss
        return total


class EMAModel:
    """Exponential Moving Average of model parameters.

    Maintains a running average of model weights for stable targets
    during consistency distillation.
    """

    def __init__(
        self,
        model: nn.Module,
        decay: float = 0.9999,
        update_after_step: int = 0,
        update_every: int = 1,
    ):
        """Initialize EMA.

        Args:
            model: Source model to track.
            decay: EMA decay rate. Higher = slower updates, more stable.
            update_after_step: Start EMA updates only after this step.
            update_every: Update EMA every N steps.
        """
        self.decay = decay
        self.update_after_step = update_after_step
        self.update_every = update_every
        self.step = 0

        # Deep copy model parameters
        self.shadow_params = [p.clone().detach() for p in model.parameters()]

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        """Update EMA parameters with current model weights."""
        self.step += 1

        if self.step < self.update_after_step:
            # Just copy weights directly before warmup is done
            for ema_p, model_p in zip(self.shadow_params, model.parameters()):
                ema_p.copy_(model_p)
            return

        if self.step % self.update_every != 0:
            return

        for ema_p, model_p in zip(self.shadow_params, model.parameters()):
            ema_p.lerp_(model_p.data, 1.0 - self.decay)

    def apply_to(self, model: nn.Module) -> None:
        """Copy EMA weights into a model (for inference/target computation)."""
        for ema_p, model_p in zip(self.shadow_params, model.parameters()):
            model_p.data.copy_(ema_p)

    def state_dict(self) -> dict:
        """Serialize EMA state."""
        return {
            "shadow_params": self.shadow_params,
            "decay": self.decay,
            "step": self.step,
        }

    def load_state_dict(self, state_dict: dict) -> None:
        """Restore EMA state."""
        self.shadow_params = state_dict["shadow_params"]
        self.decay = state_dict["decay"]
        self.step = state_dict["step"]


def get_predicted_x0(
    model_output: torch.Tensor,
    sample: torch.Tensor,
    timestep: torch.Tensor,
    alphas_cumprod: torch.Tensor,
    prediction_type: str = "epsilon",
) -> torch.Tensor:
    """Convert model output to predicted clean sample x_0.

    Args:
        model_output: Raw model output (noise, velocity, or x_0 depending on type).
        sample: Noisy input sample x_t.
        timestep: Current timestep.
        alphas_cumprod: Cumulative product of alphas.
        prediction_type: One of "epsilon" (noise), "v_prediction" (velocity), "sample" (x_0).

    Returns:
        Predicted clean sample x_0.
    """
    alpha_prod = alphas_cumprod[timestep.long()]
    while alpha_prod.ndim < sample.ndim:
        alpha_prod = alpha_prod.unsqueeze(-1)

    if prediction_type == "epsilon":
        # x_0 = (x_t - sqrt(1-alpha) * eps) / sqrt(alpha)
        pred_x0 = (sample - (1 - alpha_prod).sqrt() * model_output) / alpha_prod.sqrt()
    elif prediction_type == "v_prediction":
        # x_0 = sqrt(alpha) * x_t - sqrt(1-alpha) * v
        pred_x0 = alpha_prod.sqrt() * sample - (1 - alpha_prod).sqrt() * model_output
    elif prediction_type == "sample":
        pred_x0 = model_output
    else:
        raise ValueError(f"Unknown prediction type: {prediction_type}")

    return pred_x0


def build_skip_schedule(
    num_train_timesteps: int = 1000,
    num_inference_steps: int = 8,
) -> torch.Tensor:
    """Build evenly spaced timestep schedule for few-step inference.

    Args:
        num_train_timesteps: Total training timesteps.
        num_inference_steps: Target number of inference steps.

    Returns:
        Tensor of timesteps for inference, shape [num_inference_steps].
    """
    step_ratio = num_train_timesteps / num_inference_steps
    timesteps = (torch.arange(num_inference_steps) * step_ratio).round().long()
    timesteps = timesteps.flip(0)  # Descending: T -> 0
    return timesteps

