"""Unit tests for Consistency Distillation modules."""

import unittest
import torch
import torch.nn as nn


from src.generation.distillation import (
    DDIMSolver,
    ConsistencyDistillationLoss,
    TemporalConsistencyLoss,
    EMAModel,
    get_predicted_x0,
    build_skip_schedule,
)


def test_build_skip_schedule():
    """Test generating few-step skip schedule."""
    schedule_8 = build_skip_schedule(1000, 8)
    assert len(schedule_8) == 8
    # Should be in descending order from near 1000 down to 0
    assert schedule_8[0] > schedule_8[-1]
    assert schedule_8[-1] == 0

    schedule_4 = build_skip_schedule(1000, 4)
    assert len(schedule_4) == 4
    assert schedule_4[0] > schedule_4[-1]


def test_ddim_solver():
    """Test DDIMSolver forward noise adding and step calculation."""
    solver = DDIMSolver(num_train_timesteps=1000)
    x = torch.randn(2, 4, 8, 16, 16)
    noise = torch.randn_like(x)
    timestep = torch.tensor([500, 200])

    noisy_x = solver.add_noise(x, noise, timestep)
    assert noisy_x.shape == x.shape

    prev_timestep = torch.tensor([480, 180])
    denoised_step = solver.ddim_step(noise, timestep, prev_timestep, noisy_x)
    assert denoised_step.shape == x.shape


def test_consistency_distillation_loss():
    """Test Pseudo-Huber consistency distillation loss."""
    loss_fn = ConsistencyDistillationLoss(num_train_timesteps=1000, num_ddim_steps=25, skipping_steps=2)
    student_pred = torch.randn(2, 4, 8, 16, 16)
    target_pred = torch.randn(2, 4, 8, 16, 16)

    loss = loss_fn(student_pred, target_pred)
    assert loss.ndim == 0
    assert loss.item() >= 0.0

    # Sample timestep pairs
    t1, t2 = loss_fn.sample_timestep_pairs(batch_size=4, device=torch.device("cpu"))
    assert len(t1) == 4
    assert len(t2) == 4
    # t1 should be >= t2 because t2 is stepped backward
    assert (t1 >= t2).all()


def test_temporal_consistency_loss():
    """Test temporal consistency loss on video tensors."""
    loss_fn = TemporalConsistencyLoss()
    student = torch.randn(2, 3, 5, 16, 16)
    teacher = torch.randn(2, 3, 5, 16, 16)

    loss = loss_fn(student, teacher)
    assert loss.ndim == 0
    assert loss.item() >= 0.0


def test_ema_model():
    """Test EMA weight tracking."""
    linear = nn.Linear(10, 10)
    ema = EMAModel(linear, decay=0.9, update_after_step=0, update_every=1)

    initial_val = linear.weight.data.clone()
    # Mutate model weights
    linear.weight.data.add_(1.0)
    ema.update(linear)

    # EMA shadow params should be between initial and updated
    assert not torch.allclose(ema.shadow_params[0], linear.weight.data)

    target_linear = nn.Linear(10, 10)
    ema.apply_to(target_linear)
    assert torch.allclose(target_linear.weight.data, ema.shadow_params[0])


def test_get_predicted_x0():
    """Test converting noise prediction to clean x_0 estimate."""
    solver = DDIMSolver(num_train_timesteps=1000)
    sample = torch.randn(2, 4, 8, 16, 16)
    noise_pred = torch.randn_like(sample)
    timesteps = torch.tensor([500, 300])

    pred_x0 = get_predicted_x0(
        noise_pred,
        sample,
        timesteps,
        solver.alphas_cumprod,
        prediction_type="epsilon",
    )
    assert pred_x0.shape == sample.shape
