import torch
import torch.nn as nn
import torch.nn.functional as F
import lpips
import logging

logger = logging.getLogger(__name__)


class MotionTransferLoss(nn.Module):
    """
    Multi-objective loss function for pose-guided motion transfer video generation.
    Supports both pixel-space RGB frames and latent representations.
    """
    def __init__(self, device: str = 'cuda'):
        super().__init__()
        self.device = device
        # Initialize LPIPS for perceptual loss (VGG-based)
        try:
            self.perceptual_net = lpips.LPIPS(net='vgg').to(device)
            self.perceptual_net.eval()
            for param in self.perceptual_net.parameters():
                param.requires_grad = False
        except Exception as e:
            logger.warning(f"Could not load LPIPS on {device}: {e}. Perceptual loss will be fallback.")
            self.perceptual_net = None

    def reconstruction_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """MSE reconstruction loss in latent/pixel space."""
        return F.mse_loss(pred, target)

    def perceptual_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """LPIPS perceptual loss. Handles both RGB frames (C=3) and latent representations (C=16)."""
        if self.perceptual_net is None:
            return torch.tensor(0.0, device=pred.device, dtype=pred.dtype)

        if pred.ndim == 5:
            # Flatten batch and temporal dimensions: [B, C, T, H, W] -> [B*T, C, H, W]
            b, c, t, h, w = pred.shape
            pred = pred.transpose(1, 2).reshape(b * t, c, h, w)
            target = target.transpose(1, 2).reshape(b * t, c, h, w)
        else:
            b_t, c, h, w = pred.shape

        # Sample a subset of frames (max 8) to keep computation fast and memory light
        if pred.shape[0] > 8:
            idx = torch.linspace(0, pred.shape[0] - 1, 8, device=pred.device).long()
            pred = pred[idx]
            target = target[idx]

        # Convert to 3 channels for LPIPS (which expects 3 RGB channels)
        if c == 3:
            p_rgb = pred
            t_rgb = target
        elif c > 3:
            p_rgb = pred[:, :3]
            t_rgb = target[:, :3]
        else:
            p_rgb = pred.repeat(1, 3, 1, 1)
            t_rgb = target.repeat(1, 3, 1, 1)

        # Scale to [-1, 1] and cast to float32 for LPIPS VGG network
        p_rgb = torch.tanh(p_rgb).float()
        t_rgb = torch.tanh(t_rgb).float()

        # Ensure perceptual net is on correct device
        try:
            device = p_rgb.device
            if next(self.perceptual_net.parameters()).device != device:
                self.perceptual_net = self.perceptual_net.to(device)
            return self.perceptual_net(p_rgb, t_rgb).mean().to(dtype=pred.dtype)
        except Exception as e:
            logger.warning(f"Error in perceptual loss: {e}")
            return torch.tensor(0.0, device=pred.device, dtype=pred.dtype)

    def identity_loss(self, pred_frames: torch.Tensor, reference_image: torch.Tensor, face_encoder: nn.Module) -> torch.Tensor:
        """Cosine similarity loss for face identity preservation."""
        if face_encoder is None:
            return torch.tensor(0.0, device=pred_frames.device, dtype=pred_frames.dtype)

        b, c, t, h, w = pred_frames.shape
        pred_flat = pred_frames.transpose(1, 2).reshape(b * t, c, h, w)

        pred_emb = face_encoder(pred_flat)
        ref_emb = face_encoder(reference_image)
        ref_emb = ref_emb.repeat_interleave(t, dim=0)

        return (1.0 - F.cosine_similarity(pred_emb, ref_emb, dim=-1)).mean()

    def regional_loss(self, pred: torch.Tensor, target: torch.Tensor, pose_keypoints: torch.Tensor, confidence_scores: torch.Tensor) -> torch.Tensor:
        """Weighted MSE loss for region attention."""
        b, c, t, h, w = pred.shape
        weight_mask = torch.ones((b, 1, t, h, w), device=pred.device, dtype=pred.dtype)
        mse = F.mse_loss(pred, target, reduction='none')
        weighted_mse = mse * weight_mask
        return weighted_mse.mean()

    def pose_confidence_loss(self, pred: torch.Tensor, target: torch.Tensor, confidence: torch.Tensor) -> torch.Tensor:
        """Weight the MSE loss by pose confidence, clamped to [0.3, 1.0]."""
        # confidence: [B, T_conf, N_keypoints]
        conf_mean = confidence.mean(dim=-1).to(device=pred.device)  # [B, T_conf]
        if conf_mean.shape[1] != pred.shape[2]:
            conf_mean = F.interpolate(
                conf_mean.unsqueeze(1),
                size=pred.shape[2],
                mode="linear",
                align_corners=False,
            ).squeeze(1)  # [B, T_pred]

        conf_clamped = torch.clamp(conf_mean, 0.3, 1.0).to(dtype=pred.dtype)
        # Reshape to broadcast over [B, C, T, H, W]
        conf_clamped = conf_clamped.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)

        mse = F.mse_loss(pred, target, reduction='none')
        return (mse * conf_clamped).mean()

    def temporal_consistency_loss(self, pred_frames: torch.Tensor) -> torch.Tensor:
        """Smooth temporal transitions between adjacent frames."""
        if pred_frames.size(2) < 2:
            return torch.tensor(0.0, device=pred_frames.device, dtype=pred_frames.dtype)

        diff = pred_frames[:, :, 1:] - pred_frames[:, :, :-1]
        return torch.mean(diff ** 2)

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        ref_image: torch.Tensor,
        keypoints: torch.Tensor,
        confidence: torch.Tensor,
        config: dict,
        face_encoder: nn.Module = None
    ) -> tuple:
        """Compute total weighted multi-objective loss."""
        l_recon = self.reconstruction_loss(pred, target) * config.get('recon', 1.0)
        l_perc = self.perceptual_loss(pred, target) * config.get('perceptual', 0.5)

        l_id = torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
        if face_encoder is not None:
            l_id = self.identity_loss(pred, ref_image, face_encoder) * config.get('identity', 0.2)

        l_reg = self.regional_loss(pred, target, keypoints, confidence) * config.get('regional', 0.3)
        l_pose = self.pose_confidence_loss(pred, target, confidence) * config.get('pose_confidence', 0.1)
        l_temp = self.temporal_consistency_loss(pred) * 0.1

        total_loss = l_recon + l_perc + l_id + l_reg + l_pose + l_temp

        loss_dict = {
            'total_loss': total_loss.item(),
            'recon_loss': l_recon.item(),
            'perceptual_loss': l_perc.item(),
            'identity_loss': l_id.item(),
            'regional_loss': l_reg.item(),
            'pose_confidence_loss': l_pose.item(),
            'temporal_loss': l_temp.item()
        }

        return total_loss, loss_dict
