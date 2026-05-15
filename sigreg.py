import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseSigRegLoss(nn.Module):
    """Common paired alignment wrapper for SIGReg variants."""

    variant_name = "base"

    def __init__(self, sim_weight: float = 25.0, reg_weight: float = 25.0, eps: float = 1e-6):
        super().__init__()
        self.sim_weight = float(sim_weight)
        self.reg_weight = float(reg_weight)
        self.eps = float(eps)

    def _regularize(self, z: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        raise NotImplementedError

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        if x.ndim != 2 or y.ndim != 2:
            raise ValueError("x and y must be 2D tensors of shape [B, D]")
        if x.shape != y.shape:
            raise ValueError(f"x and y must have same shape, got {x.shape} and {y.shape}")
        if x.shape[0] < 2:
            raise ValueError("Batch size must be at least 2 to compute SIGReg statistics")

        sim_loss = F.mse_loss(x, y)
        reg_x_loss, x_aux = self._regularize(x)
        reg_y_loss, y_aux = self._regularize(y)
        reg_loss = 0.5 * (reg_x_loss + reg_y_loss)
        loss = self.sim_weight * sim_loss + self.reg_weight * reg_loss

        with torch.no_grad():
            stats = {
                "sim_loss": sim_loss.detach(),
                "reg_loss": reg_loss.detach(),
                "reg_x_loss": reg_x_loss.detach(),
                "reg_y_loss": reg_y_loss.detach(),
                "loss": loss.detach(),
                "sigreg_variant": self.variant_name,
            }
            stats.update(self._common_stats("x", x, x_aux))
            stats.update(self._common_stats("y", y, y_aux))

        return loss, stats

    def _common_stats(self, prefix: str, raw: torch.Tensor, aux: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        projected = aux["projected"]
        centered = aux["centered"]
        cov = aux["cov"]
        off_diag = cov - torch.diag(torch.diagonal(cov))
        return {
            f"std_{prefix}_mean": centered.std(dim=0, unbiased=True).mean().detach(),
            f"std_{prefix}_min": centered.std(dim=0, unbiased=True).min().detach(),
            f"{prefix}_mean_norm": raw.mean(dim=0).norm().detach(),
            f"proj_{prefix}_mean_norm": projected.mean(dim=0).norm().detach(),
            f"cov_{prefix}_diag_mean": torch.diagonal(cov).mean().detach(),
            f"cov_{prefix}_offdiag_abs_mean": off_diag.abs().mean().detach(),
        }


class WeakSigRegLoss(BaseSigRegLoss):
    """Weak-SIGReg: match the projected covariance to the identity."""

    variant_name = "weak"

    def __init__(
        self,
        sim_weight: float = 25.0,
        reg_weight: float = 25.0,
        sketch_dim: int = 64,
        eps: float = 1e-6,
    ):
        super().__init__(sim_weight=sim_weight, reg_weight=reg_weight, eps=eps)
        self.sketch_dim = int(sketch_dim)

    def _project(self, z: torch.Tensor) -> torch.Tensor:
        _, dim = z.shape
        if dim <= self.sketch_dim:
            return z
        sketch = torch.randn(self.sketch_dim, dim, device=z.device, dtype=z.dtype) / math.sqrt(dim)
        return z @ sketch.T

    def _regularize(self, z: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        projected = self._project(z)
        centered = projected - projected.mean(dim=0, keepdim=True)
        cov = (centered.T @ centered) / (centered.shape[0] - 1 + self.eps)
        target = torch.eye(cov.shape[0], device=z.device, dtype=z.dtype)
        penalty = torch.norm(cov - target, p="fro")
        return penalty, {"projected": projected, "centered": centered, "cov": cov}


class StrongSigRegLoss(BaseSigRegLoss):
    """Strong-SIGReg following LeJEPA's sliced Epps-Pulley formulation."""

    variant_name = "strong"

    def __init__(
        self,
        sim_weight: float = 25.0,
        reg_weight: float = 25.0,
        num_slices: int = 1024,
        epps_pulley_t_max: float = 3.0,
        epps_pulley_num_points: int = 17,
        eps: float = 1e-6,
    ):
        super().__init__(sim_weight=sim_weight, reg_weight=reg_weight, eps=eps)
        self.num_slices = int(num_slices)
        self.epps_pulley_t_max = float(epps_pulley_t_max)
        self.epps_pulley_num_points = int(epps_pulley_num_points)
        if self.epps_pulley_num_points < 3 or self.epps_pulley_num_points % 2 == 0:
            raise ValueError("epps_pulley_num_points must be an odd integer >= 3")

        t = torch.linspace(0.0, self.epps_pulley_t_max, self.epps_pulley_num_points, dtype=torch.float32)
        dt = self.epps_pulley_t_max / (self.epps_pulley_num_points - 1)
        weights = torch.full((self.epps_pulley_num_points,), 2.0 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        phi = torch.exp(-0.5 * t.square())
        self.register_buffer("t", t)
        self.register_buffer("phi", phi)
        self.register_buffer("weights", weights * phi)

    def _project(self, z: torch.Tensor) -> torch.Tensor:
        _, dim = z.shape
        directions = torch.randn(dim, self.num_slices, device=z.device, dtype=z.dtype)
        directions = directions / directions.norm(p=2, dim=0, keepdim=True).clamp_min(self.eps)
        return z @ directions

    def _regularize(self, z: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        projected = self._project(z)
        centered = projected - projected.mean(dim=0, keepdim=True)
        cov = (centered.T @ centered) / (centered.shape[0] - 1 + self.eps)

        t = self.t.to(device=z.device, dtype=z.dtype)
        phi = self.phi.to(device=z.device, dtype=z.dtype)
        weights = self.weights.to(device=z.device, dtype=z.dtype)

        x_t = centered.unsqueeze(-1) * t.view(1, 1, -1)
        cos_mean = torch.cos(x_t).mean(dim=0)
        sin_mean = torch.sin(x_t).mean(dim=0)
        err = (cos_mean - phi.unsqueeze(0)).square() + sin_mean.square()
        penalty = ((err @ weights) * centered.shape[0]).mean()

        return penalty, {"projected": projected, "centered": centered, "cov": cov}


def build_sigreg_loss(config: dict) -> BaseSigRegLoss:
    variant = str(config.get("variant", "weak")).lower()
    common_kwargs = {
        "sim_weight": float(config.get("sim_weight", 25.0)),
        "reg_weight": float(config.get("reg_weight", 25.0)),
        "eps": float(config.get("eps", 1e-6)),
    }
    if variant == "weak":
        return WeakSigRegLoss(
            sketch_dim=int(config.get("sketch_dim", 64)),
            **common_kwargs,
        )
    if variant == "strong":
        return StrongSigRegLoss(
            num_slices=int(config.get("num_slices", 1024)),
            epps_pulley_t_max=float(config.get("epps_pulley_t_max", 3.0)),
            epps_pulley_num_points=int(config.get("epps_pulley_num_points", 17)),
            **common_kwargs,
        )
    raise ValueError(f"Unknown SIGReg variant: {variant}")


SigRegLoss = WeakSigRegLoss
