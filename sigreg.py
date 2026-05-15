import torch
import torch.nn as nn
import torch.nn.functional as F


class SigRegLoss(nn.Module):
    """
    Paired alignment loss with Weak-SIGReg regularization.

    Input:
      x: [B, D]
      y: [B, D]

    Returns:
      loss, stats_dict
    """

    def __init__(
        self,
        sim_weight: float = 25.0,
        reg_weight: float = 25.0,
        sketch_dim: int = 64,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.sim_weight = float(sim_weight)
        self.reg_weight = float(reg_weight)
        self.sketch_dim = int(sketch_dim)
        self.eps = float(eps)

    def _weak_sigreg_term(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        n, dim = z.shape
        if dim > self.sketch_dim:
            sketch = torch.randn(self.sketch_dim, dim, device=z.device, dtype=z.dtype) / (dim ** 0.5)
            z = z @ sketch.T
            out_dim = self.sketch_dim
        else:
            out_dim = dim

        centered = z - z.mean(dim=0, keepdim=True)
        cov = (centered.T @ centered) / (n - 1 + self.eps)
        target = torch.eye(out_dim, device=z.device, dtype=z.dtype)
        penalty = torch.norm(cov - target, p="fro")
        return penalty, cov, centered.std(dim=0, unbiased=True)

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        if x.ndim != 2 or y.ndim != 2:
            raise ValueError("x and y must be 2D tensors of shape [B, D]")
        if x.shape != y.shape:
            raise ValueError(f"x and y must have same shape, got {x.shape} and {y.shape}")
        if x.shape[0] < 2:
            raise ValueError("Batch size must be at least 2 to compute SIGReg statistics")

        sim_loss = F.mse_loss(x, y)
        reg_x_loss, cov_x, std_x = self._weak_sigreg_term(x)
        reg_y_loss, cov_y, std_y = self._weak_sigreg_term(y)
        reg_loss = 0.5 * (reg_x_loss + reg_y_loss)

        loss = self.sim_weight * sim_loss + self.reg_weight * reg_loss

        with torch.no_grad():
            stats = {
                "sim_loss": sim_loss.detach(),
                "reg_loss": reg_loss.detach(),
                "reg_x_loss": reg_x_loss.detach(),
                "reg_y_loss": reg_y_loss.detach(),
                "loss": loss.detach(),
                "std_x_mean": std_x.mean().detach(),
                "std_y_mean": std_y.mean().detach(),
                "std_x_min": std_x.min().detach(),
                "std_y_min": std_y.min().detach(),
                "x_mean_norm": x.mean(dim=0).norm().detach(),
                "y_mean_norm": y.mean(dim=0).norm().detach(),
                "cov_x_diag_mean": torch.diagonal(cov_x).mean().detach(),
                "cov_y_diag_mean": torch.diagonal(cov_y).mean().detach(),
                "cov_x_offdiag_abs_mean": (cov_x - torch.diag(torch.diagonal(cov_x))).abs().mean().detach(),
                "cov_y_offdiag_abs_mean": (cov_y - torch.diag(torch.diagonal(cov_y))).abs().mean().detach(),
            }

        return loss, stats
