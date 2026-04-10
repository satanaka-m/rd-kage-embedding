import torch
import torch.nn as nn
import torch.nn.functional as F


def _off_diagonal(x: torch.Tensor) -> torch.Tensor:
    if x.ndim != 2 or x.shape[0] != x.shape[1]:
        raise ValueError(f"Expected square matrix, got {tuple(x.shape)}")
    n = x.shape[0]
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


class VicRegLoss(nn.Module):
    """
    VICReg loss with the standard three terms:
      - invariance: MSE between paired embeddings
      - variance: keep per-dimension std above a margin
      - covariance: decorrelate dimensions within each modality

    Input:
      x: [B, D]
      y: [B, D]

    Returns:
      loss, stats_dict
    """

    def __init__(
        self,
        sim_weight: float = 25.0,
        var_weight: float = 25.0,
        cov_weight: float = 1.0,
        eps: float = 1e-4,
        variance_target: float = 1.0,
    ):
        super().__init__()
        self.sim_weight = float(sim_weight)
        self.var_weight = float(var_weight)
        self.cov_weight = float(cov_weight)
        self.eps = float(eps)
        self.variance_target = float(variance_target)

    def _variance_term(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        std = torch.sqrt(z.var(dim=0, unbiased=True) + self.eps)
        penalty = F.relu(self.variance_target - std).mean()
        return penalty, std

    def _covariance_term(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = z - z.mean(dim=0)
        cov = (z.T @ z) / (z.shape[0] - 1)
        off_diag = _off_diagonal(cov)
        penalty = off_diag.pow(2).sum() / z.shape[1]
        return penalty, cov

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        if x.ndim != 2 or y.ndim != 2:
            raise ValueError("x and y must be 2D tensors of shape [B, D]")
        if x.shape != y.shape:
            raise ValueError(f"x and y must have same shape, got {x.shape} and {y.shape}")
        if x.shape[0] < 2:
            raise ValueError("Batch size must be at least 2 to compute variance/covariance")

        sim_loss = F.mse_loss(x, y)
        var_x_loss, std_x = self._variance_term(x)
        var_y_loss, std_y = self._variance_term(y)
        var_loss = 0.5 * (var_x_loss + var_y_loss)

        cov_x_loss, cov_x = self._covariance_term(x)
        cov_y_loss, cov_y = self._covariance_term(y)
        cov_loss = cov_x_loss + cov_y_loss

        loss = (
            self.sim_weight * sim_loss
            + self.var_weight * var_loss
            + self.cov_weight * cov_loss
        )

        with torch.no_grad():
            stats = {
                "sim_loss": sim_loss.detach(),
                "var_loss": var_loss.detach(),
                "cov_loss": cov_loss.detach(),
                "var_x_loss": var_x_loss.detach(),
                "var_y_loss": var_y_loss.detach(),
                "cov_x_loss": cov_x_loss.detach(),
                "cov_y_loss": cov_y_loss.detach(),
                "loss": loss.detach(),
                "std_x_mean": std_x.mean().detach(),
                "std_y_mean": std_y.mean().detach(),
                "std_x_min": std_x.min().detach(),
                "std_y_min": std_y.min().detach(),
                "x_mean_norm": x.mean(dim=0).norm().detach(),
                "y_mean_norm": y.mean(dim=0).norm().detach(),
                "cov_x_diag_mean": torch.diagonal(cov_x).mean().detach(),
                "cov_y_diag_mean": torch.diagonal(cov_y).mean().detach(),
                "cov_x_offdiag_abs_mean": _off_diagonal(cov_x).abs().mean().detach(),
                "cov_y_offdiag_abs_mean": _off_diagonal(cov_y).abs().mean().detach(),
            }

        return loss, stats
