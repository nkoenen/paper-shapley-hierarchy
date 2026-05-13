import torch
import torch.nn.functional as F
from torch import distributions
from torch.distributions import transforms as T

from pytorch_forecasting.metrics import DistributionLoss
from pytorch_forecasting.data.encoders import TorchNormalizer


class StudentTDistributionLoss(DistributionLoss):
    """
    Student-T distribution loss. Works with any target normalizer.
    """

    distribution_class = distributions.StudentT
    distribution_arguments = ["df", "loc", "scale"]

    def map_x_to_distribution(self, x: torch.Tensor) -> distributions.StudentT:
        distr = distributions.StudentT(df=x[..., 2], loc=x[..., 3], scale=x[..., 4])
        scaler = distributions.AffineTransform(loc=x[..., 0], scale=x[..., 1])
        if self._transformation is None:
            return distributions.TransformedDistribution(distr, [scaler])
        else:
            raise NotImplementedError("StudentTDistributionLoss does not yet support ", 
                                      "target normalizers with transformations.") 

    def rescale_parameters(
        self,
        parameters: torch.Tensor,
        target_scale: torch.Tensor,
        encoder,
    ) -> torch.Tensor:
        self._transformation = encoder.transformation
        df    = F.softplus(parameters[..., 0]) + 2  # enforce df > 2 for finite variance
        loc   = parameters[..., 1]
        scale = F.softplus(parameters[..., 2])
        return torch.concat([
            target_scale.unsqueeze(1).expand(-1, loc.size(1), -1),
            df.unsqueeze(-1),
            loc.unsqueeze(-1),
            scale.unsqueeze(-1),
        ], dim=-1)


class LogNormalDistributionLoss(DistributionLoss):
    """
    Log-normal distribution loss. Works with any target normalizer.
    Fits a Normal in normalized space and applies ExpTransform to recover a
    LogNormal in the original space.
    """

    distribution_class = distributions.Normal
    distribution_arguments = ["loc", "scale"]

    def map_x_to_distribution(self, x: torch.Tensor) -> distributions.TransformedDistribution:
        distr = distributions.Normal(loc=x[..., 2], scale=x[..., 3])
        scaler = distributions.AffineTransform(loc=x[..., 0], scale=x[..., 1])
        if self._transformation is None:
            return distributions.TransformedDistribution(distr, [scaler, T.ExpTransform()])
        else:
            raise NotImplementedError("LogNormalDistributionLoss does not yet support ",
                                      "target normalizers with transformations.")

    def rescale_parameters(
        self,
        parameters: torch.Tensor,
        target_scale: torch.Tensor,
        encoder,
    ) -> torch.Tensor:
        self._transformation = encoder.transformation
        loc   = parameters[..., 0]
        # σ is bounded smoothly to (1e-3, σ_max] via sigmoid
        scale = torch.sigmoid(parameters[..., 1]) * 1.0 + 1e-3
        return torch.concat([
            target_scale.unsqueeze(1).expand(-1, loc.size(1), -1),
            loc.unsqueeze(-1),
            scale.unsqueeze(-1),
        ], dim=-1)

    def loss(self, y_pred: torch.Tensor, y_actual: torch.Tensor) -> torch.Tensor:
        distribution = self.map_x_to_distribution(y_pred)
        return -distribution.log_prob(y_actual.clamp(min=1e-6))
