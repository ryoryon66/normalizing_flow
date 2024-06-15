import torch
import torch.nn as nn

from typing import Literal


class CouplingLayer(nn.Module):
    def __init__(
        self, feature_dim: int, hidden_dim: int, mask_type: Literal["odd", "even"]
    ):
        super(CouplingLayer, self).__init__()

        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim

        self.mask: torch.Tensor = torch.zeros(feature_dim).unsqueeze(0)
        if mask_type == "odd":
            self.mask[0, ::2] = 1
        elif mask_type == "even":
            self.mask[0, 1::2] = 1
        else:
            raise ValueError("Invalid mask type")

        self.s = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feature_dim//2),
            nn.Tanh(),
        )

        self.t = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feature_dim//2),
        )

        return

    def forward_flow(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mask = self.mask.to(x.device)
        x_masked = x * mask
        s = self.s(x_masked)
        t = self.t(x_masked)
        y = x * mask + (1 - mask) * (x * torch.exp(s) + t)
        
        assert s.shape[1] == self.feature_dim // 2

        log_det_jacobian = (s).sum(dim=1)

        return y, log_det_jacobian

    def inverse_flow(self, y: torch.Tensor):
        mask = self.mask.to(y.device)
        y_masked = y * mask
        s = self.s(y_masked)
        t = self.t(y_masked)
        x = y * mask + (1 - mask) * (y - t) * torch.exp(-s)

        return x


class NormalizingFlow(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int,
        num_layers: int,
    ):
        super(NormalizingFlow, self).__init__()

        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self._prior_dist = torch.distributions.Normal(
            torch.tensor(0.0), torch.tensor(1.0)
        )

        self.layers = nn.ModuleList()
        for i in range(num_layers):
            mask_type: Literal["odd", "even"] = "odd" if i % 2 == 0 else "even"
            self.layers.append(CouplingLayer(feature_dim, hidden_dim, mask_type))

        return

    def forward(
        self, x: torch.Tensor, 
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the normalizing flow model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Tuple containing the output tensor and the total log determinant of the Jacobian of the transformations.

        """

        log_det_jacobian_total = torch.zeros(x.size(0)).to(x.device)
        z = x
        for i, layer in enumerate(self.layers): 
            z, log_det_jacobian = layer.forward_flow(z)
            log_det_jacobian_total += log_det_jacobian

        return z, log_det_jacobian_total

    def inverse(self, y: torch.Tensor,steps:int=-1) -> torch.Tensor:
        steps = self.num_layers if steps == -1 else steps
        x = y
        for layer in reversed(self.layers):
            if steps == 0:
                break
            x = layer.inverse_flow(x)
            steps -= 1

        return x

    def sample_from_prior(self, num_samples: int) -> torch.Tensor:
        """
        Sample from the prior distribution.
        """
        z = self._prior_dist.sample((num_samples, self.feature_dim))
        return z

    def log_likelihood(self, x: torch.Tensor) -> torch.Tensor:
        z, log_det_jacobian_total = self.forward(x)
        log_pz = self._prior_dist.log_prob(z).sum(dim=1)
        log_px = log_pz + log_det_jacobian_total
        return log_px
