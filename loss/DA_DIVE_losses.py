"""
Losses for DA-DIVE.

The frequency loss uses orthonormal FFT normalization and log magnitudes,
which makes its scale more comparable across 1/4, 1/2, and full-resolution
outputs than a raw unnormalized real/imaginary FFT L1 loss.
"""

from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_msssim import MS_SSIM


class CharbonnierLoss(nn.Module):
    def __init__(self, eps: float = 1e-3):
        super().__init__()
        self.eps = eps

    def forward(
        self, prediction: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        return torch.sqrt(
            (prediction - target).square() + self.eps**2
        ).mean()


class GradientLoss(nn.Module):
    def __init__(self):
        super().__init__()
        sobel_x = torch.tensor(
            [[-1.0, 0.0, 1.0],
             [-2.0, 0.0, 2.0],
             [-1.0, 0.0, 1.0]]
        ).view(1, 1, 3, 3) / 8.0
        sobel_y = sobel_x.transpose(-1, -2)

        self.register_buffer("sobel_x", sobel_x)
        self.register_buffer("sobel_y", sobel_y)

    def _gradient(
        self, image: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        channels = image.shape[1]
        kernel_x = self.sobel_x.expand(channels, 1, 3, 3)
        kernel_y = self.sobel_y.expand(channels, 1, 3, 3)

        grad_x = F.conv2d(
            image,
            kernel_x,
            padding=1,
            groups=channels,
        )
        grad_y = F.conv2d(
            image,
            kernel_y,
            padding=1,
            groups=channels,
        )
        return grad_x, grad_y

    def forward(
        self, prediction: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        pred_x, pred_y = self._gradient(prediction)
        target_x, target_y = self._gradient(target)
        return (
            F.l1_loss(pred_x, target_x)
            + F.l1_loss(pred_y, target_y)
        )


class ScaleNormalizedFrequencyLoss(nn.Module):
    """
    Compares log-magnitude spectra after an orthonormal 2-D real FFT.

    A smooth radial split separates:
      - low frequencies: global color/illumination/contrast,
      - high frequencies: edges and texture.
    """

    def __init__(
        self,
        cutoff: float = 0.18,
        sharpness: float = 40.0,
        low_weight: float = 0.5,
        high_weight: float = 1.0,
    ):
        super().__init__()
        self.cutoff = cutoff
        self.sharpness = sharpness
        self.low_weight = low_weight
        self.high_weight = high_weight

    @staticmethod
    def _masked_mean(
        values: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        denominator = (
            mask.sum() * values.shape[0] * values.shape[1]
        ).clamp_min(1e-8)
        return (values * mask).sum() / denominator

    def forward(
        self, prediction: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        # FFT is kept in float32 even when mixed precision is enabled.
        prediction_f = torch.fft.rfft2(
            prediction.float(),
            dim=(-2, -1),
            norm="ortho",
        )
        target_f = torch.fft.rfft2(
            target.float(),
            dim=(-2, -1),
            norm="ortho",
        )

        pred_log_magnitude = torch.log1p(torch.abs(prediction_f))
        target_log_magnitude = torch.log1p(torch.abs(target_f))
        difference = torch.abs(
            pred_log_magnitude - target_log_magnitude
        )

        height, width_rfft = difference.shape[-2:]
        original_width = (width_rfft - 1) * 2

        fy = torch.fft.fftfreq(
            height,
            device=difference.device,
            dtype=difference.dtype,
        ).view(height, 1)
        fx = torch.fft.rfftfreq(
            original_width,
            device=difference.device,
            dtype=difference.dtype,
        ).view(1, width_rfft)

        radius = torch.sqrt(fy.square() + fx.square())
        low_mask = torch.sigmoid(
            (self.cutoff - radius) * self.sharpness
        ).view(1, 1, height, width_rfft)
        high_mask = 1.0 - low_mask

        low_loss = self._masked_mean(difference, low_mask)
        high_loss = self._masked_mean(difference, high_mask)

        return (
            self.low_weight * low_loss
            + self.high_weight * high_loss
        )


class ColorStatisticsLoss(nn.Module):
    """
    Matches per-channel mean and standard deviation.
    This gives explicit supervision to global color balance without an
    external RGB-to-Lab dependency.
    """

    def __init__(self, std_weight: float = 0.5):
        super().__init__()
        self.std_weight = std_weight

    def forward(
        self, prediction: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        pred_mean = prediction.mean(dim=(-2, -1))
        target_mean = target.mean(dim=(-2, -1))

        pred_std = prediction.var(
            dim=(-2, -1), unbiased=False
        ).add(1e-6).sqrt()
        target_std = target.var(
            dim=(-2, -1), unbiased=False
        ).add(1e-6).sqrt()

        return (
            F.l1_loss(pred_mean, target_mean)
            + self.std_weight
            * F.l1_loss(pred_std, target_std)
        )


class DADIVELoss(nn.Module):
    def __init__(
        self,
        scale_weights: Sequence[float] = (0.25, 0.5, 1.0),
        lambda_frequency: float = 0.10,
        lambda_gradient: float = 0.10,
        lambda_msssim: float = 0.20,
        lambda_color: float = 0.05,
    ):
        super().__init__()
        if len(scale_weights) != 3:
            raise ValueError(
                "DA-DIVE produces exactly three output scales."
            )

        self.scale_weights = tuple(float(v) for v in scale_weights)
        self.lambda_frequency = lambda_frequency
        self.lambda_gradient = lambda_gradient
        self.lambda_msssim = lambda_msssim
        self.lambda_color = lambda_color

        self.charbonnier = CharbonnierLoss()
        self.gradient = GradientLoss()
        self.frequency = ScaleNormalizedFrequencyLoss()
        self.color = ColorStatisticsLoss()
        self.ms_ssim = MS_SSIM(
            data_range=1.0,
            size_average=True,
            channel=3,
            win_size=3,
            win_sigma=1.5,
        )

    @staticmethod
    def _target_pyramid(
        predictions: List[torch.Tensor],
        target: torch.Tensor,
    ) -> List[torch.Tensor]:
        targets = []
        for prediction in predictions:
            if prediction.shape[-2:] == target.shape[-2:]:
                targets.append(target)
            else:
                targets.append(
                    F.interpolate(
                        target,
                        size=prediction.shape[-2:],
                        mode="bilinear",
                        align_corners=False,
                    )
                )
        return targets

    def forward(
        self,
        predictions: List[torch.Tensor],
        target: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if len(predictions) != 3:
            raise ValueError(
                f"Expected three predictions, received {len(predictions)}."
            )

        # Keep every loss calculation in float32 even when model inference
        # uses autocast. Casting preserves the gradient path to the model.
        predictions = [prediction.float() for prediction in predictions]
        target = target.float()

        targets = self._target_pyramid(predictions, target)

        reconstruction_loss = target.new_tensor(0.0)
        frequency_loss = target.new_tensor(0.0)
        gradient_loss = target.new_tensor(0.0)

        for weight, prediction, scale_target in zip(
            self.scale_weights,
            predictions,
            targets,
        ):
            reconstruction_loss = (
                reconstruction_loss
                + weight
                * self.charbonnier(prediction, scale_target)
            )
            frequency_loss = (
                frequency_loss
                + weight
                * self.frequency(prediction, scale_target)
            )
            gradient_loss = (
                gradient_loss
                + weight
                * self.gradient(prediction, scale_target)
            )

        full_prediction = predictions[-1]
        clipped_prediction = full_prediction.clamp(0.0, 1.0)
        clipped_target = target.clamp(0.0, 1.0)

        msssim_loss = 1.0 - self.ms_ssim(
            clipped_prediction,
            clipped_target,
        )
        color_loss = self.color(
            clipped_prediction,
            clipped_target,
        )

        total = (
            reconstruction_loss
            + self.lambda_frequency * frequency_loss
            + self.lambda_gradient * gradient_loss
            + self.lambda_msssim * msssim_loss
            + self.lambda_color * color_loss
        )

        components = {
            "total": total.detach(),
            "reconstruction": reconstruction_loss.detach(),
            "frequency": frequency_loss.detach(),
            "gradient": gradient_loss.detach(),
            "msssim": msssim_loss.detach(),
            "color": color_loss.detach(),
        }
        return total, components
