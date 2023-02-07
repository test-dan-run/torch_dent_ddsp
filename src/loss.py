import torch
import torch.nn as nn

from typing import List, Union

class SpectralLoss(nn.Module):

    def __init__(
        self, 
        fft_sizes: List[int] = [2048, 1024, 512, 256, 128, 64],
        magnitude_weight: float = 1.0,
        log_magnitude_weight: float = 0.0,
    ):
        
        super(SpectralLoss, self).__init__()
        assert (magnitude_weight > 0 and log_magnitude_weight == 0) or (magnitude_weight == 0 and log_magnitude_weight > 0), \
                  'Only either magnitude_weight or log_magnitude_weight can be more than 0'

        self.fft_sizes = fft_sizes
        self.mag_weight = magnitude_weight
        self.logmag_weight = log_magnitude_weight

    
    def safe_log(self, x: Union[torch.Tensor, float], eps: float = 1e-5) -> torch.Tensor:

        safe_x = torch.where(x <= eps, eps, x)

        return torch.log(safe_x)
    
    def stft(self, audio: torch.Tensor, frame_size: int = 2048, overlap: float = 0.75) -> torch.Tensor:

        if audio.dim() == 3:
            assert audio.size(1) == 1, f'input audio should be a 2D batch of time sequences: {audio.size()}'
            audio = audio.squeeze(1)

        return torch.stft(
            audio, n_fft=frame_size,
            hop_length=int(frame_size * (1-overlap)),
            win_length=frame_size,
        )
    
    def mean_difference(self, true: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:

        diff = true - pred
        return torch.mean(torch.abs(diff))
    
    def __call__(self, true: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:

        loss: float = 0.0
        # compute loss for each fft size
        for fft_size in self.fft_sizes:

            true_mag = self.stft(true, frame_size=fft_size, overlap=0.75)
            pred_mag = self.stft(pred, frame_size=fft_size, overlap=0.75)

            true_mag = torch.abs(true_mag)
            pred_mag = torch.abs(pred_mag)

            if self.mag_weight > 0:
                loss += self.mag_weight * self.mean_difference(true_mag, pred_mag)
            
            if self.logmag_weight > 0:
                true_mag = self.safe_log(true_mag)
                pred_mag = self.safe_log(pred_mag)
                loss += self.logmag_weight * self.mean_difference(true_mag, pred_mag)

        return loss