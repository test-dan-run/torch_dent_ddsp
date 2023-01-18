import torch
import torch.nn as nn

from typing import List, Union

class SpectralLoss(nn.Module):

    def __init__(
        self, 
        fft_sizes: List[int] = [2048, 1024, 512, 256, 128, 64],
        loss_type: str = 'l1',
        magnitude_weight: float = 1.0,
        log_magnitude_weight: float = 0.0,
    ):
        
        super(SpectralLoss, self).__init__()
        self.fft_sizes = fft_sizes
        self.loss_type = loss_type
        self.mag_weight = magnitude_weight
        self.logmag_weight = log_magnitude_weight

    
    def safe_log(self, x: Union[torch.Tensor, float], eps: float = 1e-5) -> torch.Tensor:

        safe_x = torch.where(x <= eps, eps, x)

        return torch.log(safe_x)
    
    def stft(self, audio: torch.Tensor, frame_size: int = 2048, overlap: float = 0.75, pad_end: bool = True) -> torch.Tensor:

        pass