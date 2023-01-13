import torch

class Distortion(torch.nn.Module):
    def __init__(self, threshold: float):
        super(Distortion, self).__init__()
        self.threshold = threshold

    def __call__(self, audio: torch.Tensor) -> torch.Tensor:
        return ( 2/torch.pi ) * torch.atan( (0.5*torch.pi*self.threshold) * audio )
