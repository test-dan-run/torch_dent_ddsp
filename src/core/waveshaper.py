import torch
import torch.nn as nn

class Distortion(torch.nn.Module):
    def __init__(self, threshold: float):
        super(Distortion, self).__init__()
        self.distortion_threshold = nn.Parameter(torch.tensor([threshold], dtype=torch.float32), requires_grad=True)

    def __call__(self, audio: torch.Tensor) -> torch.Tensor:
        return ( 2/torch.pi ) * torch.atan( (0.5*torch.pi*self.distortion_threshold) * audio )

if __name__ == '__main__':
    distort = Distortion(15)
    aud = torch.rand([1, 148212])
    print(distort(aud).shape)