import torch
import torch.nn as nn

from .utils import clip_by_value, frequency_filter

class FIRFilter(torch.nn.Module):
	"""Linear time-varying finite impulse response (LTV-FIR) filter."""

	def __init__(self, window_size=257, scale_fn = clip_by_value, magnitude_init = None, n_frequency_bins = None, n_frames = None):

		super(FIRFilter, self).__init__()
		self.window_size = window_size
		self.scale_fn = scale_fn
		self.n_frequency_bins = n_frequency_bins
		self.n_frames = n_frames

		if magnitude_init is None:
			if self.n_frames is None:
				size = (1, self.n_frequency_bins)
			else:
				size = (1, self.n_frames, self.n_frequency_bins)
			magnitude_init = torch.normal(mean=0.0, std=0.05, size=size)
			
		self.fir_magnitudes = nn.Parameter(magnitude_init, requires_grad=True)

	def __call__(self, audio: torch.Tensor) -> torch.Tensor:

		if self.scale_fn:
			magnitudes = self.scale_fn(self.fir_magnitudes)
		else:
			magnitudes = self.fir_magnitudes

		return frequency_filter(audio, magnitudes, self.window_size)

if __name__ == '__main__':

	import random
	random.seed(42)

	equalizer = FIRFilter(n_frequency_bins=1000)
	a = torch.rand([2,1,16000])
	print(equalizer(a))
	print(equalizer(a).shape)