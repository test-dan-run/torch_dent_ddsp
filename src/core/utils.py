import torch
import torch.nn.functional as F

import math
import numpy as np
from scipy import fftpack
from typing import Union, Iterable, Tuple

def clip_by_value(x: Union[float, torch.Tensor], min_value: float = 1e-12, max_value: float = 0.5) -> torch.Tensor:
	if not isinstance(x, torch.Tensor):
		x = torch.Tensor([x]) 
	return torch.clip(x, min = min_value, max = max_value)

def apply_window_to_impulse_response(impulse_response: torch.Tensor,
									 window_size: int = 0,
									 causal: bool = False) -> torch.Tensor:
	"""Apply a window to an impulse response and put in causal form.
	Args:
		impulse_response: A series of impulse responses frames to window, of shape
		[batch, n_frames, ir_size].
		window_size: Size of the window to apply in the time domain. If window_size
		is less than 1, it defaults to the impulse_response size.
		causal: Impulse response input is in causal form (peak in the middle).
	Returns:
		impulse_response: Windowed impulse response in causal form, with last
		dimension cropped to window_size if window_size is greater than 0 and less
		than ir_size.
	"""

	# If IR is in causal form, put it in zero-phase form.
	if causal:
		impulse_response = torch.fft.fftshift(impulse_response, dim=-1)

	# Get a window for better time/frequency resolution than rectangular.
	# Window defaults to IR size, cannot be bigger.
	ir_size = int(impulse_response.shape[-1])
	if (window_size <= 0) or (window_size > ir_size):
		window_size = ir_size
	window = torch.hann_window(window_size).to('cuda')

	# Zero pad the window and put in in zero-phase form.
	padding = ir_size - window_size
	if padding > 0:
		half_idx = (window_size + 1) // 2
		window = torch.concat([window[half_idx:],
							torch.zeros([padding]).to('cuda'),
							window[:half_idx]], dim=0)
	else:
		window = torch.fft.fftshift(window, dim=-1)

	# Apply the window, to get new IR (both in zero-phase form).
	window = torch.broadcast_to(window, impulse_response.shape)
	impulse_response = window * torch.real(impulse_response)

	# Put IR in causal form and trim zero padding.
	if padding > 0:
		first_half_start = (ir_size - (half_idx - 1)) + 1
		second_half_end = half_idx + 1
		impulse_response = torch.concat([impulse_response[..., first_half_start:],
									impulse_response[..., :second_half_end]],
									dim=-1)
	else:
		impulse_response = torch.fft.fftshift(impulse_response, axes=-1)

	return impulse_response

def complex_matmul(a: torch.Tensor, b: torch.Tensor, groups: int = 1) -> torch.Tensor:
    """Multiplies two complex-valued tensors."""
    # Scalar matrix multiplication of two tensors, over only the first channel
    # dimensions. Dimensions 3 and higher will have the same shape after multiplication.
    # We also allow for "grouped" multiplications, where multiple sections of channels
    # are multiplied independently of one another (required for group convolutions).
    a = a.view(a.size(0), groups, -1, *a.shape[2:]).to('cuda')
    b = b.view(groups, -1, *b.shape[1:]).to('cuda')

    a = torch.movedim(a, 2, a.dim() - 1).unsqueeze(-2)
    b = torch.movedim(b, (1, 2), (b.dim() - 1, b.dim() - 2))

    # complex value matrix multiplication
    real = a.real @ b.real - a.imag @ b.imag
    imag = a.imag @ b.real + a.real @ b.imag
    real = torch.movedim(real, real.dim() - 1, 2).squeeze(-1)
    imag = torch.movedim(imag, imag.dim() - 1, 2).squeeze(-1)
    c = torch.zeros(real.shape, dtype=torch.complex64, device=a.device)
    c.real, c.imag = real, imag

    return c.view(c.size(0), -1, *c.shape[3:])

def to_ntuple(val: Union[int, Iterable[int]], n: int) -> Tuple[int, ...]:
    """Casts to a tuple with length 'n'.  Useful for automatically computing the
    padding and stride for convolutions, where users may only provide an integer.
    Args:
        val: (Union[int, Iterable[int]]) Value to cast into a tuple.
        n: (int) Desired length of the tuple
    Returns:
        (Tuple[int, ...]) Tuple of length 'n'
    """
    if isinstance(val, Iterable):
        out = tuple(val)
        if len(out) == n:
            return out
        else:
            raise ValueError(f"Cannot cast tuple of length {len(out)} to length {n}.")
    else:
        return n * (val,)

def overlap_and_add(signal: torch.Tensor, frame_step: int):
    """Reconstructs a signal from a framed representation.
    Adds potentially overlapping frames of a signal with shape
    `[..., frames, frame_length]`, offsetting subsequent frames by `frame_step`.
    The resulting tensor has shape `[..., output_size]` where
        output_size = (frames - 1) * frame_step + frame_length
    Args:
        signal: A [..., frames, frame_length] Tensor. All dimensions may be unknown, and rank must be at least 2.
        frame_step: An integer denoting overlap offsets. Must be less than or equal to frame_length.
    Returns:
        A Tensor with shape [..., output_size] containing the overlap-added frames of signal's inner-most two dimensions.
        output_size = (frames - 1) * frame_step + frame_length
    Based on https://github.com/tensorflow/tensorflow/blob/r1.12/tensorflow/contrib/signal/python/ops/reconstruction_ops.py
    """
    outer_dimensions = signal.size()[:-2]
    frames, frame_length = signal.size()[-2:]

    subframe_length = math.gcd(frame_length, frame_step)  # gcd=Greatest Common Divisor
    subframe_step = frame_step // subframe_length
    subframes_per_frame = frame_length // subframe_length
    output_size = frame_step * (frames - 1) + frame_length
    output_subframes = output_size // subframe_length

    subframe_signal = signal.view(*outer_dimensions, -1, subframe_length)

    frame = torch.arange(0, output_subframes).unfold(0, subframes_per_frame, subframe_step)
    # frame = signal.new_tensor(frame).long()  # signal may in GPU or CPU
    frame = frame.contiguous().view(-1).to('cuda')

    result = signal.new_zeros(*outer_dimensions, output_subframes, subframe_length)
    result.index_add_(-2, frame, subframe_signal)
    result = result.view(*outer_dimensions, -1)
    return result

def frame(signal, frame_length, frame_step, pad_end=False, pad_value=0, axis=-1):
	"""
	equivalent of tf.signal.frame
	"""
	signal_length = signal.shape[axis]
	if pad_end:
		frames_overlap = frame_length - frame_step
		rest_samples = np.abs(signal_length - frames_overlap) % np.abs(frame_length - frames_overlap)
		pad_size = int(frame_length - rest_samples)
		if pad_size != 0:
			pad_axis = [0] * signal.ndim
			pad_axis[axis] = pad_size
			signal = F.pad(signal, pad_axis, "constant", pad_value)

	frames=signal.unfold(axis, frame_length, frame_step+1) 
	return frames

def get_fft_size(frame_size: int, ir_size: int, power_of_2: bool = True) -> int:
	"""Calculate final size for efficient FFT.
	Args:
		frame_size: Size of the audio frame.
		ir_size: Size of the convolving impulse response.
		power_of_2: Constrain to be a power of 2. If False, allow other 5-smooth
		numbers. TPU requires power of 2, while GPU is more flexible.
	Returns:
		fft_size: Size for efficient FFT.
	"""
	convolved_frame_size = ir_size + frame_size - 1
	if power_of_2:
		# Next power of 2.
		fft_size = int(2**np.ceil(np.log2(convolved_frame_size)))
	else:
		fft_size = int(fftpack.helper.next_fast_len(convolved_frame_size))
	return fft_size

def crop_and_compensate_delay(audio: torch.Tensor, audio_size: int, ir_size: int,
							  padding: str,
							  delay_compensation: int) -> torch.Tensor:
	"""Crop audio output from convolution to compensate for group delay.
	Args:
		audio: Audio after convolution. Tensor of shape [batch, time_steps].
		audio_size: Initial size of the audio before convolution.
		ir_size: Size of the convolving impulse response.
		padding: Either 'valid' or 'same'. For 'same' the final output to be the
		same size as the input audio (audio_timesteps). For 'valid' the audio is
		extended to include the tail of the impulse response (audio_timesteps +
		ir_timesteps - 1).
		delay_compensation: Samples to crop from start of output audio to compensate
		for group delay of the impulse response. If delay_compensation < 0 it
		defaults to automatically calculating a constant group delay of the
		windowed linear phase filter from frequency_impulse_response().
	Returns:
		Tensor of cropped and shifted audio.
	Raises:
		ValueError: If padding is not either 'valid' or 'same'.
	"""
	# Crop the output.
	if padding == 'valid':
		crop_size = ir_size + audio_size - 1
	elif padding == 'same':
		crop_size = audio_size
	else:
		raise ValueError('Padding must be \'valid\' or \'same\', instead '
						'of {}.'.format(padding))

	# Compensate for the group delay of the filter by trimming the front.
	# For an impulse response produced by frequency_impulse_response(),
	# the group delay is constant because the filter is linear phase.
	total_size = int(audio.shape[-1])
	crop = total_size - crop_size
	start = ((ir_size - 1) // 2 -
			1 if delay_compensation < 0 else delay_compensation)
	end = crop - start
	return audio[:, start:-end]

def fft_convolve(audio: torch.Tensor,
				 impulse_response: torch.Tensor,
				 padding: str = 'same',
				 delay_compensation: int = -1) -> torch.Tensor:
	"""Filter audio with frames of time-varying impulse responses.
	Time-varying filter. Given audio [batch, n_samples], and a series of impulse
	responses [batch, n_frames, n_impulse_response], splits the audio into frames,
	applies filters, and then overlap-and-adds audio back together.
	Applies non-windowed non-overlapping STFT/ISTFT to efficiently compute
	convolution for large impulse response sizes.
	Args:
		audio: Input audio. Tensor of shape [batch, audio_timesteps].
		impulse_response: Finite impulse response to convolve. Can either be a 2-D
		Tensor of shape [batch, ir_size], or a 3-D Tensor of shape [batch,
		ir_frames, ir_size]. A 2-D tensor will apply a single linear
		time-invariant filter to the audio. A 3-D Tensor will apply a linear
		time-varying filter. Automatically chops the audio into equally shaped
		blocks to match ir_frames.
		padding: Either 'valid' or 'same'. For 'same' the final output to be the
		same size as the input audio (audio_timesteps). For 'valid' the audio is
		extended to include the tail of the impulse response (audio_timesteps +
		ir_timesteps - 1).
		delay_compensation: Samples to crop from start of output audio to compensate
		for group delay of the impulse response. If delay_compensation is less
		than 0 it defaults to automatically calculating a constant group delay of
		the windowed linear phase filter from frequency_impulse_response().
	Returns:
		audio_out: Convolved audio. Tensor of shape
			[batch, audio_timesteps + ir_timesteps - 1] ('valid' padding) or shape
			[batch, audio_timesteps] ('same' padding).
	Raises:
		ValueError: If audio and impulse response have different batch size.
		ValueError: If audio cannot be split into evenly spaced frames. (i.e. the
		number of impulse response frames is on the order of the audio size and
		not a multiple of the audio size.)
	"""
	# Get shapes of audio.
	if audio.dim() == 3:
		if audio.size(1) == 1:
			audio = torch.squeeze(audio, dim=1)
	batch_size, audio_size = audio.size()

	# Add a frame dimension to impulse response if it doesn't have one.
	if impulse_response.dim() == 2:
		impulse_response = impulse_response[:, None, :]

	# Broadcast impulse response.
	if impulse_response.size(0) == 1 and batch_size > 1:
		impulse_response = torch.tile(impulse_response, [batch_size, 1, 1])

	# Get shapes of impulse response.
	batch_size_ir, n_ir_frames, ir_size = impulse_response.size()

	# Validate that batch sizes match.
	if batch_size != batch_size_ir:
		raise ValueError('Batch size of audio ({}) and impulse response ({}) must '
						'be the same.'.format(batch_size, batch_size_ir))

	# Cut audio into frames.
	frame_size = int(np.ceil(audio_size / n_ir_frames))
	hop_size = frame_size
	audio_frames = frame(audio, frame_size, hop_size, pad_end=True)

	# Check that number of frames match.
	n_audio_frames = int(audio_frames.size(1))
	if n_audio_frames != n_ir_frames:
		raise ValueError(
			'Number of Audio frames ({}) and impulse response frames ({}) do not '
			'match. For small hop size = ceil(audio_size / n_ir_frames), '
			'number of impulse response frames must be a multiple of the audio '
			'size.'.format(n_audio_frames, n_ir_frames))

	# Pad and FFT the audio and impulse responses.
	fft_size = get_fft_size(frame_size, ir_size, power_of_2=True)
	audio_fft = torch.fft.rfft(audio_frames, fft_size)
	ir_fft = torch.fft.rfft(impulse_response, fft_size)

	# Multiply the FFTs (same as convolution in time).
	ir_fft.imag *= -1
	audio_ir_fft = complex_matmul(audio_fft, ir_fft)

	# print(f"inspecting inside: audio fft shape-->{audio_fft.shape},ir fft shape-->{ir_fft.shape}, audio_ir_fft-->{audio_ir_fft.shape}")
	# Take the IFFT to resynthesize audio.
	audio_frames_out = torch.fft.irfft(audio_ir_fft)
	audio_out = overlap_and_add(audio_frames_out, hop_size)

	# Crop and shift the output audio.
	return crop_and_compensate_delay(audio_out, audio_size, ir_size, padding,
									delay_compensation)

def frequency_impulse_response(
	magnitudes: torch.Tensor,
	window_size: int = 0) -> torch.Tensor:

	# imaginary part == 0
	magnitudes = torch.complex(magnitudes, torch.zeros_like(magnitudes))
	impulse_response = torch.fft.irfft(magnitudes)

	# window and put in causal form
	impulse_response = apply_window_to_impulse_response(impulse_response, window_size)

	return impulse_response

def frequency_filter(audio: torch.Tensor,
						magnitudes: torch.Tensor,
						window_size: int = 0,
						padding: str = 'same') -> torch.Tensor:
	"""Filter audio with a finite impulse response filter.
	Args:
		audio: Input audio. Tensor of shape [batch, channel, audio_timesteps].
		magnitudes: Frequency transfer curve. Float32 Tensor of shape [batch,
		n_frames, n_frequencies] or [batch, n_frequencies]. The frequencies of the
		last dimension are ordered as [0, f_nyqist / (n_frequencies -1), ...,
		f_nyquist], where f_nyquist is (sample_rate / 2). Automatically splits the
		audio into equally sized frames to match frames in magnitudes.
		window_size: Size of the window to apply in the time domain. If window_size
		is less than 1, it is set as the default (n_frequencies).
		padding: Either 'valid' or 'same'. For 'same' the final output to be the
		same size as the input audio (audio_timesteps). For 'valid' the audio is
		extended to include the tail of the impulse response (audio_timesteps +
		window_size - 1).
	Returns:
		Filtered audio. Tensor of shape
			[batch, audio_timesteps + window_size - 1] ('valid' padding) or shape
			[batch, audio_timesteps] ('same' padding).
	"""
	impulse_response = frequency_impulse_response(magnitudes,
													window_size=window_size)

	return fft_convolve(audio, impulse_response, padding=padding)