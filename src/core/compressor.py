import torch
import torch.nn.functional as F
from torch.autograd import Variable

from .utils import overlap_and_add

def upsample_with_windows(
    inputs: torch.Tensor,
    upsample_factor: int,
    add_endpoint: bool = True) -> torch.Tensor:
    """Upsample a series of frames using using overlapping hann windows.
    Good for amplitude envelopes.
    Args:
        inputs: Framewise 3-D tensor. Shape [batch_size, n_frames, n_channels].
        n_timesteps: The time resolution of the output signal.
        add_endpoint: Hold the last timestep for an additional step as the endpoint.
        Then, n_timesteps is divided evenly into n_frames segments. If false, use
        the last timestep as the endpoint, producing (n_frames - 1) segments with
        each having a length of n_timesteps / (n_frames - 1).
    Returns:
        Upsampled 3-D tensor. Shape [batch_size, n_timesteps, n_channels].
    Raises:
        ValueError: If input does not have 3 dimensions.
        ValueError: If attempting to use function for downsampling.
        ValueError: If n_timesteps is not divisible by n_frames (if add_endpoint is
        true) or n_frames - 1 (if add_endpoint is false).
    """

    if len(inputs.shape) != 3:
        raise ValueError('Upsample_with_windows() only supports 3 dimensions, not {}.'.format(inputs.shape))

    # Mimic behavior of tf.image.resize.
    # For forward (not endpointed), hold value for last interval.
    if add_endpoint:
        inputs = torch.concat([inputs, inputs[:, :, -1:]], dim=2)

    # Constant overlap-add, half overlapping windows.
    window_length = 2 * upsample_factor
    window = torch.hann_window(window_length)  # [window]

    # Broadcast multiply.
    # Add dimension for windows [batch_size, n_channels, n_frames, window].
    inputs = inputs[:, :, :, None]
    window = window[None, None, None, :]
    inputs_windowed = (inputs * window) # batch, n_channel, n_frames, 1 * batch, n_channel, 1, window_size

    out = overlap_and_add(inputs_windowed, upsample_factor)

    # Trim the rise and fall of the first and last window.
    return out[:, :, upsample_factor:-upsample_factor]

def gain_smoothing(target_step: torch.Tensor, prev_step: torch.Tensor, attack_time: torch.Tensor, release_time: torch.Tensor) -> torch.Tensor:

    return torch.where(
                target_step >= prev_step, 
                attack_time  * prev_step + (1 - attack_time ) * target_step, 
                release_time * prev_step + (1 - release_time) * target_step
                )

class DynamicRangeCompressor(torch.nn.Module):
    def __init__(self, sample_rate: int, threshold: float, ratio: float, makeup: float, attack: float, release: float, downsample_factor: float = 16.0):

        super(DynamicRangeCompressor, self).__init__()

        self.sample_rate = sample_rate
        self.threshold = Variable(torch.tensor([threshold]), requires_grad=True)
        self.ratio = Variable(torch.tensor([ratio]), requires_grad=True)
        self.makeup = Variable(torch.tensor([makeup]), requires_grad=True)

        attack_time =  torch.exp( -torch.log10(torch.Tensor([9.0])) ) / (sample_rate * attack  * 1.0e-3 * downsample_factor)
        self.attack_time = Variable(torch.tensor([attack_time]), requires_grad=True)
        
        release_time = torch.exp( -torch.log10(torch.Tensor([9.0])) ) / (sample_rate * release * 1.0e-3 * downsample_factor)
        self.release_time = Variable(torch.tensor([release_time]), requires_grad=True)

        self.downsample_factor = downsample_factor

    def __call__(self, audio: torch.Tensor) -> torch.Tensor:
        
        if audio.dim() == 2:
            # expand to [batch, channel, timestep]
            audio = audio[None, :, :]
        audio_db = 20 * torch.log10(torch.abs(audio) + 1e-8)

        # compute gain based on threshold and ratio
        compressed_audio_db = torch.where(
            torch.greater(audio_db, self.threshold),
            self.threshold + ( (audio_db-self.threshold)/self.ratio ), audio_db 
            )
        gain = compressed_audio_db - audio_db # [batch, length]
        gain_downsampled = F.interpolate(gain, scale_factor=1/self.downsample_factor, mode = 'linear')
        gain_downsampled_smoothed = torch.zeros(gain_downsampled.size())

        prev_step = gain_downsampled[:,:, 0]
        for i in range(gain_downsampled.size(2)):
            prev_step = gain_smoothing(gain_downsampled[:,:,i], prev_step, self.attack_time, self.release_time)
            gain_downsampled_smoothed[:,:,i] = prev_step

        gain_downsampled_smoothed_upsampled = upsample_with_windows(gain_downsampled, int(self.downsample_factor))

        smoothed_compressed_audio_db = audio_db + gain_downsampled_smoothed_upsampled[:,:,:gain.shape[2]] + self.makeup
        smoothed_compressed_audio = 10.0 ** ((smoothed_compressed_audio_db)/20.0)
        smoothed_compressed_audio = torch.where(audio < 0, -smoothed_compressed_audio, smoothed_compressed_audio)

        return smoothed_compressed_audio

if __name__ == '__main__':
    drc = DynamicRangeCompressor(16000, -10.0, 30.0, 0.0, 1.0e-7, 1.0e-3, 2.0)
    a = torch.rand([1,16000])
    print(drc(a))
    print(drc(a).shape)