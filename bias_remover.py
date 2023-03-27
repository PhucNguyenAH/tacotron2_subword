import sys
import torch
from stft import STFT


class hifiganBiasRemover(torch.nn.Module):
     """ Removes model bias from audio produced with waveglow """

     def __init__(self, model, filter_length=1024, n_overlap=4,
                    win_length=1024, mode='zeros'):
          super(hifiganBiasRemover, self).__init__()
          self.stft = STFT(filter_length=filter_length,
                              hop_length=int(filter_length/n_overlap),
                              win_length=win_length).cuda()
          if mode == 'zeros':
               mel_input = torch.zeros(
                    (1, 80, 88)).cuda()
          elif mode == 'normal':
               mel_input = torch.randn(
                    (1, 80, 88)).cuda()
          else:
               raise Exception("Mode {} if not supported".format(mode))

          with torch.no_grad():
               bias_audio = model(mel_input).float() # [B, 1, T]

               bias_spec, _ = self.stft.transform(bias_audio.squeeze(0))

          self.register_buffer('bias_spec', bias_spec[:, :, 0][:, :, None])

     def forward(self, audio, strength=0.1):
          audio_spec, audio_angles = self.stft.transform(audio.cuda().float())
          audio_spec_denoised = audio_spec.cuda() - self.bias_spec * strength
          audio_spec_denoised = torch.clamp(audio_spec_denoised, 0.0)
          audio_denoised = self.stft.inverse(audio_spec_denoised, audio_angles.cuda())
          return audio_denoised

class waveglowBiasRemover(torch.nn.Module):
     """ Removes model bias from audio produced with waveglow """

     def __init__(self, model_path='Outdir/waveglow_256channels.pt', filter_length=1024, n_overlap=4,
                    win_length=1024, mode='zeros'):
          super(waveglowBiasRemover, self).__init__()
          self.stft = STFT(filter_length=filter_length,
                              hop_length=int(filter_length/n_overlap),
                              win_length=win_length).cuda()

          waveglow = torch.load(model_path)['model'].cuda().eval()

          if mode == 'zeros':
               mel_input = torch.zeros(
                    (1, 80, 88),
                    dtype=waveglow.upsample.weight.dtype,
                    device=waveglow.upsample.weight.device)
          elif mode == 'normal':
               mel_input = torch.randn(
                    (1, 80, 88),
                    dtype=waveglow.upsample.weight.dtype,
                    device=waveglow.upsample.weight.device)
          else:
               raise Exception("Mode {} if not supported".format(mode))

          with torch.no_grad():
               bias_audio = waveglow.infer(mel_input, sigma=0.0).float()
               bias_spec, _ = self.stft.transform(bias_audio)

          self.register_buffer('bias_spec', bias_spec[:, :, 0][:, :, None])

     def forward(self, audio, strength=0.1):
          audio_spec, audio_angles = self.stft.transform(audio.cuda().float())
          audio_spec_denoised = audio_spec - self.bias_spec * strength
          audio_spec_denoised = torch.clamp(audio_spec_denoised, 0.0)
          audio_denoised = self.stft.inverse(audio_spec_denoised, audio_angles)
          return audio_denoised[:, 0][0]