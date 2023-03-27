import sys
sys.path.append('/data/nhandt23/Workspace/tacotron2')
import numpy as np
from scipy.io.wavfile import read
import torch
import layers
from hparams import create_hparams
from tqdm import tqdm

hparams = create_hparams()
stft = layers.TacotronSTFT(
          hparams.filter_length, hparams.hop_length, hparams.win_length,
          hparams.n_mel_channels, hparams.sampling_rate, hparams.mel_fmin,
          hparams.mel_fmax)

if __name__ == "__main__":
     input_file = sys.argv[1]
     output_dir = sys.argv[2]

     with open(input_file, "r", encoding="utf-8") as f:
          lines = f.read().splitlines()
          for line in tqdm(lines):
               p, s = line.split("|")

               p = "/data/nhandt23/Workspace/Dataset/N_F01_v3.1.0/N_F01_KhanhLinh_AllAudio_channel2_22050_processed_mute_silences/" + p + ".wav"

               sampling_rate, data = read(p)
               audio, sampling_rate = torch.FloatTensor(data.astype(np.float32)), sampling_rate
               if sampling_rate != stft.sampling_rate:
                    raise ValueError("{} {} SR doesn't match target {} SR".format(sampling_rate, stft.sampling_rate))
               audio_norm = audio / hparams.max_wav_value
               audio_norm = audio_norm.unsqueeze(0)
               audio_norm = torch.autograd.Variable(audio_norm, requires_grad=False)
               melspec = stft.mel_spectrogram(audio_norm)
               melspec = torch.squeeze(melspec, 0)

               save_file = output_dir + "/" + p.split("/")[-1]
               np.save(save_file.replace(".wav",".npy"), melspec.cpu().detach().numpy())