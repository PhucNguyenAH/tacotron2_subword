from g2p.text_to_sequence import Text2Seq
import torch, json, glob
import librosa, sys, os
import numpy as np
from hparams import create_hparams
from model import Tacotron2
from train import prepare_dataloaders
from hparams import create_hparams
from utils import to_gpu
import layers
from utils import load_wav_to_torch

def GTA_Synthesis(training_files, checkpoint_path, save_mel_folder):

     hparams = create_hparams()

     hparams.training_files = training_files
     
     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

     model = Tacotron2(hparams).cuda()

     if torch.cuda.is_available():
          model.load_state_dict(torch.load(checkpoint_path)['state_dict'])
          _ = model.to(device).eval()
     else:
          model.load_state_dict(torch.load(checkpoint_path, map_location='cpu')['state_dict'])
          _ = model.to(device).eval()  

     stft = layers.TacotronSTFT(
               hparams.filter_length, hparams.hop_length, hparams.win_length,
               hparams.n_mel_channels, hparams.sampling_rate, hparams.mel_fmin,
               hparams.mel_fmax)

     with open(training_files, "r", encoding="utf-8") as f:
          lines = f.read().splitlines()
          for line in lines:
               audiopath, text = line.split("|")

               file_name = text.split("/")[-1]

               text = torch.from_numpy(np.load(text).astype(int))[:,0]

               audio, _ = load_wav_to_torch(audiopath)
               audio_norm = audio / hparams.max_wav_value
               audio_norm = audio_norm.unsqueeze(0)
               audio_norm = torch.autograd.Variable(audio_norm, requires_grad=False)
               melspec = stft.mel_spectrogram(audio_norm)
               mel = torch.squeeze(melspec, 0)

               input_lengths = to_gpu(torch.LongTensor([len(text)])).long()
               text = to_gpu(text.unsqueeze(0)).long()
               max_len = torch.max(input_lengths.data).item()
               output_lengths = to_gpu(torch.LongTensor([mel.shape[1]])).long()
               mel = to_gpu(mel.unsqueeze(0)).float()
               
               x = (text, input_lengths, mel, max_len, output_lengths)
               
               mel_outputs, _, _, _ = model(x)
          
               np.save(save_mel_folder + "/" + file_name, mel_outputs.cpu().detach().numpy())




if __name__ == "__main__":
     training_files = sys.argv[1]
     checkpoint_path = sys.argv[2]
     save_mel_folder = sys.argv[3]
     GTA_Synthesis(training_files, checkpoint_path, save_mel_folder)