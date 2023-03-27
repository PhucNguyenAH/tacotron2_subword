import numpy as np
from scipy.io.wavfile import read
import torch
import matplotlib
import matplotlib.pylab as plt
from librosa.util import normalize
from librosa.filters import mel as librosa_mel_fn
import torch.nn.functional as F

def get_mask_from_lengths(lengths):
    max_len = torch.max(lengths).item()
    ids = torch.arange(0, max_len, out=torch.cuda.LongTensor(max_len))
    mask = (ids < lengths.unsqueeze(1)).bool()
    return mask


def load_wav_to_torch(full_path):
    sampling_rate, data = read(full_path)
    return torch.FloatTensor(data.astype(np.float32)), sampling_rate


def load_filepaths_and_text(filename, split="|"):
    with open(filename, encoding='utf-8') as f:
        filepaths_and_text = [line.strip().split(split) for line in f]
    return filepaths_and_text


def to_gpu(x):
    x = x.contiguous()

    if torch.cuda.is_available():
        x = x.cuda(non_blocking=True)
    return torch.autograd.Variable(x)

mel_basis = {}
hann_window = {}

def load_wav(full_path):
    sampling_rate, data = read(full_path)
    return data, sampling_rate
def dynamic_range_compression(x, C=1, clip_val=1e-5):
    return np.log(np.clip(x, a_min=clip_val, a_max=None) * C)
def dynamic_range_decompression(x, C=1):
    return np.exp(x) / C
def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    return torch.log(torch.clamp(x, min=clip_val) * C)
def dynamic_range_decompression_torch(x, C=1):
    return torch.exp(x) / C
def spectral_normalize_torch(magnitudes):
    output = dynamic_range_compression_torch(magnitudes)
    return output
def spectral_de_normalize_torch(magnitudes):
    output = dynamic_range_decompression_torch(magnitudes)
    return output
def mel_spectrogram(filename, n_fft=1024, num_mels=80, sampling_rate=22050, hop_size=256, win_size=1024, fmin=0.0, fmax=8000.0, center=False, stftTaco=False, stftCUDA=False):
     
     audio, sampling_rate = load_wav(filename)
     MAX_WAV_VALUE = 32768.0
     audio = audio / MAX_WAV_VALUE
     audio = normalize(audio) * 0.95
     audio = torch.FloatTensor(audio)
     y = audio.unsqueeze(0)
     
     if torch.min(y) < -1.:
          print('min value is ', torch.min(y))
     if torch.max(y) > 1.:
          print('max value is ', torch.max(y))
     global mel_basis, hann_window
     if fmax not in mel_basis:
          mel = librosa_mel_fn(sampling_rate, n_fft, num_mels, fmin, fmax)
          mel_basis[str(fmax)+'_'+str(y.device)] = torch.from_numpy(mel).float().to(y.device)
          hann_window[str(y.device)] = torch.hann_window(win_size).to(y.device)
     y = torch.nn.functional.pad(y.unsqueeze(1), (int((n_fft-hop_size)/2), int((n_fft-hop_size)/2)), mode='reflect')
     y = y.squeeze(1)
     spec = torch.stft(y, n_fft, hop_length=hop_size, win_length=win_size, window=hann_window[str(y.device)],
                    center=center, pad_mode='reflect', normalized=False, onesided=True)
     spec = torch.sqrt(spec.pow(2).sum(-1)+(1e-9))
     spec = torch.matmul(mel_basis[str(fmax)+'_'+str(y.device)], spec)
     spec = spectral_normalize_torch(spec)
     return spec

def plot_spectrogram(spectrogram, filename):
     fig, ax = plt.subplots(figsize=(12, 3))
     im = ax.imshow(spectrogram, aspect="auto", origin="lower",
                    interpolation='none')
     plt.colorbar(im, ax=ax)
     plt.xlabel("Frames")
     plt.ylabel("Channels")
     plt.tight_layout()
     plt.savefig(filename)

class Alignment_Generator(torch.nn.Module):
    def __init__(self):
        super(Alignment_Generator, self).__init__()

    def LR(self, duration_predictor_output):
        frame_lens = torch.sum(duration_predictor_output, -1)
        expand_max_frame_len = torch.max(frame_lens, -1)[0]
        alignment = torch.zeros(duration_predictor_output.size(0), expand_max_frame_len, duration_predictor_output.size(1))
        alignment = create_alignment(alignment, duration_predictor_output)
        # plot_spectrogram(alignment.cpu().detach().numpy().T, "plot_3.png")
        return alignment

    def forward(self, duration_predictor_output):
        output = self.LR(duration_predictor_output)
        return output

def create_alignment(base_mat, duration_predictor_output):
    height, width = duration_predictor_output.shape
    for i in range(height):
        count = 0
        for j in range(width):
            o: int = int(duration_predictor_output[i][j].item())
            for k in range(o):
                base_mat[i][count + k][j] = 1
            count = count + o
    return base_mat

if __name__ == "__main__":

    # duration_predictor_output = np.load("/data/nhandt23/Workspace/hifigan/S_F010005VN11_NewsSocial_001_02.npy").astype(int)
    # duration_predictor_output = torch.from_numpy(duration_predictor_output)[:,1]
    # duration_predictor_output = torch.unsqueeze(duration_predictor_output, 0)
    # length_regulator = Alignment_Generator()
    # output = length_regulator(duration_predictor_output)

    kl = torch.nn.KLDivLoss(size_average=False, reduction='sum')
    output = torch.from_numpy(np.array([[0.1, 0.5, 0.4,0.2, 0.4, 0.4]])).float()
    target = torch.from_numpy(np.array([[0.6, 0.3, 0.1,0.00001, 0.5, 0.4]])).float()
    loss = kl(torch.log(target), output)
    print(loss)


    a=target
    b=output
    print( torch.sum(a*(torch.log(a)-torch.log(b)), dim=-1 ))

    P=b
    Q=a
    print((P * (P / Q).log()).sum())

    print(F.kl_div(Q.log(), P, None, None, 'sum'))