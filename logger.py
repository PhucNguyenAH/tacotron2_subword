import random
import torch
from torch.utils.tensorboard import SummaryWriter
from plotting_utils import plot_alignment_to_numpy, plot_spectrogram_to_numpy
from plotting_utils import plot_gate_outputs_to_numpy

import json
from scipy.io.wavfile import write
from hifigan_infer.hifigan_model import Generator
from hifigan_infer.hifigan_utils import load_checkpoint 

class AttrDict(dict):
     def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self
class Tacotron2Logger(SummaryWriter):
    def __init__(self, logdir):
        super(Tacotron2Logger, self).__init__(logdir)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        vocoder_config_path = "hifigan_infer/config_v1.json"
        with open(vocoder_config_path) as f:
            data = f.read()
        json_config = json.loads(data)
        h = AttrDict(json_config)
        torch.manual_seed(h.seed)

        try:
            generator = Generator(h).to(self.device)
            state_dict_g = load_checkpoint("/data/nhandt23/Workspace/hifigan/Outdir/Universal/g_02080000", self.device)
            generator.load_state_dict(state_dict_g['generator'])
            generator.eval()
            generator.remove_weight_norm()
            self.generator = generator
        except:
            self.generator = None

    def log_training(self, reduced_loss, grad_norm, learning_rate, duration, iteration, mel_loss=None, gate_loss=None, align_loss=None, align_bert_loss=None):
            self.add_scalar("training.loss", reduced_loss, iteration)

            self.add_scalar("training.melloss", mel_loss, iteration)
            self.add_scalar("training.gateloss", gate_loss, iteration)
            self.add_scalar("training.alignloss", align_loss, iteration)
            self.add_scalar("training.alignbertloss", align_bert_loss, iteration)

            self.add_scalar("grad.norm", grad_norm, iteration)
            self.add_scalar("learning.rate", learning_rate, iteration)
            self.add_scalar("duration", duration, iteration)

    def log_validation(self, reduced_loss, model, y, y_pred, iteration):
        self.add_scalar("validation.loss", reduced_loss, iteration)
        _, mel_outputs, gate_outputs, alignments, alignments_bert = y_pred
        mel_targets, gate_targets, _ = y

        # plot distribution of parameters
        for tag, value in model.named_parameters():
            tag = tag.replace('.', '/')
            self.add_histogram(tag, value.data.cpu().numpy(), iteration)

        # plot alignment, mel target and predicted, gate target and predicted
        idx = random.randint(0, alignments.size(0) - 1)
        self.add_image(
            "alignment",
            plot_alignment_to_numpy(alignments[idx].data.cpu().numpy().T),
            iteration, dataformats='HWC')
        self.add_image(
            "alignment_bert",
            plot_alignment_to_numpy(alignments_bert[idx].data.cpu().numpy().T),
            iteration, dataformats='HWC')
        self.add_image(
            "mel_target",
            plot_spectrogram_to_numpy(mel_targets[idx].data.cpu().numpy()),
            iteration, dataformats='HWC')
        self.add_image(
            "mel_predicted",
            plot_spectrogram_to_numpy(mel_outputs[idx].data.cpu().numpy()),
            iteration, dataformats='HWC')
        self.add_image(
            "gate",
            plot_gate_outputs_to_numpy(
                gate_targets[idx].data.cpu().numpy(),
                torch.sigmoid(gate_outputs[idx]).data.cpu().numpy()),
            iteration, dataformats='HWC')
        
        if self.generator != None:
            with torch.no_grad():
                y_g_hat = self.generator(mel_outputs)
                audio = y_g_hat.squeeze()

                MAX_WAV_VALUE = 32768.0
                audio = audio * MAX_WAV_VALUE

            wav_taco_hifi = audio.cpu().detach().numpy()#.astype('int16')

            self.add_audio('inference', wav_taco_hifi[idx], iteration, 22050)
