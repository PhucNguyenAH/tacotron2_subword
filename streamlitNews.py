from g2p.text_to_sequence import Text2Seq
import torch, json, glob
import librosa, sys, os
import numpy as np
from scipy.io.wavfile import write
import soundfile
from hparams import create_hparams
from model import Tacotron2
from hifigan_infer.hifigan_model import Generator
from hifigan_infer.hifigan_utils import load_checkpoint 
import matplotlib
import matplotlib.pylab as plt
from tqdm import tqdm
from bias_remover import hifiganBiasRemover, waveglowBiasRemover
import onnxruntime
import onnx
import requests

############
import streamlit as st
from annotated_text import annotated_text

TACO_MODEL = None
HIFI_MODEL = None

class AttrDict(dict):
     def __init__(self, *args, **kwargs):
          super(AttrDict, self).__init__(*args, **kwargs)
          self.__dict__ = self

def prepare_data(text2seq, text, padding=True, device='cpu'):
     # Text Processing #
     sequence = text2seq.grapheme_to_sequence(text, padding)
     sequence = [int(seq) for seq in sequence]
     sequence = np.array(sequence)
     sequence = np.expand_dims(sequence, axis=0)
     src_pos = np.array([i + 1 for i in range(sequence.shape[1])])
     src_pos = np.stack([src_pos])
     sequence = torch.from_numpy(sequence).to(device).long()
     src_pos = torch.from_numpy(src_pos).to(device).long()
     return sequence, src_pos

def norm_sent(sent,api_norm):
     
     data = {"paragraph": sent} #norm tiếng Việt
     data = {"paragraph": sent, "lang":"vi"} #norm tiếng Anh
     result = requests.post(api_norm, auth=('speech_oov','4D6$&%9qeEhvRTeR'), json = data)
     normed_list = result.json().get("normed_sents")
     output = " ".join(normed_list)
     return output

def text2sequence(text):
     config_223_bies = {
          'g2p_model_path': 'g2p_resources-v2.2.3/vi_phonetisaurus-v1.0.0.fst',
          'g2p_config': 'g2p_resources-v2.2.3/config_phonetisaurus_v1.0.3.yml',
          'phone_id_list_file': 'g2p_resources-v2.2.3/phone_id-v2.2.0.map',
          'delimiter': None,
          'ignore_white_space': True,
          'padding': True,
          'device':'cpu'
     }
     config_224 = {
          'g2p_model_path': 'g2p_resources-v2.2.4/vi_phonetisaurus-v1.0.0.fst',
          'g2p_config': 'g2p_resources-v2.2.3/config_phonetisaurus_tacotron2_v1.0.2.yml',
          'phone_id_list_file': 'g2p_resources-v2.2.4/phone_id-v2.2.0.map',
          'delimiter': None,
          'ignore_white_space': True,
          'padding': True,
          'device':'cpu'
     }
     config_223_nobie = {
          'g2p_model_path': 'g2p_resources-v2.2.3/vi_phonetisaurus-v1.0.0.fst',
          'g2p_config': 'g2p_resources-v2.2.3/config_phonetisaurus_v1.0.3_taco.yml',
          'phone_id_list_file': 'g2p_resources-v2.2.3/phone_id-v2.2.0_taco.map',
          'delimiter': None,
          'ignore_white_space': False,
          'padding': True,
          'device':'cpu'
     }

     config_223_be = {
          'g2p_model_path': 'g2p_resources-v2.2.3/vi_phonetisaurus-v1.0.0.fst',
          'g2p_config': 'g2p_resources-v2.2.3/config_phonetisaurus_v1.0.3_be.yml',
          'phone_id_list_file': 'g2p_resources-v2.2.3/phone_id-v2.2.0_merge_B_I_S.map',
          'delimiter': None,
          'ignore_white_space': True,
          'padding': True,
          'device':'cpu'
     }

     config_302_be = {
          'g2p_model_path': 'g2p_resources-v2.2.5/13k_foreign_checked_011121.multi_pronunciation.g2p_model.fst',
          'g2p_config': 'g2p_resources-v2.2.5/config_phonetisaurus_v1.0.3_map3.0.0.yml',
          'phone_id_list_file': 'g2p_resources-v2.2.5/Phone_ID_Map.v3.0.0/vn_xsampa_phoneID_map_v3.0.0_011221.merge_BE',
          'delimiter': None,
          'ignore_white_space': True,
          'padding': True,
          'device':'cpu'
     }

     config = config_302_be

     text2seq = Text2Seq(config['g2p_model_path'], 
                         g2p_config=config['g2p_config'], 
                         phone_id_list_file=config['phone_id_list_file'], 
                         delimiter=config['delimiter'],
                         ignore_white_space=config['ignore_white_space']
                         )
     
     sequence, src_pos = prepare_data(text2seq, text, padding=config['padding'], device=config['device'])

     return sequence


def taco_hifi(texts, taco_path, hifi_path, bias_remove ="", vocoder_config_path="hifigan_infer/config_v1.json"):

     # Load Tacotron 2 Model
     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
     hparams = create_hparams()
     hparams.max_decoder_steps = 2000
     model = Tacotron2(hparams).cuda()

     print(torch.cuda.is_available())

     if torch.cuda.is_available():
          model.load_state_dict(torch.load(taco_path)['state_dict'])
          _ = model.to(device).eval()
     else:
          model.load_state_dict(torch.load(taco_path, map_location='cpu')['state_dict'])
          _ = model.to(device).eval()   
     ##############################################################
     # Load Hifigan
     MAX_WAV_VALUE = 32768.0 * 1.5
     with open(vocoder_config_path) as f:
          data = f.read()
     json_config = json.loads(data)
     h = AttrDict(json_config)
     torch.manual_seed(h.seed)
     generator = Generator(h).to(device)
     state_dict_g = load_checkpoint(hifi_path, device)
     generator.load_state_dict(state_dict_g['generator'])
     generator.eval()
     generator.remove_weight_norm()
     ##############################################################
     WAV = None

     for text in tqdm(texts):
          text = text.lower()
          sequence = text2sequence(text).to(device)
          mel_outputs, _, _, _ = model.inference(sequence)

          with torch.no_grad():
               y_g_hat = generator(mel_outputs)
               audio = y_g_hat.squeeze() * MAX_WAV_VALUE

          if bias_remove == "waveglow":
               audio = waveglowBiasRemover('Outdir/waveglow_256channels.pt').cuda()(audio.unsqueeze(0), strength=0.01)
          elif bias_remove == "hifigan":
               audio = hifiganBiasRemover(generator, mode ='zeros').cuda()(audio.unsqueeze(0), 1.0)

          audio, _ = librosa.effects.trim(audio.cpu().detach().numpy().astype('float32')[0][0])

          if WAV is None:
               WAV = audio
          else:
               WAV = np.concatenate( (WAV, audio ), axis=0)
     
     write("Outdir/news/test.wav", 22050, WAV.astype('int16'))

if __name__ == "__main__":

     # news = "Trong bộ ảnh lần này của Bẫy Ngọt Ngào, cặp đôi nhân vật Camy (Bảo Anh) và Đăng Minh (Quốc Trường) xuất hiện tình cảm bên nhau. Đây là thời điểm mà Camy và Đăng Minh còn đang nếm trải dư vị hạnh phúc khi vừa bước vào giai đoạn hôn nhân."
     
     news = st.text_area('News', '')
     
     texts = []
     api = "http://210.211.99.11:80/fasttts/api/internal_api/v2/tts_norm_ext" # server 11

     gen = st.button("Generate Audio")
     
     if gen:
          news_list = news.split(".")
          for text in news_list:
               text = norm_sent(text, api)
               print(text)
               texts.append(text)

          if len(sys.argv) >1:
               taco_chkpt = sys.argv[1]
               hifi_chkpt = sys.argv[2]
          else:
               taco_chkpt = "/home/nhandt23/Desktop/fasttts/tmp/01_hoaianh/06_tacotron/checkpoint_101000"
               hifi_chkpt = "/home/nhandt23/Desktop/fasttts/tmp/universal/Finetune/g_02000000"

          taco_hifi(texts, taco_chkpt, hifi_chkpt, bias_remove ="hifigan")

          for text in texts:
               annotated_text((text, "", "#8ef"))

          st.audio("Outdir/news/test.wav", format="audio/wav", start_time=0)