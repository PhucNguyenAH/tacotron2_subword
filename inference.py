from g2p.text_to_sequence import Text2Seq
import torch, json, glob
import librosa, sys, os
import numpy as np
from scipy.io.wavfile import write
import soundfile
from hparams import create_hparams
from model import BERT_Tacotron2
from hifigan_infer.hifigan_model import Generator
from hifigan_infer.hifigan_utils import load_checkpoint 
import matplotlib
import matplotlib.pylab as plt
from tqdm import tqdm
from bias_remover import hifiganBiasRemover, waveglowBiasRemover
import onnxruntime
import onnx
from data_utils import get_embedding, get_embedding_cls
from transformers import BertTokenizer, BertModel
from tokenizers import Tokenizer
from unicodedata import normalize

TACO_MODEL = None
HIFI_MODEL = None

class AttrDict(dict):
     def __init__(self, *args, **kwargs):
          super(AttrDict, self).__init__(*args, **kwargs)
          self.__dict__ = self

def plot_spectrogram(spectrogram, filename):
     fig, ax = plt.subplots(figsize=(12, 3))
     im = ax.imshow(spectrogram, aspect="auto", origin="lower",
                    interpolation='none')
     plt.colorbar(im, ax=ax)
     plt.xlabel("Frames")
     plt.ylabel("Channels")
     plt.tight_layout()
     plt.savefig(filename)
     plt.close()
     
#     fig.canvas.draw()
#     data = save_figure_to_numpy(fig)
#     plt.close()
#     return data

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

     # config_302_be = {
     #      'g2p_model_path': 'g2p_resources-v2.2.5/13k_foreign_checked_011121.multi_pronunciation.g2p_model.fst',
     #      'g2p_config': 'g2p_resources-v2.2.5/config_phonetisaurus_v1.0.3_map3.0.0.yml',
     #      'phone_id_list_file': 'g2p_resources-v2.2.5/Phone_ID_Map.v3.0.0/vn_xsampa_phoneID_map_v3.0.0_011221.merge_BE',
     #      'delimiter': None,
     #      'ignore_white_space': True,
     #      'padding': True,
     #      'device':'cpu'
     # }

     # config_302_be = {
     #      'g2p_model_path': 'g2p_v3/05_vn_g2p_model.fst',
     #      'g2p_config': 'g2p_v3/config_phonetisaurus.vn.v3.south.yml',
     #      'phone_id_list_file': 'g2p_v3/Phone_ID_Map.v3.0.0/vn_xsampa_phoneID_map_v3.0.0_011221.merge_BE',
     #      'delimiter': None,
     #      'ignore_white_space': True,
     #      'padding': True,
     #      'device':'cpu'
     # }
     
     config_302_be = {
          'g2p_model_path': 'g2p_v3_2/05_vn_g2p_model.fst',
          'g2p_config': 'g2p_v3_2/config_phonetisaurus.vn.v3.south.20pau.yml',
          'phone_id_list_file': 'g2p_v3_2/phone_id-v3.0.1.map.merge_all.20pau',
          'delimiter': None,
          'ignore_white_space': True,
          'padding': True,
          'device':'cpu'
     }

     # config_302_be = {
     #      'g2p_model_path': 'g2p_v3_3/05_vn_g2p_model.fst',
     #      'g2p_config': 'g2p_v3_3/config_phonetisaurus.vn.v3.north.20pau.yml',
     #      'phone_id_list_file': 'g2p_v3_3/phone_id-v3.0.1.map.merge_all.20pau',
     #      'delimiter': None,
     #      'ignore_white_space': True,
     #      'padding': True,
     #      'device':'cpu'
     # }

     config = config_302_be

     text2seq = Text2Seq(config['g2p_model_path'], 
                         g2p_config=config['g2p_config'], 
                         phone_id_list_file=config['phone_id_list_file'], 
                         delimiter=config['delimiter'],
                         ignore_white_space=config['ignore_white_space']
                         )
     
     sequence, src_pos = prepare_data(text2seq, text, padding=config['padding'], device=config['device'])

     return sequence

def test_g2p(script_file="train.txt"):
     texts = []
     with open(script_file, "r", encoding="utf-8") as f:
          lines = f.read().splitlines()
          for line in lines:
               p, s = line.split("\t")
               texts.append(s)     
     for text in texts:
          sequence = text2sequence(text)

def waveglow_infer(mel_outputs_postnet, path='Outdir/waveglow_256channels.pt', output="test"):
     # Load Waveglow
     waveglow_path = path
     waveglow = torch.load(waveglow_path)['model']
     waveglow.cuda().eval()

     with torch.no_grad():
          audio = waveglow.infer(mel_outputs_postnet, sigma=0.666)

     wav_taco_waveglow = audio[0].data.cpu().numpy()
     write("Outdir/demo/audio/"+output+".wav", 22050, wav_taco_waveglow )

def hifigan_infer(mel_outputs, path, output="test", vocoder_config_path="hifigan_infer/config_v1.json", bias_remove=""):
     # Load HifiGAN
     global HIFI_MODEL

     if HIFI_MODEL == None:
          device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
          with open(vocoder_config_path) as f:
               data = f.read()
          json_config = json.loads(data)
          h = AttrDict(json_config)
          torch.manual_seed(h.seed)

          generator = Generator(h).to(device)
          state_dict_g = load_checkpoint(path, device)
          generator.load_state_dict(state_dict_g['generator'])
          generator.eval()
          generator.remove_weight_norm()
     else:
          generator = HIFI_MODEL

     with torch.no_grad():
          y_g_hat = generator(mel_outputs)
          audio = y_g_hat.squeeze()

          MAX_WAV_VALUE = 32768.0 * 1.7
          audio = audio * MAX_WAV_VALUE 

     if bias_remove == "waveglow":
          audio = waveglowBiasRemover('Outdir/waveglow_256channels.pt').cuda()(audio.unsqueeze(0), strength=0.01)
     elif bias_remove == "hifigan":
          audio = hifiganBiasRemover(generator, mode ='zeros').cuda()(audio.unsqueeze(0), 0.9)

     wav_taco_hifi = audio.cpu().detach().numpy().astype('int16')

     write("Outdir/demo/audio/"+output+".wav", 22050, wav_taco_hifi)

def hifigan_onnx_infer(mel_outputs, path, config_file = "NF01_3.1.0_fromWav_v3_8000/config.json"):
     sess_options = onnxruntime.SessionOptions()
     model_hifi = onnxruntime.InferenceSession(path)
     
     with open(config_file) as f:
          data = f.read()
     json_config = json.loads(data)
     h = AttrDict(json_config)

     mel = mel_outputs.detach().cpu().numpy()

     audio = model_hifi.run(None, {model_hifi.get_inputs()[0].name: mel})
     audio = np.asarray(audio)
     write("Outdir/demo/onnx_audio.wav", 22050, audio)

def hifigan_tflite_infer(mel_outputs, path):

     import tensorflow as tf

     mel = mel_outputs.detach().cpu().numpy()
     
     interpreter = tf.lite.Interpreter(model_path=path)
     input_details = interpreter.get_input_details()
     output_details = interpreter.get_output_details()
     interpreter.resize_tensor_input(input_details[0]['index'],  [1, mel.shape[1], mel.shape[2]], strict=True)
     interpreter.allocate_tensors()
     interpreter.set_tensor(input_details[0]['index'], mel)
     interpreter.invoke()
     audio = interpreter.get_tensor(output_details[0]['index'])
     
     write("Outdir/demo/tflite_audio.wav", 22050, audio)

def tacotron2_infer(sequence, embeddings, phoneme_embeddings_cls, bert_embeddings_cls, path, filename="test"):
     global TACO_MODEL

     if TACO_MODEL==None:
          device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
          hparams = create_hparams()
          hparams.max_decoder_steps = 6000
          checkpoint_path = latest_checkpoint_path(path)
          print("Load: ", checkpoint_path)
          model = BERT_Tacotron2(hparams).cuda()

          print(torch.cuda.is_available())

          if torch.cuda.is_available():
               model.load_state_dict(torch.load(checkpoint_path)['state_dict'])
               _ = model.to(device).eval()
          else:
               model.load_state_dict(torch.load(checkpoint_path, map_location='cpu')['state_dict'])
               _ = model.to(device).eval()   
          TACO_MODEL = model
     else:
          model = TACO_MODEL

     mel_outputs, mel_outputs_postnet, _, alignments, alignments_bert, _ = model.inference(sequence, embeddings, phoneme_embeddings_cls, bert_embeddings_cls)

     plot_spectrogram(alignments[0].cpu().detach().numpy().T, "Outdir/demo/alignment/"+filename+".png" )
     plot_spectrogram(alignments_bert[0].cpu().detach().numpy().T, "Outdir/demo/alignment_bert/"+filename+".png" )
     plot_spectrogram(mel_outputs[0].cpu().detach().numpy(), "Outdir/demo/mels/"+filename+".png")

     return mel_outputs, mel_outputs_postnet, model

def sequence2text(sequence):
     
     with open("g2p_resources-v2.2.5/Phone_ID_Map.v3.0.0/vn_xsampa_phoneID_map_v3.0.0_011221.merge_BE", "r", encoding="utf-8") as f:
          lines = f.read().splitlines()
          phones = {}
          for line in lines:
               phone, ids = line.split("\t")
               phones[int(ids)] = phone
     text = ""
     for seq in sequence.cpu().detach().numpy()[0]:
          text = text + phones[int(seq)].replace("_S","_B") + str(int(seq)) +" | " #+ str(seq) + "\n"
     return text

def latest_checkpoint_path(dir_path, regex="checkpoint_*000"):
     try:
          f_list = glob.glob(os.path.join(dir_path, regex))
          f_list.sort(key=lambda f: int("".join(filter(str.isdigit, f))))
          x = f_list[-1]
          print(x)
          return x
     except:
          return dir_path

def taco_hifi(texts, taco_path, hifi_path):

     # Load Tacotron 2 Model
     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
     hparams = create_hparams()
     hparams.max_decoder_steps = 2000
     checkpoint_path = latest_checkpoint_path(taco_path)
     print("Load: ", checkpoint_path)
     model = Tacotron2(hparams).cuda()

     print(torch.cuda.is_available())

     if torch.cuda.is_available():
          model.load_state_dict(torch.load(checkpoint_path)['state_dict'])
          _ = model.to(device).eval()
     else:
          model.load_state_dict(torch.load(checkpoint_path, map_location='cpu')['state_dict'])
          _ = model.to(device).eval()   
     ##############################################################
     # Load Hifigan
     MAX_WAV_VALUE = 32768.0
     vocoder_config_path = "/".join(hifi_path.split("/")[:-1]) +"/config.json"
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

     for line in tqdm(texts):
          path, text = line.split("\t")
          text = text.lower()
          print(text)
          sequence = text2sequence(text).to(device)
          print(sequence2text(sequence))
          mel_outputs, mel_outputs_postnet, _, alignments = model.inference(sequence)

          with torch.no_grad():
               y_g_hat = generator(mel_outputs)
               audio = y_g_hat.squeeze() * MAX_WAV_VALUE
          wav_taco_hifi = audio.cpu().detach().numpy().astype('int16')
          write("Outdir/demo/audio/test.wav", 22050, wav_taco_hifi)

if __name__ == "__main__":

     token_vocab = sys.argv[1]
     infer_script = "Outdir/demo/text_khanhlinh.norm"
     taco_chkpt = f"Outdir/ssim/checkpoints_{token_vocab}/checkpoint_best"
     hifi_chkpt = "Outdir/SF02_2.2.0_finetune/g_01900000"

     print("token_vocab:",token_vocab)
     
     model_bert = BertModel.from_pretrained('bert-base-multilingual-cased')
     tokenizer_custom = Tokenizer.from_file(f"data/vibert_{token_vocab}.json")
     tokenizer_bert = BertTokenizer.from_pretrained('bert-base-multilingual-cased')


     for output_directory in ["Outdir/demo/alignment","Outdir/demo/alignment_bert","Outdir/demo/mels","Outdir/demo/audio"]:
          if not os.path.isdir(output_directory):
               os.makedirs(output_directory, exist_ok=True)
               os.chmod(output_directory, 0o775)

     with open(infer_script, "r", encoding="utf-8") as f:
          lines = f.read().splitlines()
          for line in tqdm(lines[:]):
               path, text = line.split("|")
               if os.path.exists("Outdir/demo/audio/"+path+".wav"):
                    continue
               text = normalize("NFKC", text).lower()
               sequence = text2sequence(text).to("cuda")
               embeddings = torch.stack([get_embedding(text, model_bert, tokenizer_custom)]).to("cuda")
               phoneme_embeddings_cls = torch.stack([get_embedding_cls(text, model_bert, tokenizer_bert).repeat(sequence.size(1),1)])
               bert_embeddings_cls = torch.stack([get_embedding_cls(text, model_bert, tokenizer_bert).repeat(embeddings.size(1),1)])

               mel_outputs,_,_ = tacotron2_infer(sequence.to("cuda"), embeddings.to("cuda"), phoneme_embeddings_cls.to("cuda"), bert_embeddings_cls.to("cuda"), taco_chkpt, filename=path)

               hifigan_infer(mel_outputs, hifi_chkpt, output=path, bias_remove = "hifigan")
               