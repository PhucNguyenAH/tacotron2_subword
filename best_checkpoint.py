from scipy.io.wavfile import write
import IPython.display as ipd
from g2p.text_to_sequence import Text2Seq

import sys
import numpy as np
import pandas as pd
import torch
import json
import librosa
from glob import glob
from pydub import AudioSegment
from unicodedata import normalize

from hparams import create_hparams
from model import BERT_Tacotron2
from layers import TacotronSTFT, STFT
from audio_processing import griffin_lim
from soft_dtw_cuda import SoftDTW

from train import load_model
import os
from tqdm import tqdm
from hifigan_infer.hifigan_model import Generator
from hifigan_infer.hifigan_utils import load_checkpoint 
from bias_remover import hifiganBiasRemover, waveglowBiasRemover
import multiprocessing as mp
from fastdtw import fastdtw
import scipy.spatial
import pyworld
import shutil
import onnxruntime
import onnx
from data_utils import get_embedding, get_embedding_cls
from transformers import BertTokenizer, BertModel
from tokenizers import Tokenizer
import matplotlib.pylab as plt
import math
import Audio

HIFI_MODEL = None
TACO_MODEL = None

hifi_chkpt = "Outdir/SF02_2.2.0_finetune/g_01900000"
BENCHMNARK = "./benchmark_KL"
INFERENCE = "./wav_infer_KL"
LOGGING = "./logging_KL"

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

    # config_302_be = {
    #     'g2p_model_path': 'g2p_v3_2/05_vn_g2p_model.fst',
    #     'g2p_config': 'g2p_v3_2/config_phonetisaurus.vn.v3.south.20pau.yml',
    #     'phone_id_list_file': 'g2p_v3_2/phone_id-v3.0.1.map.merge_all.20pau',
    #     'delimiter': None,
    #     'ignore_white_space': True,
    #     'padding': True,
    #     'device':'cpu'
    # }

    config_302_be = {
        'g2p_model_path': 'g2p_v3_3/05_vn_g2p_model.fst',
        'g2p_config': 'g2p_v3_3/config_phonetisaurus.vn.v3.north.20pau.yml',
        'phone_id_list_file': 'g2p_v3_3/phone_id-v3.0.1.map.merge_all.20pau',
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

def test_g2p(script_file="train.txt"):
    texts = []
    with open(script_file, "r", encoding="utf-8") as f:
        lines = f.read().splitlines()
        for line in lines:
            p, s = line.split("|")
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
    try:
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

        write(output, 22050, wav_taco_hifi)
    except:
        pass

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

def tacotron2_infer(sequence, embeddings, phoneme_embeddings_cls, bert_embeddings_cls, checkpoint_path, filename="test"):
    global TACO_MODEL
    ckp_num = checkpoint_path.split("_")[-1]
    token_vocab = checkpoint_path.split("/")[-2].split("_")[-1]
    
    if TACO_MODEL==None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        hparams = create_hparams()
        hparams.max_decoder_steps = 6000
        hparams.sub_n_symbols = int(token_vocab)
        # checkpoint_path = latest_checkpoint_path(path)
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

    mel_outputs, mel_outputs_postnet, _, alignments, alignments_bert, INFER_FLAG = model.inference(sequence, embeddings, phoneme_embeddings_cls, bert_embeddings_cls)
    
    plot_spectrogram(alignments[0].cpu().detach().numpy().T, f"Outdir/demo_KL_{token_vocab}_{ckp_num}/alignment/"+filename+".png" )
    plot_spectrogram(alignments_bert[0].cpu().detach().numpy().T, f"Outdir/demo_KL_{token_vocab}_{ckp_num}/alignment_bert/"+filename+".png" )
    plot_spectrogram(mel_outputs[0].cpu().detach().numpy(), f"Outdir/demo_KL_{token_vocab}_{ckp_num}/mels/"+filename+".png")

    return mel_outputs, mel_outputs_postnet, model, INFER_FLAG

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

def detect_leading_silence(sound, silence_threshold=-50.0, chunk_size=10):
    '''
    sound is a pydub.AudioSegment
    silence_threshold in dB
    chunk_size in ms
    iterate over chunks until you find the first one with sound
    '''
    trim_ms = 0  # ms
    while sound[trim_ms:trim_ms+chunk_size].dBFS < silence_threshold:
        trim_ms += chunk_size

        if trim_ms > len(sound):
            return None

    return trim_ms

def load_wavs(wav_dir, sr):
    wavs = list()
    for file in os.listdir(wav_dir):
        file_path = os.path.join(wav_dir, file)
        wav, _ = librosa.load(file_path, sr=sr, mono=True)
        # wav = wav.astype(np.float64)
        wavs.append(wav)
    return wavs

def world_encode_spectral_envelop(sp, fs, dim=24):
    # Get Mel-Cepstral coefficients (MCEPs)
    sp = sp.astype(np.float64)
    coded_sp = pyworld.code_spectral_envelope(sp, fs, dim)
    return coded_sp

def world_decompose(wav, fs, frame_period=5.0):
    # Decompose speech signal into f0, spectral envelope and aperiodicity using WORLD
    wav = wav.astype(np.float64)

    f0, timeaxis = pyworld.harvest( wav, fs, frame_period=frame_period, f0_floor=71.0, f0_ceil=800.0)

    # Finding Spectogram
    sp = pyworld.cheaptrick(wav, f0, timeaxis, fs)

    # Finding aperiodicity
    ap = pyworld.d4c(wav, f0, timeaxis, fs)
    return f0, timeaxis, sp, ap

def world_encode_data(wave, fs, frame_period=5.0, coded_dim=24):
    f0s = list()
    timeaxes = list()
    sps = list()
    aps = list()
    coded_sps = list()
    for wav in wave:
        f0, timeaxis, sp, ap = world_decompose(wav=wav,
                                               fs=fs,
                                               frame_period=frame_period)
        coded_sp = world_encode_spectral_envelop(sp=sp, fs=fs, dim=coded_dim)
        f0s.append(f0)
        timeaxes.append(timeaxis)
        sps.append(sp)
        aps.append(ap)
        coded_sps.append(coded_sp)
    return f0s, timeaxes, sps, aps, coded_sps

def get_feature(wav, fs=16000):
    f0, timeaxis, sp, ap, mc = world_encode_data(wav, fs)
    return f0, mc


def evaluate_mcd_wav(file_path1, file_path2):
    try:
        # read source features , target features and converted mcc
        src_data,_ = librosa.load(file_path1, sr=16000)
        trg_data,_ = librosa.load(file_path2, sr=16000)

        src_data = np.expand_dims(src_data, axis=0)
        trg_data = np.expand_dims(trg_data, axis=0)

        src_f0, src_mcc = get_feature(src_data)
        trg_f0, trg_mcc = get_feature(trg_data)

        src_f0, src_mcc = src_f0[0], src_mcc[0]
        trg_f0, trg_mcc = trg_f0[0], trg_mcc[0]
        # non-silence parts
        trg_idx = np.where(trg_f0>0)[0]
        # print('trg idx: ', trg_idx)
        trg_mcc = trg_mcc[trg_idx,:24]
        # print('trg_mcc shape: ', trg_mcc.shape)
        src_idx = np.where(src_f0>0)[0]
        src_mcc = src_mcc[src_idx,:24]
        # DTW
        _, path = fastdtw(src_mcc, trg_mcc, dist=scipy.spatial.distance.euclidean)
        twf = np.array(path).T
        cvt_mcc_dtw = src_mcc[twf[0]]
        trg_mcc_dtw = trg_mcc[twf[1]]
        # MCD 
        diff2sum = np.sum((cvt_mcc_dtw - trg_mcc_dtw)**2, 1)
        mcd = np.mean(10.0 / np.log(10.0) * np.sqrt(2 * diff2sum), 0)
        # logging.info('{} {}'.format(basename, mcd))
        # print('utterance mcd: {}'.format(mcd))

        return mcd
    except:
        return None

def evaluate_sdtw_wav(file_path1, file_path2):
    try:
        sdtw = SoftDTW(use_cuda=True, gamma=0.1)
        # read source features , target features and converted mcc
        src_data = torch.from_numpy(np.transpose(Audio.tools.get_mel(file_path1).numpy().astype(np.float32))).to("cuda").unsqueeze(0)
        trg_data = torch.from_numpy(np.transpose(Audio.tools.get_mel(file_path2).numpy().astype(np.float32))).to("cuda").unsqueeze(0)

        loss = sdtw(src_data, trg_data)
        
        return loss.mean().item()
    except:
        return None


if __name__ == '__main__':
    token_vocab = sys.argv[1]
    LIMIT_SILENCE = 3000
    print("token_vocab:",token_vocab)
    model_bert = BertModel.from_pretrained('bert-base-multilingual-cased')
    tokenizer_custom = Tokenizer.from_file(f"data/vibert_{token_vocab}.json")
    tokenizer_bert = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
    checkpoints = sorted(glob(f"./checkpoints_KL_{token_vocab}/checkpoint_*000"))
    if os.path.exists(LOGGING + f"_{token_vocab}" + ".csv"):
        df = pd.read_csv(LOGGING + f"_{token_vocab}" + ".csv", index_col=False)
        if "start_silence" not in df.head():
            df["start_silence"] = np.nan
        if "limit_silence" not in df.head():
            df["limit_silence"] = np.nan
        if "error" not in df.head():
            df["error"] = np.nan
        if "Soft_DTW" not in df.head():
            df["Soft_DTW"] = np.nan
    else:
        df = pd.DataFrame(columns = ['Checkpoints', 'MCD', 'Soft_DTW', "start_silence", "limit_silence", "error"])

    for ckp in checkpoints:
        TACO_MODEL = None
        START_STRIM = 0
        LIMIT_STRIM = 0
        INFER_FLAG = True
        ckp_num = ckp.split("_")[-1]
        for output_directory in [f"Outdir/demo_KL_{token_vocab}_{ckp_num}/alignment", f"Outdir/demo_KL_{token_vocab}_{ckp_num}/alignment_bert",f"Outdir/demo_KL_{token_vocab}_{ckp_num}/mels"]:
            if not os.path.isdir(output_directory):
                os.makedirs(output_directory, exist_ok=True)
                os.chmod(output_directory, 0o775)
        
        print("Checkpoint:", ckp)
        # Inference
        print("Inference")
        infer_save = INFERENCE + "_" + token_vocab + "_" + ckp_num
        if not os.path.exists(infer_save):
            os.makedirs(infer_save)
        with open("data/vi_dataset/script/val.txt", "r", encoding="utf-8") as f:
            lines = f.read().splitlines()
            for line in tqdm(lines):
                torch.cuda.empty_cache()
                wav, text = line.split("\t")
                wavpath = wav.split("/")[-1] + ".wav"
                if os.path.exists(os.path.join(infer_save, wavpath)):
                    continue
                text = normalize("NFC", text).lower()
                sequence = text2sequence(text).to("cuda")
                embeddings = torch.stack([get_embedding(text, model_bert, tokenizer_custom)]).to("cuda")
                phoneme_embeddings_cls = torch.stack([get_embedding_cls(text, model_bert, tokenizer_bert).repeat(sequence.size(1),1)])
                bert_embeddings_cls = torch.stack([get_embedding_cls(text, model_bert, tokenizer_bert).repeat(embeddings.size(1),1)])
                mel_outputs,_,_, INFER_FLAG = tacotron2_infer(sequence.to("cuda"), embeddings.to("cuda"), phoneme_embeddings_cls.to("cuda"), bert_embeddings_cls.to("cuda"), ckp, filename=wav)
                if not INFER_FLAG:
                    break
                hifigan_infer(mel_outputs, hifi_chkpt, output=os.path.join(infer_save, wavpath), bias_remove = "hifigan")

        if not INFER_FLAG:
            continue

        # Remove silence
        print("Remove silence")
        bench_save = BENCHMNARK + "_" + token_vocab + "_" + ckp_num
        if not os.path.exists(bench_save):
            os.makedirs(bench_save)
        inferences = glob(f"./wav_infer_KL_{token_vocab}_{ckp_num}/*.wav")
        for infer in tqdm(inferences):
            try:
                save_path = infer.replace("wav_infer","benchmark")
                sound = AudioSegment.from_file(infer, format="wav")

                start_trim = detect_leading_silence(sound)
                if start_trim is None:
                    continue
                START_STRIM += start_trim
                if start_trim >= LIMIT_SILENCE:
                    LIMIT_STRIM += 1
                end_trim = detect_leading_silence(sound.reverse())

                duration = len(sound)
                trimmed_sound = sound[start_trim:duration-end_trim]
                trimmed_sound.export(save_path, format="wav")
            except:
                continue

        infer_error = 1000 - len(glob(f"benchmark_KL_{token_vocab}_{ckp_num}/*.wav"))

        # MCD, Soft DTW
        if ckp.split("/")[-1] not in df['Checkpoints'].values:
            print("Calculate MCD, Soft DTW")
            mcd_test = []
            sdtw_test = []
            sdtw_flag = True

            test_infer = glob(f"benchmark_KL_{token_vocab}_{ckp_num}/*.wav")

            for infer in tqdm(test_infer):
                torch.cuda.empty_cache()
                groundtruth = infer.replace(f"benchmark_KL_{token_vocab}_{ckp_num}","data/vi_dataset/wav")
                ## MCD
                mcd = evaluate_mcd_wav(infer,groundtruth)
                if mcd is not None:
                    mcd_test.append(float(mcd))

                ## Soft DTW
                if sdtw_flag:
                    sdtw = evaluate_sdtw_wav(infer,groundtruth)
                    if sdtw is not None:
                        if math.isinf(sdtw):
                            sdtw_flag = False
                        sdtw_test.append(float(sdtw))
            print("Process MCD for GroundTruth and testset")
            print(sum(mcd_test)/len(mcd_test))

            print("Process Soft DTW for GroundTruth and testset")
            print(sum(sdtw_test)/len(sdtw_test))
            new_log = {'Checkpoints': ckp.split("/")[-1], 'MCD': sum(mcd_test)/len(mcd_test), 'Soft_DTW': sum(sdtw_test)/len(sdtw_test), 
                       "start_silence": START_STRIM, "limit_silence": LIMIT_STRIM, "error": infer_error}

            df = df.append(new_log, ignore_index = True)
            print("Logging:",df)
            df.to_csv(LOGGING + f"_{token_vocab}" + ".csv", index=False)

        if math.isnan(df.loc[df["Checkpoints"] == ckp.split("/")[-1], "Soft_DTW"].values[0]):
            # Soft DTW
            print("Calculate Soft DTW")
            sdtw_test = []
            sdtw_flag = True

            test_infer = glob(f"benchmark_{ckp_num}/*.wav")

            for infer in tqdm(test_infer):
                torch.cuda.empty_cache()
                groundtruth = infer.replace(f"benchmark_KL_{ckp_num}","data/vi_dataset/wav")

                ## Soft DTW
                if sdtw_flag:
                    sdtw = evaluate_sdtw_wav(infer,groundtruth)
                    if sdtw is not None:
                        if math.isinf(sdtw):
                            sdtw_flag = False
                        sdtw_test.append(float(sdtw))

            print("Process Soft DTW for GroundTruth and testset")
            print(sum(sdtw_test)/len(sdtw_test))
            df.loc[df["Checkpoints"] == ckp.split("/")[-1], "Soft_DTW"] = sum(sdtw_test)/len(sdtw_test)
            print("Logging:",df)
            df.to_csv(LOGGING + f"_{token_vocab}" + ".csv", index=False)

        if math.isnan(df.loc[df["Checkpoints"] == ckp.split("/")[-1], "start_silence"].values[0]):
            df.loc[df["Checkpoints"] == ckp.split("/")[-1], "start_silence"] = START_STRIM
            print("Logging:",df)
            df.to_csv(LOGGING + f"_{token_vocab}" + ".csv", index=False)

        if math.isnan(df.loc[df["Checkpoints"] == ckp.split("/")[-1], "limit_silence"].values[0]):
            df.loc[df["Checkpoints"] == ckp.split("/")[-1], "limit_silence"] = LIMIT_STRIM
            print("Logging:",df)
            df.to_csv(LOGGING + f"_{token_vocab}" + ".csv", index=False)
        
        if math.isnan(df.loc[df["Checkpoints"] == ckp.split("/")[-1], "error"].values[0]):
            df.loc[df["Checkpoints"] == ckp.split("/")[-1], "error"] = infer_error
            print("Logging:",df)
            df.to_csv(LOGGING + f"_{token_vocab}" + ".csv", index=False)