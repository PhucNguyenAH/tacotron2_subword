import json
import os
import numpy as np 
import argparse
import multiprocessing as mp 
import sys
from fastdtw import fastdtw
from glob import glob
# from preprocessing import WORLD_processing
import librosa
# from WORLD_processing import *
import scipy.spatial
import pyworld
import shutil
from tqdm import tqdm

# https://github.com/pritishyuvraj/Voice-Conversion-GAN/blob/master/preprocess.py

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

if __name__ =='__main__':
    mcd_test = []

    Test_infer = glob("benchmark/*.wav")

    for infer in tqdm(Test_infer):
        groundtruth = infer.replace("benchmark","data/vi_dataset/wav")
        mcd = evaluate_mcd_wav(groundtruth,infer)
        if mcd is not None:
            mcd_test.append(float(mcd))
    print("Process MCD for GroundTruth and testset")
    print(sum(mcd_test)/len(mcd_test))