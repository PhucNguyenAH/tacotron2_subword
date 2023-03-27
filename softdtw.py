import json
import os
import numpy as np 
import argparse
import multiprocessing as mp 
import sys
# from fastdtw import fastdtw
from soft_dtw_cuda import SoftDTW
from glob import glob
# from preprocessing import WORLD_processing
import librosa
# from WORLD_processing import *
import scipy.spatial
import pyworld
import shutil
import torch
import math
from tqdm import tqdm
import Audio


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
    sdtw = SoftDTW(use_cuda=True, gamma=0.1)
    # read source features , target features and converted mcc
    src_data = torch.from_numpy(np.transpose(Audio.tools.get_mel(file_path1).numpy().astype(np.float32))).to("cuda").unsqueeze(0)
    trg_data = torch.from_numpy(np.transpose(Audio.tools.get_mel(file_path2).numpy().astype(np.float32))).to("cuda").unsqueeze(0)

    loss = sdtw(src_data, trg_data)
    if math.isinf(loss.mean().item()) and loss.mean().item() > 0:
        return None
    return loss.mean().item()

if __name__ =='__main__':
    sdtw_test = []

    Test_infer = glob("benchmark/*.wav")

    for infer in tqdm(Test_infer):
        groundtruth = infer.replace("benchmark","data/vi_dataset/wav")
        sdtw = evaluate_mcd_wav(groundtruth,infer)
        if sdtw is not None:
            sdtw_test.append(float(sdtw))
    print("Process Soft DTW for GroundTruth and testset")
    print(sum(sdtw_test)/len(sdtw_test))