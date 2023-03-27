import os
import numpy as np

f = open("data/vi_dataset/script/train.txt")
data = f.readlines()
f.close()

for line in data:
    wav, text = line.strip().split("\t")
    wav_path = "data/vi_dataset/wav/" + wav + ".wav"
    duration_path = "data/vi_dataset/durations/" + wav + ".npy"
    if not os.path.exists(wav_path):
        print('wav_path:',wav_path)
    if not os.path.exists(duration_path):
        print("duration_path:",duration_path)

f = open("data/vi_dataset/script/val.txt")
data = f.readlines()
f.close()

for line in data:
    wav, text = line.strip().split("\t")
    wav_path = "data/vi_dataset/wav/" + wav + ".wav"
    duration_path = "data/vi_dataset/durations/" + wav + ".npy"
    if not os.path.exists(wav_path):
        print('wav_path:',wav_path)
    if not os.path.exists(duration_path):
        print("duration_path:",duration_path)
