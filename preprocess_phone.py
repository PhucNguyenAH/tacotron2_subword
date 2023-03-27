import os
import numpy as np

if not os.path.exists("vi_dataset/preprocess/"):
    os.makedirs("vi_dataset/preprocess/")

f = open("vi_dataset/script/train.txt")
data = f.readlines()
f.close()

f_train = open("vi_dataset/preprocess/train.txt", "w")
for line in data:
    wav, text = line.strip().split("|")
    wav_path = "vi_dataset/wav/" + wav + ".wav"
    duration_path = "vi_dataset/durations/" + wav + ".npy"
    f_train.write(wav_path+"|"+duration_path+"\n")
f_train.close()

f = open("vi_dataset/script/val.txt")
data = f.readlines()
f.close()

f_train = open("vi_dataset/preprocess/val.txt", "w")
for line in data:
    wav, text = line.strip().split("|")
    wav_path = "vi_dataset/wav/" + wav + ".wav"
    duration_path = "vi_dataset/durations/" + wav + ".npy"
    f_train.write(wav_path+"|"+duration_path+"\n")
f_train.close()