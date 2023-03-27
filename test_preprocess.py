import os
import torch
from text import text_to_sequence
from text.symbols import symbols
import numpy as np

_symbol_to_id = {s: i for i, s in enumerate(symbols)}

if not os.path.exists("vi_dataset/wav/"):
    os.makedirs("vi_dataset/preprocess/")

if not os.path.exists("vi_dataset/text/"):
    os.makedirs("vi_dataset/text/")

def _symbols_to_sequence(symbols):
    i = 0
    s = ""
    sequence = []
    while i < len(symbols):
        if symbols[i] == "<":
            while "</en>" not in s:
                s += symbols[i].lower()
                i += 1
        else:
            s = symbols[i].lower()
            i += 1
        sequence.append(_symbol_to_id[s])
        s=""
    return sequence

f = open("vi_dataset/script/train.txt")
data = f.readlines()
f.close()

f_train = open("vi_dataset/preprocess/train.txt", "w")
for line in data:
    wav, text = line.strip().split("|")
    wav_path = "vi_dataset/wav/" + wav + ".wav"
    text_path = "vi_dataset/text/" + wav + ".npy"
    sequence = torch.IntTensor(_symbols_to_sequence(text))
    f_train.write(wav_path+"|"+text_path+"\n")
    with open(text_path, 'wb') as f:
        np.save(f, sequence)

f_train.close()

f_train = open("vi_dataset/preprocess/val.txt", "w")
for line in data:
    wav, text = line.strip().split("|")
    wav_path = "vi_dataset/wav/" + wav + ".wav"
    text_path = "vi_dataset/text/" + wav + ".npy"
    sequence = torch.IntTensor(_symbols_to_sequence(text))
    f_train.write(wav_path+"|"+text_path+"\n")
    with open(text_path, 'wb') as f:
        np.save(f, sequence)

f_train.close()