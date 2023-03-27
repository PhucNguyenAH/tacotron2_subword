from nltk import word_tokenize
import re 
import torch
from sklearn.utils import shuffle

from g2p.utils.utils import norm_word, replace_multi_space, remove_puntuation, revert_norm_word

def tokenize(input_str):
    input_str = re.sub('^,', '', input_str)
    # input_str = re.sub('\.+\s*\.', '.', input_str) # replace multiple period with one period
    input_str = re.sub('(\.+\s+)+', '.', input_str) # replace multiple period with one period
    tokens = word_tokenize(input_str)
    for i in range(len(tokens)):
        tokens[i] = norm_word(tokens[i])
        # if i != len(tokens) - 1:
        #     tokens[i] = tokens[i].replace('.','')

    input_str = " ".join(tokens)
    input_str = remove_puntuation(input_str)
    input_str = revert_norm_word(input_str)

    return replace_multi_space(input_str)

def prepare_seq2seq(lines):
    src_all = list()
    tgt_all = list()
    for line in lines:
        line = tokenize(line)
        src = line.replace(',','')
        src = src.replace('.','')
        src = src.replace('?','')
        src = src.lower()
        src_all.append(src.strip())
        tgt_all.append(line.strip())

    return src_all, tgt_all

def read_data(data_path):
    data = torch.load(data_path)
    dict = data['dict']
    tgt_word2idx = dict['tgt']
    print(tgt_word2idx)

def split_train_test(lines, path, train_ratio=0.8, val_ratio=0.1):
    lines = shuffle(lines)
    train_size = int(len(lines) * train_ratio)
    val_size = int(len(lines) * val_ratio)
    print('train size: ', train_size)
    print('val size: ', val_size)
    train_data = lines[:train_size]
    val_data = lines[train_size:train_size + val_size]
    test_data = lines[train_size + val_size:]
    save_data(train_data, 'train', path)
    save_data(val_data, 'val', path)
    save_data(test_data, 'test', path)