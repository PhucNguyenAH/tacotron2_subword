import torch
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from transformers import BertTokenizer
from multiprocessing import cpu_count
import math
import os

import hparams

hp = hparams.create_hparams()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def add_cls_sep(text):
    return "[CLS] " + text + " [SEP]"

def get_embedding(text, bert_model, tokenizer):
    hp = hparams.create_hparams()
    text = text.replace("<en>","")
    text = text.replace("</en>","")
    
    tokenized_text = tokenizer.encode(text)
    indexed_tokens = tokenized_text.ids
    sequence = torch.IntTensor(indexed_tokens[1:-1])
    return sequence

def get_embedding_cls(text, bert_model, tokenizer):
    hp = hparams.create_hparams()
    text = text.replace("<en>","")
    text = text.replace("</en>","")
    
    text_cleaned = text
    text_processed = add_cls_sep(text_cleaned)
    tokenized_text = tokenizer.tokenize(text_processed)
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    segments_ids = [0 for i in range(len(indexed_tokens))]
    tokens_tensor = torch.tensor([indexed_tokens])
    segments_tensors = torch.tensor([segments_ids])

    with torch.no_grad():
        encoded_layers = bert_model(tokens_tensor, segments_tensors)[0]

    bert_embeddings = encoded_layers[0]
    bert_embeddings = bert_embeddings[0].unsqueeze(0)
    return bert_embeddings

class BERTTacotron2Dataset(Dataset):
    """ LJSpeech """

    def __init__(self, dataset_path="train", text_path=hp.training_preprocess, embedding_path=hp.bert_embeddings_train_path, embedding_cls_path=hp.bert_embeddings_cls_train_path):
        self.dataset_path = os.path.join("dataset", dataset_path)
        self.text_path = text_path
        self.text = process_text(self.text_path)

        self.embedding_path = embedding_path
        self.embedding_cls_path = embedding_cls_path
        self.alignloss = hp.alignloss

    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        index = idx + 1
        mel_name = os.path.join(
            self.dataset_path, "ljspeech-mel-%05d.npy" % index)
        mel_target = np.load(mel_name)

        phoneme = torch.from_numpy(np.load(self.text[idx]).astype(int))[:,0]

        bert_embedding = torch.from_numpy(np.load(os.path.join(self.embedding_path, str(idx)+".npy")))

        embedding_cls = np.load(os.path.join(
            self.embedding_cls_path, str(idx)+".npy"))
        embedding_cls = torch.from_numpy(embedding_cls)

        phoneme_embedding_cls = embedding_cls.repeat(phoneme.size(0),1)
        bert_embedding_cls = embedding_cls.repeat(bert_embedding.size(0),1)

        stop_token = np.array([0. for _ in range(mel_target.shape[0])])
        stop_token[-1] = 1.

        sample = {"text": phoneme, "mel_target": mel_target, "bert_embedding": bert_embedding, 
        "bert_embedding_cls": bert_embedding_cls, "phoneme_embedding_cls":phoneme_embedding_cls,  "stop_token": stop_token}

        return sample


def process_text(train_text_path):
    with open(train_text_path, "r", encoding="utf-8") as f:
        txt = []
        for line in f.readlines():
            txt.append(line.strip())

        return txt


def reprocess(batch, cut_list):
    texts = [batch[ind]["text"] for ind in cut_list]
    bert_embeddings = [batch[ind]["bert_embedding"] for ind in cut_list]
    bert_embeddings_cls = [batch[ind]["bert_embedding_cls"] for ind in cut_list]
    phoneme_embeddings_cls = [batch[ind]["phoneme_embedding_cls"] for ind in cut_list]
    mel_targets = [batch[ind]["mel_target"] for ind in cut_list]
    stop_tokens = [batch[ind]["stop_token"] for ind in cut_list]

    length_text = np.array([])
    for text in texts:
        length_text = np.append(length_text, text.shape[0])

    length_bert = np.array([])
    for emb in bert_embeddings:
        length_bert = np.append(length_bert, emb.shape[0])

    length_mel = np.array([])
    for mel in mel_targets:
        length_mel = np.append(length_mel, mel.shape[0])

    texts = pad_normal(texts)
    stop_tokens = pad_normal(stop_tokens, PAD=1.)
    mel_targets = pad_mel(mel_targets)
    bert_embeddings = pad_normal(bert_embeddings)
    bert_embeddings_cls = pad_emb(bert_embeddings_cls)
    phoneme_embeddings_cls = pad_emb(phoneme_embeddings_cls)
    # print("TEXT:",texts)
    if hp.alignloss != "":
        align = get_alignment(texts)
    else:
        align = texts

    out = {"text": texts, "mel_target": mel_targets, "stop_token": stop_tokens, "bert_embeddings": bert_embeddings, "bert_embeddings_cls": bert_embeddings_cls, 
    "phoneme_embeddings_cls":phoneme_embeddings_cls, "length_mel": length_mel, "length_text": length_text, "length_bert": length_bert, "align": align}

    return out

def get_alignment(self, filename):
    duration_predictor_output = np.load(filename).astype(int)
    duration_predictor_output = torch.from_numpy(duration_predictor_output)[:,1]
    duration_predictor_output = torch.unsqueeze(duration_predictor_output, 0)
    
    frame_lens = torch.sum(duration_predictor_output, -1)
    expand_max_frame_len = torch.max(frame_lens, -1)[0]
    alignment = torch.zeros(duration_predictor_output.size(0), expand_max_frame_len, duration_predictor_output.size(1))
    alignment = create_alignment(alignment, duration_predictor_output)
    return alignment[0]

def collate_fn(batch):
    len_arr = np.array([d["text"].shape[0] for d in batch])
    index_arr = np.argsort(-len_arr)
    batchsize = len(batch)
    real_batchsize = int(math.sqrt(batchsize))

    cut_list = list()
    for i in range(real_batchsize):
        cut_list.append(index_arr[i*real_batchsize:(i+1)*real_batchsize])

    output = list()
    for i in range(real_batchsize):
        output.append(reprocess(batch, cut_list[i]))

    return output


def pad_normal(inputs, PAD=0):

    def pad_data(x, length, PAD):
        x_padded = np.pad(
            x, (0, length - x.shape[0]), mode='constant', constant_values=PAD)
        return x_padded

    max_len = max((len(x) for x in inputs))
    padded = np.stack([pad_data(x, max_len, PAD) for x in inputs])

    return padded


def pad_mel(inputs):

    def pad(x, max_len):
        PAD = 0
        if np.shape(x)[0] > max_len:
            raise ValueError("not max_len")

        s = np.shape(x)[1]
        x_padded = np.pad(x, (0, max_len - np.shape(x)
                              [0]), mode='constant', constant_values=PAD)
        return x_padded[:, :s]

    max_len = max(np.shape(x)[0] for x in inputs)
    mel_output = np.stack([pad(x, max_len) for x in inputs])

    return mel_output


def pad_emb(inputs):

    def pad(x, max_len):
        if np.shape(x)[0] > max_len:
            raise ValueError("not max_len")
        s = x.size(1)
        x_padded = F.pad(x, (0, 0, 0, max_len-x.size(0)))
        return x_padded[:, :s]

    max_len = max(x.size(0) for x in inputs)
    mel_output = torch.stack([pad(x, max_len) for x in inputs])

    return mel_output
