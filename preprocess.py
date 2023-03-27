from numpy import outer
import torch
import numpy as np
from tqdm import tqdm
from transformers import BertTokenizer, BertModel
from tokenizers import Tokenizer
import os

from data import ljspeech
from data_utils import get_embedding, get_embedding_cls, process_text
import hparams
import argparse

def preprocess_ljspeech(filenname, in_dir, out_dir):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    metadata = ljspeech.build_from_path(filenname, in_dir, out_dir)
    write_metadata(metadata, out_dir)


def write_metadata(metadata, out_dir):
    print("***Metadata:",out_dir.split('/')[-1])
    with open(os.path.join(out_dir, out_dir.split('/')[-1]+".txt"), 'w', encoding='utf-8') as f:
        for m in metadata:
            f.write(m + '\n')

def main(tokenizer_vocab):
    model_bert = BertModel.from_pretrained('bert-base-multilingual-cased')
    tokenizer_custom = Tokenizer.from_file(f"data/vibert_{tokenizer_vocab}.json")
    tokenizer_bert = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
    hp = hparams.create_hparams()
    hp.bert_embeddings_train_path = hp.bert_embeddings_train_path.split("/")[0] + "_" + str(tokenizer_vocab) + "/" + hp.bert_embeddings_train_path.split("/")[1]
    hp.bert_embeddings_val_path = hp.bert_embeddings_val_path.split("/")[0] + "_" + str(tokenizer_vocab) + "/" + hp.bert_embeddings_val_path.split("/")[1]
    hp.bert_embeddings_cls_train_path = hp.bert_embeddings_cls_train_path.split("/")[0] + "_" + str(tokenizer_vocab) + "/" + hp.bert_embeddings_cls_train_path.split("/")[1]
    hp.bert_embeddings_cls_val_path = hp.bert_embeddings_cls_val_path.split("/")[0] + "_" + str(tokenizer_vocab) + "/" + hp.bert_embeddings_cls_val_path.split("/")[1]
    out_path = os.path.join("dataset", "train")
    preprocess_ljspeech(hp.datafiles, hp.training_files, out_path)
    out_path = os.path.join("dataset", "val")
    preprocess_ljspeech(hp.datafiles, hp.validation_files, out_path)

    text_path = os.path.join(hp.training_files)
    texts = process_text(text_path)

    if not os.path.exists(hp.bert_embeddings_train_path):
        os.makedirs(hp.bert_embeddings_train_path, exist_ok=True)
    
    if not os.path.exists(hp.bert_embeddings_val_path):
        os.makedirs(hp.bert_embeddings_val_path, exist_ok=True)

    if not os.path.exists(hp.bert_embeddings_cls_train_path):
        os.makedirs(hp.bert_embeddings_cls_train_path, exist_ok=True)
    
    if not os.path.exists(hp.bert_embeddings_cls_val_path):
        os.makedirs(hp.bert_embeddings_cls_val_path, exist_ok=True)

    if not os.path.exists("data/vi_dataset/preprocess"):
        os.makedirs("data/vi_dataset/preprocess", exist_ok=True)

    if not os.path.exists("data/vi_dataset/text"):
        os.makedirs("data/vi_dataset/text", exist_ok=True)

    f_train = open("data/vi_dataset/preprocess/train.txt", "w")
    for ind, text in enumerate(texts):
        wav, character = text.strip().split("\t")
        character = character.lower()
        sub_embedding = get_embedding(character, model_bert, tokenizer_custom)
        with open(os.path.join(hp.bert_embeddings_train_path, str(ind) + ".npy"), 'wb') as f:
            np.save(f, sub_embedding)
        bert_embedding_cls = get_embedding_cls(character, model_bert, tokenizer_bert)
        np.save(os.path.join(hp.bert_embeddings_cls_train_path, str(ind) + ".npy"),
            bert_embedding_cls.numpy(), allow_pickle=False)
        text_path = "data/vi_dataset/durations/" + wav + ".npy"
        
        f_train.write(text_path+"\n")

        if (ind+1) % 100 == 0:
            print("Done", (ind+1))
    f_train.close()
    
    text_path = hp.validation_files
    texts = process_text(text_path)

    f_train = open("data/vi_dataset/preprocess/val.txt", "w")
    for ind, text in enumerate(texts):
        wav, character = text.strip().split("\t")
        character = character.lower()
        sub_embedding = get_embedding(character, model_bert, tokenizer_custom)
        with open(os.path.join(hp.bert_embeddings_val_path, str(ind) + ".npy"), 'wb') as f:
            np.save(f, sub_embedding)
        bert_embedding_cls = get_embedding_cls(character, model_bert, tokenizer_bert)
        np.save(os.path.join(hp.bert_embeddings_cls_val_path, str(ind) + ".npy"),
                bert_embedding_cls.numpy(), allow_pickle=False)
        text_path = "data/vi_dataset/durations/" + wav + ".npy"
        f_train.write(text_path+"\n")

        if (ind+1) % 100 == 0:
            print("Done", (ind+1))
    f_train.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--tokenizer_vocab', type=int, help='tokenizer vocab size')
    args = parser.parse_args()
    main(args.tokenizer_vocab)
