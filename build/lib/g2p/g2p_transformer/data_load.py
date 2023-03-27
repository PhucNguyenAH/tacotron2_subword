from g2p.g2p_transformer.hyperparams import Hyperparams as params

import numpy as np
import codecs
import random


def load_source_vocab():
    vocab = [line.split()[0] for line in codecs.open(params.src_vocab, 'r', 'utf-8').read().splitlines() if int(line.split()[1]) >= params.min_cnt]
    word2idx = {word: idx for idx, word in enumerate(vocab)}
    idx2word = {idx: word for idx, word in enumerate(vocab)}
    return word2idx, idx2word


def load_target_vocab():
    vocab = [line.split()[0] for line in codecs.open(params.tgt_vocab, 'r', 'utf-8').read().splitlines() if int(line.split()[1]) >= params.min_cnt]
    word2idx = {word: idx for idx, word in enumerate(vocab)}
    idx2word = {idx: word for idx, word in enumerate(vocab)}
    return word2idx, idx2word


def convert_word2idx(word2idx, samples):
    samples_idxes = list()
    for source_sent in samples:
        sample_idx = list()
        for word in (source_sent + " </s>").split():
            if word in word2idx:
                sample_idx.append(word2idx[word])
            else:
                sample_idx.append(word2idx['<unk>'])

        if len(sample_idx) <= params.maxlen:
            samples_idxes.append(np.array(sample_idx))

    # Pad
    sample_idx_pad = np.zeros([len(samples_idxes), params.maxlen], np.int32)
    for i, sample_idx in enumerate(samples_idxes):
        sample_idx_pad[i] = np.lib.pad(sample_idx, [0, params.maxlen - len(sample_idx)], 'constant', constant_values=(0, 0))

    return sample_idx_pad


def create_data(source_sents, target_sents): 
    source2idx, idx2source = load_source_vocab()
    target2idx, idx2target = load_target_vocab()
    
    # Index
    source_idx, target_idx, source_text, target_text = [], [], [], []
    for source_sent, target_sent in zip(source_sents, target_sents):
        # x = [source2idx[word] for word in (source_sent + " </S>").split()] # 1: OOV, </S>: End of Text
        # y = [target2idx[word] for word in (target_sent + " </S>").split() if word in target2idx else target2idx['<UNK>']]

        x = list()
        for word in (source_sent + " </s>").split():
            if word in source2idx:
                x.append(source2idx[word])
            else:
                x.append(source2idx['<unk>'])

        y = list()
        for word in (target_sent + " </s>").split():
            if word in target2idx:
                y.append(target2idx[word])
            else:
                y.append(target2idx['<unk>'])

        if max(len(x), len(y)) <= params.maxlen:
            source_idx.append(np.array(x))
            target_idx.append(np.array(y))
            source_text.append(source_sent)
            target_text.append(target_sent)

    # Pad      
    source_idxes = np.zeros([len(source_idx), params.maxlen], np.int32)
    target_idxes = np.zeros([len(target_idx), params.maxlen], np.int32)
    for i, (x, y) in enumerate(zip(source_idx, target_idx)):
        source_idxes[i] = np.lib.pad(x, [0, params.maxlen - len(x)], 'constant', constant_values=(0, 0))
        target_idxes[i] = np.lib.pad(y, [0, params.maxlen - len(y)], 'constant', constant_values=(0, 0))
    
    return source_idxes, target_idxes, source_text, target_text


def load_train_data():
    src_sents = [line for line in open(params.source_train, 'r').read().split("\n")]
    tgt_sents = [line for line in open(params.target_train, 'r').read().split("\n")]

    source_idxes, target_idxes, source_text, target_text = create_data(src_sents, tgt_sents)
    return source_idxes, target_idxes


def load_test_data():
    src_sents = [line for line in open(params.source_test, 'r').read().split("\n")]
    tgt_sents = [line for line in open(params.target_test, 'r').read().split("\n")]

    source_idxes, target_idxes, source_text, target_text = create_data(src_sents, tgt_sents)
    return source_idxes, source_text, target_text  # (1064, 150)


def get_batch_indices(total_length, batch_size):
    current_index = 0
    indexs = [i for i in range(total_length)]
    random.shuffle(indexs)
    while 1:
        if current_index + batch_size >= total_length:
            break
        current_index += batch_size
        yield indexs[current_index: current_index + batch_size], current_index


