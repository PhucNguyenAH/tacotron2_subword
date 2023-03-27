import codecs
import os
import math

import numpy as np

from g2p.g2p_transformer.hyperparams import Hyperparams as params
from g2p.g2p_transformer.data_load import load_test_data, load_source_vocab, load_target_vocab, convert_word2idx
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu
from g2p.g2p_transformer.transformer import Transformer
from torch.autograd import Variable
import torch


def eval():
    # Load data
    source_idxes, source_texts, target_texts = load_test_data()
    source2idx, idx2source = load_source_vocab()
    target2idx, idx2target = load_target_vocab()
    encoder_vocab = len(source2idx)
    decoder_vocab = len(target2idx)

    # load model
    model = Transformer(params, encoder_vocab, decoder_vocab)
    model.load_state_dict(torch.load(params.model_dir + '/model_epoch_%02d' % params.eval_epoch + '.pth'))
    print('Model Loaded.')
    model.eval()
    if params.use_gpu:
        model.cuda()
    # Inference
    if not os.path.exists('results'):
        os.mkdir('results')
    with open(params.eval_result + '/model%d.txt' % params.eval_epoch, 'w') as fout:
        list_of_refs, hypotheses = [], []
        scores = list()
        for i in range(len(source_idxes) // params.batch_size):
            # Get mini-batches
            source_idx_batches = source_idxes[i * params.batch_size : (i + 1) * params.batch_size]
            source_text_batches = source_texts[i * params.batch_size : (i + 1) * params.batch_size]
            target_text_batches = target_texts[i * params.batch_size : (i + 1) * params.batch_size]

            # Autoregressive inferencemultihead_attention
            if params.use_gpu:
                source_idx_batches_tensor = Variable(torch.LongTensor(source_idx_batches).cuda())
                preds_t = torch.LongTensor(np.zeros((params.batch_size, params.maxlen), np.int32)).cuda()
            else:
                source_idx_batches_tensor = Variable(torch.LongTensor(source_idx_batches))
                preds_t = torch.LongTensor(np.zeros((params.batch_size, params.maxlen), np.int32))
            preds = Variable(preds_t)
            for j in range(params.maxlen):
                _, _preds, _ = model(source_idx_batches_tensor, preds)
                preds_t[:, j] = _preds.data[:, j]
                preds = Variable(preds_t.long())
            preds = preds.data.cpu().numpy()

            # Write to file
            for source, target, pred in zip(source_text_batches, target_text_batches, preds):  # sentence-wise
                got = " ".join(idx2target[idx] for idx in pred).split("</s>")[0].strip()
                fout.write("-   source: " + source + "\n")
                # print("-   source: " + source)
                fout.write("- expected: " + target + "\n")
                # print("- expected: " + target)
                fout.write("-      got: " + got + "\n\n")
                # print("-      got: " + got + "\n")
                fout.flush()

                # bleu score
                ref = target.split()
                hypothesis = got.split()
                if len(ref) > 3 and len(hypothesis) > 3:
                    list_of_refs.append([ref])
                    hypotheses.append(hypothesis)
            # Calculate bleu score
            score = corpus_bleu(list_of_refs, hypotheses)
            scores.append(score)
            fout.write("Bleu Score = " + str(100 * score))
            print('Bleu Score: ', score)
        fout.write('Bleu Score MEAN: ' + str(np.array(scores).mean()))
        print('Bleu Score MEAN: ', np.array(scores).mean())


def infer(model, source2idx, idx2target, sample):
    batch_size = 1
    sample_idx = convert_word2idx(source2idx, [sample])

    # Autoregressive inferencemultihead_attention
    if params.use_gpu:
        source_idx_tensor = Variable(torch.LongTensor(sample_idx).cuda())
        preds_t = torch.LongTensor(np.zeros((batch_size, params.maxlen), np.int32)).cuda()
    else:
        source_idx_tensor = Variable(torch.LongTensor(sample_idx))
        preds_t = torch.LongTensor(np.zeros((batch_size, params.maxlen), np.int32))
    preds = Variable(preds_t)
    for j in range(params.maxlen):
        _, _preds, _ = model(source_idx_tensor, preds)
        preds_t[:, j] = _preds.data[:, j]
        preds = Variable(preds_t.long())
    preds = preds.data.cpu().numpy()

    got = " ".join(idx2target[idx] for idx in preds[0]).split("</s>")[0].strip().replace(' | ', '|').replace(' $ ', ' ')
    print("- source: " + sample)
    print("-    got: " + got + "\n")

    return got


def load_model(model_path, use_gpu):
    # Load data
    source2idx, idx2source = load_source_vocab()
    target2idx, idx2target = load_target_vocab()
    encoder_vocab = len(source2idx)
    decoder_vocab = len(target2idx)

    # load model
    model = Transformer(params, encoder_vocab, decoder_vocab)
    # model.load_state_dict(torch.load(model_path))
    model.load_state_dict(torch.load(model_path, map_location=lambda storage, loc: storage))
    print('Model Loaded.')
    model.eval()
    if use_gpu:
        model.cuda()

    # torch.save({'state_dict': model.state_dict(), 'source2idx':source2idx, 'idx2source':idx2source, 'target2idx':target2idx, 'idx2target':idx2target}, 'resources/g2p_model_v2/model_epoch_100_best_all.pt')

    return model, source2idx, idx2target


if __name__ == '__main__':
    # eval()
    print('Done')

    model, source2idx, idx2target = load_model(params.model_dir + '/model_epoch_%02d' % params.eval_epoch + '_best.pth', use_gpu=True)
    while True:
        sample = input('Input your sentence: ')
        sample = ' '.join(list(sample))
        infer(model, source2idx, idx2target, sample)



