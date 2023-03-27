import torch
from torch.autograd import Variable
import numpy as np

from g2p.g2p import G2P
from g2p.g2p_transformer.hyperparams import Hyperparams as params
from g2p.g2p_transformer.transformer import Transformer
from g2p.g2p_transformer.data_load import convert_word2idx
from conf import default_transformer_cfg


class G2P_Transformer(G2P):
    def __init__(self, model_path, config=default_transformer_cfg):
        super(G2P_Transformer, self).__init__(config)
        self.use_gpu = self.config['g2p']['use_gpu']
        self.model_path = model_path
        self.model, self.source2idx, self.idx2target = self.load_model()

    def load_model(self, model_path = None):
        if model_path is not None:
            self.model_path = model_path
        model_dict = torch.load(self.model_path, map_location=lambda storage, loc:storage)
        source2idx, idx2source = model_dict['source2idx'], model_dict['idx2source']
        target2idx, idx2target = model_dict['target2idx'], model_dict['idx2target']
        encoder_vocab = len(source2idx)
        decoder_vocab = len(target2idx)

        # load model
        model = Transformer(params, encoder_vocab, decoder_vocab)
        # model.load_state_dict(torcah.load(model_path))
        model.load_state_dict(model_dict['state_dict'])
        model.eval()
        if self.use_gpu:
            model.cuda()
        print('Model Loaded.')

        return model, source2idx, idx2target

    def infer(self, sample):
        print('Transformer')
        sample = ' '.join(list(sample))
        batch_size = 1
        sample_idx = convert_word2idx(self.source2idx, [sample])

        # Autoregressive inferencemultihead_attention
        if self.use_gpu:
            source_idx_tensor = Variable(torch.LongTensor(sample_idx).cuda())
            preds_t = torch.LongTensor(np.zeros((batch_size, params.maxlen), np.int32)).cuda()
        else:
            source_idx_tensor = Variable(torch.LongTensor(sample_idx))
            preds_t = torch.LongTensor(np.zeros((batch_size, params.maxlen), np.int32))
        preds = Variable(preds_t)
        for j in range(params.maxlen):
            _, _preds, _ = self.model(source_idx_tensor, preds)
            preds_t[:, j] = _preds.data[:, j]
            preds = Variable(preds_t.long())
        preds = preds.data.cpu().numpy()

        got = " ".join(self.idx2target[idx] for idx in preds[0]).split("</s>")[0].strip().replace(' | ', '|').replace(' $ ', ' ')
        print("- source: " + sample)
        print("-    got: " + got + "\n")

        return got
