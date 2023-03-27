import sys
sys.path.append('..')
from g2p.g2p import G2P_Transformer, G2P_Phonetisaurus
from g2p.lexicon import build_phone_lexicon

if __name__=='__main__':
    model_path='/media/khoamd/Hdd2TB_01/00_VinTTS/02_code/02_Tool/g2p/resources/transformer_resources/model_epoch_84_best_all.pt'
    phonetisaurus_model_path = '/media/khoamd/Hdd2TB_01/00_VinTTS/02_code/02_Tool/g2p/resources/phonetisaurus_models/02_10kForeign_27kForeign_17kVnSylable.lex.fst'

    input_file = 'input.txt'
    transformer_output = 'output_transformer.txt'
    phonetisaurus_output = 'output_phonetisaurus.txt'

    g2p = G2P_Transformer(model_path)#, config='../conf/config_phonetisaurus.cfg')
    g2pp = G2P_Phonetisaurus(phonetisaurus_model_path)#, config='../conf/config_phonetisaurus.cfg')

    f_trans = open(transformer_output, 'w')
    f_phonetisaurus = open(phonetisaurus_output, 'w')
    with open(input_file, 'r') as f:
        for line in f:
            word = line.strip()
            f_trans.write(g2p.g2p(word) + '\n')
            f_phonetisaurus.write("Phonetisaurus :" + g2pp.g2p(word) + '\n')
            print("Phonetisaurus :" + g2pp.g2p(word) + '\n')
    f_trans.close()
    f_phonetisaurus.close()
    #g2pp.build_vocab()
