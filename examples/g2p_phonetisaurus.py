import sys
sys.path.insert(0, '..')
from g2p.g2p import G2P_Phonetisaurus


if __name__=='__main__':
    phonetisaurus_model_path = '/data/work/models/00_vocab_g2p/01_vocab_kaldiformat/01_vietnamese_v2/phonetisaurus.fst'

    input_file = '/home/thinhnv13/Downloads/oov_vi.txt'
    phonetisaurus_output = '/home/thinhnv13/Downloads/oov_vi_phones.txt'
    g2p_phonetisaurus = G2P_Phonetisaurus(phonetisaurus_model_path, config="/data/work/models/00_vocab_g2p/01_vocab_kaldiformat/01_vietnamese_v2/config_phonetisaurus.yml")

    f_phonetisaurus = open(phonetisaurus_output, 'w')
    with open(input_file, 'r') as f:
        for line in f:
            print(line)
            word = line.strip().lower()
            f_phonetisaurus.write(''.join(str(g2p_phonetisaurus.g2p(word))) + '\n')
            
    f_phonetisaurus.close()
