import sys
sys.path.insert(0, '..')
from g2p.text_to_sequence import Text2Seq

if __name__ == '__main__':
    phonetisaurus_model_path = '/data/work/Releases/01_g2p/g2p_resources/g2p_resources-v2.1.3/vi_phonetisaurus-v1.0.0.fst'
    config = '/data/work/Releases/01_g2p/g2p_resources/g2p_resources-v2.1.3/config_phonetisaurus_v1.0.2.yml'
    input_file = 'input_t2s.txt'
    phonetisaurus_output = 'output_phonetisaurus_sequence.txt'
    t2s = Text2Seq(phonetisaurus_model_path, g2p_config=config,
                   phone_id_list_file='/data/work/Releases/01_g2p/g2p_resources/g2p_resources-v2.1.3/phone_id-v2.1.0.map', ignore_white_space=True, delimiter=None)

    f_phonetisaurus = open(phonetisaurus_output, 'w')
    with open(input_file, 'r') as f:
        for line in f:
            word = line.strip().lower()
            sequence = str(t2s.grapheme_to_sequence(word))
            # sequence2 = str(t2s.text_to_sequence(word, is_phone=False, padding=True))
            f_phonetisaurus.write(' '.join(sequence) + '\n')
            print('sequence: {}'.format(sequence))
            # print('sequence2: {}'.format(sequence2))

    f_phonetisaurus.close()
    # g2pp.build_vocab()
