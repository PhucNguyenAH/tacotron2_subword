import sys
sys.path.insert(0, '..')
from g2p.text_to_sequence import Text2Seq


if __name__=='__main__':
    phonetisaurus_model_path = '/data/work/neuraltts_work/g2p_ngram/nGram/models/02_10kForeign_27kForeign_17kVnSylable.lex.fst'
    input_file = 'input_t2s.txt'
    phonetisaurus_output_z = 'output_phonetisaurus_sequence_z.txt'
    t2s_z = Text2Seq(phonetisaurus_model_path, delimiter = 'z', ignore_white_space = True)

    f_phonetisaurus_z = open(phonetisaurus_output_z, 'w')
    with open(input_file, 'r') as f:
        for line in f:
            word = line.strip().lower()
            
            f_phonetisaurus_z.write(' '.join(str(t2s_z.grapheme_to_sequence(word))) + '\n')
            print('Phonetisaurus delimiter z text: {} \nSequence delimiter: {}'.format(word, t2s_z.grapheme_to_sequence(word,)))
    f_phonetisaurus_z.close()
    #g2pp.build_vocab()