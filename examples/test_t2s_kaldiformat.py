import sys
sys.path.insert(0, '..')
from g2p.text_to_sequence import Text2Seq


if __name__=='__main__':
    phonetisaurus_model_path = '/data/work/models/00_vocab_g2p/01_vocab_kaldiformat/vinbdi-slp-tts_resources_models-v1.0.0/resources/vietnamese/g2p/phonetisaurus.fst'
    g2p_config = '/data/work/models/00_vocab_g2p/01_vocab_kaldiformat/vinbdi-slp-tts_resources_models-v1.0.0/resources/vietnamese/g2p/config_phonetisaurus.yml'
    phone_id_list_file = '/data/work/models/00_vocab_g2p/01_vocab_kaldiformat/vinbdi-slp-tts_resources_models-v1.0.0/resources/vietnamese/g2p/phone_id.new_align.for_fastspeech2.map'
    phonetisaurus_output = 'output_t2s_kaldiformat.txt'
    t2s = Text2Seq(phonetisaurus_model_path, g2p_config=g2p_config, phone_id_list_file=phone_id_list_file,  delimiter=None, ignore_white_space=True)
    
    text = 'xác minh việc đổ chất thải của công ty đức thành trên địa bàn gia nghĩa và huyện đắk g long qua khảo sát thực tế tại khu vực mỏ đá bốn a bon srê ú xã đắk nia gia nghĩa có tồn trữ khối lượng tro bay với khối lượng khoảng hai nghìn hai trăm mét khối tại khu vực dọc quốc lộ hai mươi tám xã quảng khê huyện đắk g long khoảng hai trăm mười mét khối tro bay"'

    f_phonetisaurus = open(phonetisaurus_output, 'w')
    seq = t2s.grapheme_to_sequence(text)
    f_phonetisaurus.write(' '.join(str(seq)) + '\n')
    print('Phonetisaurus kaldiformat text: {} \nSequence delimiter: {}'.format(text, seq))
    f_phonetisaurus.close()
    #g2pp.build_vocab()