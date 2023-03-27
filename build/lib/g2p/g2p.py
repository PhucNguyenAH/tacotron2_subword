import re
import yaml
import os

import phonetisaurus
from g2p.utils.utils import current_millis_time
from g2p.lexicon import build_lexicon
from conf import default_phonetisaurus_cfg
from resources import resources_dir

class G2P:
    def __init__(self, config_obj):
        #if not isinstance()
        if  isinstance(config_obj, str):
            if not os.path.isfile(config_obj):
                raise IOError('No such file: {}'.format(config_obj))
            with open(config_obj) as f:
                self.config = yaml.load(f, Loader=yaml.Loader)
        else:
            self.config = config_obj
        if self.config['resources']['load_default']:
            vi_lex_file = os.path.join(resources_dir, self.config['resources']['vi_lexicon_file'])
            en_lex_file = os.path.join(resources_dir, self.config['resources']['en_lexicon_file'])
            foreign_lex_file = os.path.join(resources_dir, self.config['resources']['foreign_lexicon_file'])
        else:
            vi_lex_file = self.config['resources']['vi_lexicon_file']
            en_lex_file = self.config['resources']['en_lexicon_file']
            foreign_lex_file = self.config['resources']['foreign_lexicon_file']
        
        self.lexicon = build_lexicon(vi_lex_file, en_lex_file, foreign_lex_file)
        self.viLex, self.enLex, self.foreignLex = self.lexicon
        self._punctuation = self.config['g2p']['punctuation'].replace('\\','')
        
    def load_config(self, config_obj):
        pass

    def load_model(self):
        raise NotImplementedError

    # infer unkown word using g2p model
    def infer(self, sample):
        raise NotImplementedError

    # convert grapheme to phoneme
    def g2p(self, text, _punctuation = None):
        if _punctuation is not None:
            self._punctuation = _punctuation
        tmp_string=""
        unk_list = []
        for nword in text.split(): 
            if nword in self.viLex: # found in vietnamese lexicon
                lexout = self.viLex[nword]
                tmp_string = tmp_string + ' ' +  re.sub(' ','|', ' '.join(lexout.split()))
            elif nword in self.foreignLex:
                lexout = self.foreignLex[nword]
                tmp_string = tmp_string + ' ' + re.sub(' ','|',' '.join(lexout.split()))
            elif nword in self.enLex: # found in english lexicon
                lexout = self.enLex[nword]
                tmp_string = tmp_string + ' ' +  re.sub(' ','|', ' '.join(lexout.split()))
            else:
                if nword in self._punctuation:
                    tmp_string = tmp_string + ' ' + nword 
                else:
                    unk_list.append(nword)
                    unk_g2p = self.infer(nword)
                    # if unk_g2p and unk_g2p.strip() != ' ':
                    tmp_string = tmp_string + ' ' + unk_g2p
                    
        if unk_list:
            print('The word not in dictionaries: {}'.format(unk_list))
        tmp_string = tmp_string.strip()
        if tmp_string and self.config['kaldi_format']['kaldi_format']:
            tmp_string = self.convert_kaldi_format(tmp_string)
        
        return tmp_string

    def convert_kaldi_format(self, phone_seq):
        tmp_string = ''
        begin = self.config['kaldi_format']['begin']
        end = self.config['kaldi_format']['end']
        inner = self.config['kaldi_format']['inner']
        single = self.config['kaldi_format']['single']
        #Hardcode
        g2p_punctuation = self.config['kaldi_format']['g2p_punctuation'] #{'!': 'lpau', '\'': None, '(': None, ')': None, ',': 'mpau', '.': 'lpau', ':': 'lpau', ';':'lpau', '?': 'lpau', ' ': None}
        #Check g2p_punctuation converter include all punctuation
        for punc in self._punctuation:
            if punc not in g2p_punctuation:
                raise ValueError('Punctuation \'{}\' must be configured in g2p_punctuation'.format(punc))
        for syllable_phone_seq in phone_seq.split(' '):
            
            syllable_phone_seqs = [syllable_phone for syllable_phone in syllable_phone_seq.split('|') if syllable_phone.strip() ]
            if not syllable_phone_seqs:
                continue
            
            # print(syllable_phone_seqs)
            # Check punctuation is single word
            if len(syllable_phone_seqs) > 1:
                for phone in syllable_phone_seqs:
                    if phone and phone in self._punctuation:
                        raise RuntimeError('Punctuation \'{}\' must be single word!'.format(phone))
            
            if len(syllable_phone_seqs) == 1:
                if syllable_phone_seqs[0] in self._punctuation:
                    phone_punc = g2p_punctuation[syllable_phone_seqs[0]]
                    if phone_punc:
                        tmp_string = tmp_string + ' ' + g2p_punctuation[syllable_phone_seqs[0]]
                else:
                    tmp_string = tmp_string + ' ' + syllable_phone_seqs[0] + single
            elif len(syllable_phone_seqs) == 2:
                tmp_string = tmp_string + ' ' + syllable_phone_seqs[0] + begin + '|' + syllable_phone_seqs[1] + end
            else:
                tmp_string = tmp_string + ' ' + syllable_phone_seqs[0] + begin
                tmp_string = tmp_string + '|'
                for i in range(1,len(syllable_phone_seqs) - 1):
                    tmp_string = tmp_string + syllable_phone_seqs[i] + inner + '|'
                
                tmp_string = tmp_string + syllable_phone_seqs[-1] + end
        return tmp_string.strip()

class G2P_Phonetisaurus(G2P):
    def __init__(self, model_path, config=default_phonetisaurus_cfg):
        super(G2P_Phonetisaurus, self).__init__(config)
        self.model_path = model_path
        self.load_config()
        self.model = self.load_model()
    
    def load_config(self):
        self.nbest = int(self.config['g2p']['nbest'])
        self.beam = int(self.config['g2p']['beam'])
        self.thresh = float(self.config['g2p']['thresh'])
        self.write_fsts = self.config['g2p']['write_fsts']
        self.accumulate = self.config['g2p']['accumulate']
        self.pmass = float(self.config['g2p']['pmass'])
        
    def load_model(self, model_path = None):
        if model_path is not None:
            self.model_path = model_path
        model = phonetisaurus.Phonetisaurus(self.model_path)
        return model
    def infer(self, sample):
        '''
        This code is originally adapted from: https://github.com/AdolfVonKleist/Phonetisaurus/blob/master/python/script/phoneticize.py
        '''
        # print('nbest: {}'.format(type(self.nbest)))
        # print('Phonetisaurus')
        results = self.model.Phoneticize(sample, self.nbest, self.beam, self.thresh, self.write_fsts, self.accumulate, self.pmass)
        pronunciations = []
        for result in results:
            pronunciation = [self.model.FindOsym (u) for u in result.Uniques]
            #print(pronunciation)
            phones = []
            for phone in pronunciation:
                phones.append(phone.replace('9','_'))
            pronunciations.append('|'.join(phones))
        got = ' '.join(pronunciations)
        # print("- source: " + sample)
        # print("-    got: " + got + "\n")
        return got
    
if __name__=='__main__':
    # model_path='/media/khoamd/Hdd2TB_01/00_VinTTS/02_code/02_Tool/g2p/resources/transformer_resources/model_epoch_84_best_all.pt'
    # g2p = G2P_Transformer(model_path)
    start = current_millis_time()
    sample='xin chào deexmind'
    
    phonetisaurus_model_path = '/media/khoamd/Hdd2TB_01/00_VinTTS/02_code/02_Tool/g2p/resources/phonetisaurus_models/02_10kForeign_27kForeign_17kVnSylable.lex.fst'
    g2pp = G2P_Phonetisaurus(phonetisaurus_model_path)
    print(g2pp.g2p(sample))
    
    #print(g2pp.infer(sample))
    
    processingtime = current_millis_time() - start
    print('processingtime: {}'.format(processingtime))