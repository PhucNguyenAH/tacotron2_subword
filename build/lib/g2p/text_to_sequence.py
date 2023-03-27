import os

from g2p.g2p import G2P_Phonetisaurus
from resources import resources_dir
from g2p.lexicon import load_phone_id_list, load_character_id_list

class Text2Seq:
    '''
    - g2p_model_type: phonetisaurus or transformer
    - ignore_white_space: A boolean variable to decide ignore whitespace in outputs.
    - delimiter: If this variable is None then nothing change, else phone_delimiter will be appended to the end phone of syllable.
        + For examples: phone_delimiter = 'z' then 'a|n|h' will be converted into 'a|n|hz'
    '''
    def __init__(self, g2p_model_path, g2p_config = None, phone_id_list_file=None, g2p_model_type='phonetisaurus', delimiter = None, ignore_white_space = False):
        if phone_id_list_file is None:
            if delimiter is not None:
                phone_id_list_file = os.path.join(resources_dir, 'phone_id_list.delimiter.' + delimiter + '.txt')
            else:
                phone_id_list_file = os.path.join(resources_dir, 'phone_id_list.txt')
        if g2p_model_type == 'phonetisaurus':
            if g2p_config is not None:
                g2p_model = G2P_Phonetisaurus(g2p_model_path, g2p_config)
            else:
                g2p_model = G2P_Phonetisaurus(g2p_model_path)
        elif g2p_model_type == 'transformer':
            raise ValueError('The {} g2p model is no longer supported.')
            # if g2p_config is not None:
            #     g2p_model = G2P_Transformer(g2p_model_path, g2p_config)
            # else:
            #     g2p_model = G2P_Transformer(g2p_model_path)
        lexicon_data = g2p_model.lexicon
        self.config = g2p_model.config
        self.g2p = g2p_model
        self.delimiter = delimiter
        self.ignore_white_space = ignore_white_space

        self._pad = self.config['t2s']['pad']
        self._special = self.config['t2s']['special']
        self._EOS = self.config['t2s']['EOS']
        self._BOS = self.config['t2s']['BOS']
        if 'white_space' in self.config['t2s']:
            self.white_space = self.config['t2s']['white_space']
        else:
            self.white_space = ' '
        
        self._letters = self.config['t2s']['letters']
        self._punctuation = self.config['g2p']['punctuation'].replace('\\','')

        other_symbols = list(self._pad) + list(self._special) + list(self._EOS) + list(self._BOS) + list(self._punctuation)

        self.phone_to_id, self.id_to_phone = load_phone_id_list(lexicon_data, phone_id_list_file, other_symbols = other_symbols, phone_delimiter=self.delimiter)
        self.symbol_to_id, self.id_to_symbol = load_character_id_list(self._letters, other_symbols = other_symbols)
        #Append delimiter after load phone_to_id and symbol_to_id
        if self.delimiter is not None:
            self._EOS = self._EOS + self.delimiter
            self._BOS = self._BOS + self.delimiter
    
    def pad_sequence(self, sequence, is_phone = True):
        if is_phone:
            EOS = self.phone_to_id[self._EOS]
            BOS = self.phone_to_id[self._BOS]
        else:
            EOS = self.symbol_to_id[self._EOS]
            BOS = self.symbol_to_id[self._BOS]
        sequence.insert(0,BOS)
        sequence.append(EOS)
        return sequence
    
    def _convert_phone_to_id(self, phone):
        phone_id = None
        if phone and (phone in self.phone_to_id):
            phone_id = self.phone_to_id[phone]
            
        else:
            print('WARNING: phone \"{}\" is not in phone id map'.format(phone))
        
        return phone_id
    
    '''
    Convert text or phone to interger sequence:
    - inputs: it can be text or phone sequence.
    - is_phone: A boolean variable to determine inputs is text or phone sequence.
    - ignore_white_space: A boolean variable to decide ignore whitespace in outputs.
    Note: If delimiter (of this instance) is None then nothing change, else the delimiter will be appended to the end phoneme of syllable.
        Examples: 
                + delimiter = 'z'
                so 'p|h|i|n t|h|i|m' will be converted to 'p|h|i|nz t|h|i|mz'. 
    '''
    def text_to_sequence(self, inputs, is_phone = True, padding = False):
        delimiter = self.delimiter
        sequence = []

        if is_phone:
            for syllable in inputs.split(' '):
                if self.delimiter is not None:
                    syllable = syllable + self.delimiter
                for phone in syllable.split('|'):
                    phone_id = self._convert_phone_to_id(phone)
                    if phone_id:
                        sequence.append(phone_id)
                if not self.ignore_white_space:
                    if self.delimiter is not None:
                        phone_id = self._convert_phone_to_id(self.white_space + self.delimiter)
                        if phone_id:
                            sequence.append(phone_id)
                    else:
                        phone_id = self._convert_phone_to_id(self.white_space)
                        if phone_id:
                            sequence.append(phone_id)

            if not self.ignore_white_space:
                sequence = sequence[:-1]
        else:
            for char in list(inputs.replace('\\','')):
                sequence.append(self.symbol_to_id[char])
        if padding:
            sequence = self.pad_sequence(sequence, is_phone=is_phone)
        return sequence
    
    '''
    Convert text to phone sequence and finally to interger sequence:
    - text: it is text input.
    - ignore_white_space: A boolean variable to decide ignore whitespace in outputs.
    Note: If delimiter (of this instance) is None then nothing change, else the delimiter will be appended to the end phoneme of syllable.
        Examples: 
                + delimiter = 'z'
                so 'p|h|i|n t|h|i|m' will be converted to 'p|h|i|nz t|h|i|mz'. 
    - padding: If it is True, Id End of sentence character _EOS will be appended at the end the sequence and Id of Begin of sentence _BOS will be inserted at the begin of sentence.

    '''
    def grapheme_to_sequence(self, text, padding = True):
        phone_sequence = self.g2p.g2p(text)
        sequence = self.phone_to_sequence(phone_sequence, padding=padding)
        return sequence
    
    '''
    Convert phone sequence (examples: 'p|h|i|n t|h|i|m') to interger sequence:
    - text: it is text input.
    - ignore_white_space: A boolean variable to decide ignore whitespace in outputs.
    Note: If delimiter (of this instance) is None then nothing change, else the delimiter will be appended to the end phoneme of syllable.
        Examples: 
                + delimiter = 'z'
                so 'p|h|i|n t|h|i|m' will be converted to 'p|h|i|nz t|h|i|mz'. 
    - padding: If it is True, Id End of sentence character _EOS will be appended at the end the sequence and Id of Begin of sentence _BOS will be inserted at the begin of sentence.

    '''
    def phone_to_sequence(self, phone_sequence, padding = True):
        sequence = []
        for syllable in phone_sequence.split(' '):
            if self.delimiter is not None:
                syllable = syllable + self.delimiter
            for phone in syllable.split('|'):
                phone_id = self._convert_phone_to_id(phone)
                if phone_id:
                    sequence.append(phone_id)
                # if phone:
                #     try:
                #         phone_id = self.phone_to_id[phone]
                #         if phone_id:
                #             sequence.append(self.phone_to_id[phone])
                #     except KeyError:
                #         print('WARNING: phone {} is not in phone id map'.format(phone))
            if not self.ignore_white_space:
                if self.delimiter is not None:
                    phone_id = self._convert_phone_to_id(self.white_space + self.delimiter)
                    if phone_id:
                        sequence.append(phone_id)
                else:
                    phone_id = self._convert_phone_to_id(self.white_space)
                    if phone_id:
                        sequence.append(phone_id)

        if not self.ignore_white_space:
            sequence = sequence[:-1]
        if padding:
            sequence = self.pad_sequence(sequence)
        return sequence
