
import codecs
import os

def norm_vnmese_accent(str):
   words = str.split(' ')
   for i in range(len(words)):
       if len(words[i]) <= 3:
           if not words[i].startswith('qu'):
               words[i] = words[i].replace("uỳ", "ùy")
               words[i] = words[i].replace("uý", "úy")
               words[i] = words[i].replace("uỷ", "ủy")
               words[i] = words[i].replace("uỹ", "ũy")
               words[i] = words[i].replace("uỵ", "ụy")
           else:
               words[i] = words[i].replace("ùy", "uỳ")
               words[i] = words[i].replace("úy", "uý")
               words[i] = words[i].replace("ủy", "uỷ")
               words[i] = words[i].replace("ũy", "uỹ")
               words[i] = words[i].replace("ụy", "uỵ")
           words[i] = words[i].replace("oà", "òa")
           words[i] = words[i].replace("oá", "óa")
           words[i] = words[i].replace("oả", "ỏa")
           words[i] = words[i].replace("oã", "õa")
           words[i] = words[i].replace("oạ", "ọa")
           words[i] = words[i].replace("oè", "òe")
           words[i] = words[i].replace("oé", "óe")
           words[i] = words[i].replace("oẻ", "ỏe")
           words[i] = words[i].replace("oẽ", "õe")
           words[i] = words[i].replace("oẹ", "ọe")
       else:
           words[i] = words[i].replace("òa", "oà")
           words[i] = words[i].replace("óa", "oá")
           words[i] = words[i].replace("ỏa", "oả")
           words[i] = words[i].replace("õa", "oã")
           words[i] = words[i].replace("ọa", "oạ")
           words[i] = words[i].replace("òe", "oè")
           words[i] = words[i].replace("óe", "oé")
           words[i] = words[i].replace("ỏe", "oẻ")
           words[i] = words[i].replace("õe", "oẽ")
           words[i] = words[i].replace("ọe", "oẹ")
           
   return ' '.join(words)



def load_phone_lexicon(full_path, assert2fields = False, value_processor = None):
    phones = []
    if value_processor is None:
        value_processor = lambda x: x[0]
    lex = {}
    with codecs.open(full_path, mode='r', encoding='utf-8-sig') as f:
        for line in f:
            # line = norm_vnmese_accent(line)
            #print('LINE: {}'.format(line))
            # line = unicodedata.normalize("NFC", line)
            parts = line.strip().split()
            if assert2fields:
                assert(len(parts) == 2)
            lex[parts[0]] = value_processor(parts[1:])

            for p in parts[1:]:
                if p not in phones:
                    phones.append(p)
    return lex, phones

def load_lexicon(full_path, assert2fields = False, value_processor = None):
    if value_processor is None:
        value_processor = lambda x: x[0]
    lex = {}
    with codecs.open(full_path, mode='r', encoding='utf-8-sig') as f:
        for line in f:
            #line = norm_vnmese_accent(line)
            #print('LINE: {}'.format(line))
            # line = unicodedata.normalize("NFC", line)
            parts = line.strip().split()
            if assert2fields:
                assert(len(parts) == 2)
            lex[parts[0]] = value_processor(parts[1:])
        
    return lex
def build_phone_lexicon(vi_lexicon_file, en_lexicon_file, foreign_lexicon_file):
    viLex, viPhones = load_lexicon(vi_lexicon_file, value_processor = lambda x: " ".join(x))
    #print('viPhones: {} len: {}'.format(viPhones, len(viPhones)))
    enLex, enPhones = load_lexicon(en_lexicon_file, value_processor = lambda x: " ".join(x))
    foreignLex, foreignPhones = load_lexicon(foreign_lexicon_file, value_processor = lambda x: " ".join(x))
    
    phones = [] #viPhones + enPhones + foreignPhones
    phones.extend(x for x in viPhones if x not in phones)
    phones.extend(x for x in enPhones if x not in phones)
    phones.extend(x for x in foreignPhones if x not in phones)

    lexicon = (viLex, enLex, foreignLex)
    return phones, lexicon
def build_lexicon(vi_lexicon_file, en_lexicon_file, foreign_lexicon_file):
    viLex = load_lexicon(vi_lexicon_file, value_processor = lambda x: " ".join(x))
    enLex = load_lexicon(en_lexicon_file, value_processor = lambda x: " ".join(x))
    foreignLex = load_lexicon(foreign_lexicon_file, value_processor = lambda x: " ".join(x))
    lexicon = (viLex, enLex, foreignLex)
    return lexicon

'''
Load phone_id_list from lexicon 
- lexicon: Input lexicon
- phone_id_list_file: If this file is exist, phone_id_list will be directly loaded from this file.
- other_symbols: The symbols are not in lexicon.
- dump: Boolean variable to decide whether dump phone_id_list to file or not.
- phone_delimiter: If this variable is None then nothing change, else phone_delimiter will be appended to the end phone of syllable.
    + For examples: phone_delimiter = 'z' then 'a|n|h' will be converted into 'a|n|hz'
'''
def load_phone_id_list(lexicon, phone_id_list_file, other_symbols = [], dump=True, phone_delimiter = None):
    phones = []
    if not os.path.isfile(phone_id_list_file):
        print('WARNING: phone id list file {} is not exists, build phone_id_list from lexicon instead!')
        for lex in lexicon:
            for key, value in lex.items():
                # if phone_delimiter is not None:
                #     value = value + phone_delimiter
                phone_list = value.split(' ')
                if len(phone_list) < 1:
                    raise RuntimeError('No phone found: word {}, phones {}'.format(key, value))
                for phone in phone_list:
                    if phone == '':
                        raise RuntimeError('Wrong phone: word {}, phones {}'.format(key, value))
                    if phone not in phones:
                        print(phone)
                        phones.append(phone)
        phones = sorted(phones)
        
        #simply duplicate all of phoneme by appending phone_delimiter
        if phone_delimiter is not None:
            extend_phones = [phone + phone_delimiter for phone in phones]
            extend_other_symbols = [symbol + phone_delimiter for symbol in other_symbols if symbol]
            phones =extend_other_symbols + phones + extend_phones
        else:
            extend_other_symbols = [symbol for symbol in other_symbols if symbol ]
            phones = extend_other_symbols + phones
        phone_to_id = {s: i for i, s in enumerate(phones)}
        id_to_phone = {i: s for i, s in enumerate(phones)}
        if dump:
            with open(phone_id_list_file, 'w') as fdump:
                for phone in phones:
                    fdump.write(phone + '\t' + str(phone_to_id[phone]) + '\n')
    else:
        with open(phone_id_list_file, 'r') as f:
            phone_to_id = {}
            id_to_phone = {}
            for line in f:
                values = line.rstrip().split('\t')
                #print('values: {}'.format(values))
                #key phoneme is whitespace
                # if len(values) == 1 and line[0] == ' ':
                #     idphone = values[0]
                #     phone = ' '
                # else:
                phone, idphone = values
                if phones and (phone in phones):
                    raise ValueError('Duplicate phone in phone id list')
                phone_to_id[phone] = idphone
                id_to_phone[idphone] = phone
                
    return phone_to_id, id_to_phone
def load_character_id_list(letters, other_symbols = []):
    symbols = list(letters) + other_symbols
    symbol_to_id = {s: i for i, s in enumerate(symbols)}
    id_to_symbol = {i: s for i, s in enumerate(symbols)}
    return symbol_to_id, id_to_symbol
        
