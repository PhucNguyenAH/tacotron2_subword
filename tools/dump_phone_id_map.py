import sys
import os
import yaml
import argparse
sys.path.insert(0, '..')

from resources import resources_dir
from g2p.lexicon import build_lexicon, load_phone_id_list

if __name__ == '__main__':
    '''
        Usage: python dump_phone_id_map.py -c $config -p $phone_id_map_file --other_symbols=pau
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config_file', type=str,
                        help='G2p configuration file', required=True)
    parser.add_argument('-p', '--phone_id_map_file', type=str,
                        help='Output phone id map file', required=True)
    parser.add_argument('-d', '--delimiter', type=str,
                        help='If this variable is None then nothing change, else phone_delimiter will be appended to the end phone of syllable', default=None)
    parser.add_argument('--other_symbols', type=str,
                        help='Other symbol which added to phone id map. List of symbol separated by comma', default='spau,mpau,lpau,slpau')
    args = parser.parse_args()
    config_file = args.config_file
    phone_id_map_file = args.phone_id_map_file
    other_symbols = args.other_symbols
    other_symbols = other_symbols.split(',')
    delimiter = args.delimiter

    with open(config_file) as f:
        config = yaml.load(f, Loader=yaml.Loader)

    if config['resources']['load_default']:
        vi_lex_file = os.path.join(
            resources_dir, config['resources']['vi_lexicon_file'])
        en_lex_file = os.path.join(
            resources_dir, config['resources']['en_lexicon_file'])
        foreign_lex_file = os.path.join(
            resources_dir, config['resources']['foreign_lexicon_file'])
    else:
        vi_lex_file = config['resources']['vi_lexicon_file']
        en_lex_file = config['resources']['en_lexicon_file']
        foreign_lex_file = config['resources']['foreign_lexicon_file']
    lexicon = build_lexicon(vi_lex_file, en_lex_file, foreign_lex_file)
    
    # Add specific symbol
    other_symbols.append(config['t2s']['special'])
    other_symbols.append(config['t2s']['pad'])
    other_symbols.append(config['t2s']['EOS'])
    other_symbols.append(config['t2s']['BOS'])
    other_symbols.append(config['t2s']['white_space'])
    other_symbols = other_symbols + list(config['g2p']['punctuation'])
    if os.path.isfile(phone_id_map_file):
        print('Warining File exists: {}. This file will be removed'.format(phone_id_map_file))
        os.remove(phone_id_map_file)

    load_phone_id_list(lexicon, phone_id_map_file,
                       other_symbols=other_symbols, dump=True, phone_delimiter=delimiter)
