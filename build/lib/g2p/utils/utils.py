import os
from collections import Counter
from string import punctuation
import pandas as pd
import csv
import re
import codecs
import string
import time

punct_dict = {';':'.', ':':',', '!' : '.'}
special_char_dict_v2 = {',':'COMMA', '.':'PERIOD', '?':'QUESTION'}

def current_millis_time():
    return int(round(time.time() * 1000))

def make_vocab(fpath, fname):
    text = open(fpath, 'r').read()
    words = text.split()
    word2cnt = Counter(words)
    # if not os.path.exists('vocab'):
    #     os.mkdir('vocab')
    with open('{}'.format(fname), 'w') as fout:
        fout.write("{}\t1000000\n{}\t1000000\n{}\t1000000\n{}\t1000000\n".format("<pad>", "<unk>", "<s>", "</s>"))
        for word, cnt in word2cnt.most_common(len(word2cnt)):
            fout.write("{}\t{}\n".format(word, cnt))
        # fout.write("{}\t{}\n".format('_', '100'))


def process_g2p_english(f_file):
    graphes, phonemes = list(), list()
    fo = open(f_file, 'r')
    count = 0

    for line in fo:
        count += 1
        if count % 1000:
            print('count = %d' % count)
        graph = line.split(' ')[0]
        phoneme = ' '.join(line.split(' ')[1:])

        if graph[0] in punctuation:
            graph = graph[1:]

        graph = re.sub('\(1\)', '', graph)
        graph = re.sub('\(2\)', '', graph)
        graph = re.sub('\(3\)', '', graph)
        graph = re.sub('\(HUE\)', '', graph)
        graph = re.sub('\(NAM\)', '', graph)
        if graph not in graphes:
            graphes.append(graph)
            phonemes.append(phoneme)

    return graphes, phonemes


def process_g2p_vnmese(f_file):
    graphes, phonemes = list(), list()
    fo = open(f_file, 'r')
    count = 0
    for line in fo:
        count += 1
        if count % 1000:
            print('count = %d' % count)
        graph = line.split(' ', 1)[0].strip()
        phoneme = line.split(' ', 1)[1].strip()
        phoneme = phoneme.replace(' ', ' $ ').replace('|', ' | ')

        if graph in punctuation:
            continue

        graph = re.sub('\(1\)', '', graph)
        graph = re.sub('\(2\)', '', graph)
        graph = re.sub('\(3\)', '', graph)
        graph = re.sub('\(HUE\)', '', graph)
        graph = re.sub('\(NAM\)', '', graph)

        # graph_ch = list()
        # for c in list(graph):
        #     if c in punctuation:
        #         graph_ch.append(' ')
        #     else: graph_ch.append(c)
        # graph = ''.join(graph_ch)

        if graph not in graphes:
            graphes.append(graph)
            phonemes.append(phoneme)
    return graphes, phonemes


def save_list(l, file):
    fo = open(file, 'w')
    for e in l:
        fo.write(e.strip() + '\n')


def preprocess(text):
    vnese_lower = 'aáàảãạăắằẳẵặâấầẩẫậeéèẻẽẹêếềểễệiíìỉĩịoóòỏõọôốồổỗộơớờởỡợuúùủũụưứừửữựyýỳỷỹỵdđ'
    vnese_upper = 'AÁÀẢÃẠĂẮẰẲẴẶÂẤẦẨẪẬEÉÈẺẼẸÊẾỀỂỄỆIÍÌỈĨỊOÓÒỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢUÚÙỦŨỤƯỨỪỬỮỰYÝỲỶỸỴDĐ'
    text = re.sub('\([a-zA-Z_' +vnese_lower +vnese_upper + ']*\)', '', text)
    return text


def load_g2p(f_g2p):
    fo = open(f_g2p, 'r')
    lines = list()
    for line in fo:
        input_text = ' '.join(list(line.split()[0]))
        target_text = ' '.join(line.split()[1:])
        lines.append(input_text + '\t' + target_text)

    return lines

def load_g2p_english(graph_file, phoneme_file):
    lines = list()
    f_graph = open(graph_file, 'r')
    f_phoneme = open(phoneme_file, 'r')
    graphes = f_graph.readlines()
    phonemes = f_phoneme.readlines()
    for i in range(len(graphes)):
        line = graphes[i] + '\t' + phonemes[i]
        lines.append(line)

    return lines

def load_foreign_words(f_foreign):
    csv_writer = csv.writer(open('sample_synthesize.csv', 'w'), delimiter='\t')
    csv_writer.writerow(['written', 'spoken'])
    df = pd.read_csv(f_foreign)
    words_foreign = df.word.values.tolist()
    trans_foreign = df.transcription.values.tolist()
    lines = list()
    for i in range(len(words_foreign)):
        input_text = str(words_foreign[i]).lower()
        target_text = str(trans_foreign[i]).lower()
        target_text = target_text.replace('-','_')
        target_text = target_text.replace(' ', '$')
        target_text = target_text.replace('_', ' ')
        target_text = target_text.replace('$', ' _ ')
        input_text = ' '.join(preprocess(input_text))
        target_text = preprocess(target_text)
        # input_words = input_text.split()
        # target_words = target_text.split()
        # if len(input_words) == len(target_words):
        line = input_text + '\t' + target_text
        lines.append(line)

    return lines

def load_foreign_words_v2(f_foreign):
    df = pd.read_csv(f_foreign)
    words_foreign = df.word.values.tolist()
    trans_foreign = df.transcription.values.tolist()
    lines = list()
    line_errors = list()
    for i in range(len(words_foreign)):
        input_text = str(words_foreign[i]).lower()
        target_text = str(trans_foreign[i]).lower()
        target_text = target_text.replace('-','_')
        # target_text = target_text.replace(' ', '$')
        # target_text = target_text.replace('_', ' ')
        # target_text = target_text.replace('$', ' _ ')
        input_text = preprocess(input_text)
        target_text = preprocess(target_text)
        input_words = input_text.split()
        target_words = target_text.split()
        if len(input_words) == len(target_words):
            for j in range(len(input_words)):
                target_words[j] = target_words[j].replace('_', ' ')
                line = ' '.join(list(input_words[j])) + '\t' + target_words[j]
                if line not in lines:
                    lines.append(line)
        else:
            target_text = target_text.replace('-','_')
            line_errors.append(input_text + '\t' + target_text)
            lines.append(' '.join(list(input_text)) + '\t' + target_text)

    return lines, line_errors

def load_csv_2_cols(csv_file, col1, col2):
    df = pd.read_csv(csv_file, delimiter='\t')
    col1_data = df[col1].values.tolist()
    col2_data = df[col2].values.tolist()
    lines = list()
    for i in range(len(col1_data)):
        lines.append(str(col1_data[i]) + '\t' + str(col2_data[i]))
    return lines

def load_csv_3_cols(csv_file, col1, col2, col3):
    df = pd.read_csv(csv_file, names=[col1, col2, col3], delimiter='\t')
    line_col1 = df[col1].values.tolist()
    line_col2 = df[col2].values.tolist()
    line_col3 = df[col3].values.tolist()
    lines = list()
    for i in range(len(line_col1)):
        lines.append(str(line_col1[i]) + ' ' + str(line_col2[i]) + '\t' + str(line_col3[i]))
    return lines

def load_seq2seq(csv_file, tag_col, src_col, tgt_col):
    df = pd.read_csv(csv_file, delimiter='\t')
    tag = df[tag_col].values.tolist()
    src = df[src_col].values.tolist()
    tgt = df[tgt_col].values.tolist()
    lines = list()
    for i in range(len(src)):
        # lines.append(str(tag[i].strip()) + ' ' + str(src[i]) + '\t' + str(tgt[i]))
        lines.append(str(tag[i].strip()) + ' ' + str(tgt[i]) + '\t' + str(src[i]))
    return lines

def save_data(lines, type, path):
    src = list()
    tgt = list()
    f_src = open(path + type + '.src.txt', 'w')
    f_tgt = open(path + type + '.tgt.txt', 'w')
    for line in lines:
        pair = line.split('\t')
        f_src.write(pair[0].strip() + '\n')
        f_tgt.write(pair[1].strip() + '\n')
        src.append(pair[0])
        tgt.append(pair[1])

def save_file(lines, path, f_name):
    f_out = open(path + f_name, 'w')
    for line in lines:
        f_out.write(line + '\n')

def word2char(f_word, f_char):
    fo = open(f_word, 'r')
    fw = open(f_char, 'w')
    for line in fo:
        fw.write(' '.join(list(line)))

def save_csv(l1, l2, csv_file):
    csv_writer = csv.writer(open(csv_file, 'w'), delimiter='\t')
    csv_writer.writerow(['graph_word', 'graph_char', 'phoneme'])
    l1_char = list()
    for e in l1:
        l1_char.append(' '.join(list(e.strip())))
    for i in range(len(l1)):
        csv_writer.writerow([l1[i], l1_char[i], l2[i].strip()])

def revert_norm_word(str):
    words = str.split()
    for i in range(1, len(words)):
        if words[i] in special_char_dict_v2.keys():
            words[i - 1] = words[i - 1] + words[i]
            words[i] = ''

    return ' '.join(words)

# remove punctuation exclude .,?
def remove_puntuation(str):
    tokens = list()
    for token in str.split():
        if token in punct_dict.keys():
            tokens.append(punct_dict[token])
        elif token in special_char_dict_v2.keys():
            tokens.append(token)
        elif token in punctuation:
            tokens.append('')
        else:
            tokens.append(token)

    input_str = ' '.join(tokens)
    input_str = re.sub('\.+\s*[\.*|,*|\?*]', '.', input_str)
    input_str = re.sub(',+\s*[\.*|,*|\?*]', ',', input_str)
    input_str = re.sub('\?+\s*[\.*|,*|\?*]', ',', input_str)
    return input_str



# manually tokenize
def norm_word(word):
    char = list(word)
    pre_is_word = False
    pre_is_punct = False
    for i in range(len(char)):
        if char[i] in punctuation:
            if pre_is_word:
                char[i] = ' ' + char[i]
            else:
                char[i] = char[i] + ' '
            pre_is_word = False
            pre_is_punct = True
        else:
            if pre_is_punct:
                char[i] = ' ' + char[i]
            pre_is_word = True
            pre_is_punct = False

    return ''.join(char)

def replace_multi_period(str):
    str = re.sub('(\.+\s+)+\.*', ' . ', str)
    str = re.sub('\.+', ' . ', str)
    return str

def replace_multi_space(str):
    str = re.sub(' +', ' ', str)
    return str

def save_csv_v2(src, tgt, csv_file):
    assert len(src) == len(tgt), '[Warning] The training instance count is not equal.'
    csv_writer = csv.writer(open(csv_file, 'w'), delimiter='\t')
    csv_writer.writerow(['src', 'tgt'])
    for i in range(len(src)):
        csv_writer.writerow([src[i], tgt[i]])

def read_lines(f_file):
    fo = open(f_file, 'r')
    return fo.readlines()

def read_csv(f_csv, cols):
    df = pd.read_csv(f_csv, delimiter='\t')
    lines = list()
    for i in range(len(cols)):
        lines.append(df[cols[i]].values.tolist())

    return lines