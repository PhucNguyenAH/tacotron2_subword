import requests
import sys
from tqdm import tqdm

def norm_sent(sent,api_norm):
    data = {"paragraph": sent} #norm tiếng Việt
    # data = {"paragraph": sent, "lang":"en"} #norm tiếng Anh
    # result = requests.post(api_norm, json = data) 
    result = requests.post(api_norm, json = data, auth=('speech_oov', '4D6$&%9qeEhvRTeR'))

    print(result)
    
    wr_words = result.json().get("wr_words")
    sp_words = result.json().get("sp_words")
    normed_list = result.json().get("normed_sents")
    output = " ".join(normed_list)
    print(f'\nTagging: {wr_words} \n')
    print(f'Expansion: {sp_words} \n')
    print(f'Final: {output} \n')
    return output


def norm_doc(path, api_norm):
    normed_list = []
    with open(path,mode="r", encoding='utf8') as f:
        for sent in f:
            normed_sent = norm_sent(sent, api_norm)
            normed_list.append(normed_sent)
    with open(path + "_norm", mode="w", encoding = "utf8") as f:
        for ele in normed_list:
            f.write(f'{ele}\n')
    print(f"Done normalizing document: {path}")

if __name__ == "__main__":
    api_norm_v1='http://10.124.68.83:5041/internal_api/v1/tts_norm' #(may 83)
    # api_norm_v2="http://10.254.148.20:5042/internal_api/v2/tts_norm" #(may 20)
    api_norm_v2="http://123.16.13.79:8080/fasttts/api/internal_api/v2/tts_norm"

    input_file = sys.argv[1]
    output_file = sys.argv[2]

    fw = open(output_file,"w+", encoding="utf-8")

    with open(input_file, "r", encoding="utf-8") as f:
        lines = f.read().splitlines()
        for line in tqdm(lines):
            p, s = line.split("|")
            s = norm_sent(s,api_norm_v2) 
            fw.write(p + "|" + s + "\n")
    fw.close()
