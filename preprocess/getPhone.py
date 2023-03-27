from tqdm import tqdm
import numpy as np
import sys, torch 
from g2p.text_to_sequence import Text2Seq

def prepare_data(text2seq, text, padding=True, device='cpu'):
    # Text Processing #
    sequence = text2seq.grapheme_to_sequence(text, padding)
    sequence = [int(seq) for seq in sequence]
    sequence = np.array(sequence)
    sequence = np.expand_dims(sequence, axis=0)
    src_pos = np.array([i + 1 for i in range(sequence.shape[1])])
    src_pos = np.stack([src_pos])
    sequence = torch.from_numpy(sequence).to(device).long()
    src_pos = torch.from_numpy(src_pos).to(device).long()
    return sequence, src_pos

def text2sequence(texts):

     config_302_be = {
          'g2p_model_path': 'g2p_resources-v2.2.5/13k_foreign_checked_011121.multi_pronunciation.g2p_model.fst',
          'g2p_config': 'g2p_resources-v2.2.5/config_phonetisaurus_v1.0.3_map3.0.0.yml',
          'phone_id_list_file': 'g2p_resources-v2.2.5/Phone_ID_Map.v3.0.0/vn_xsampa_phoneID_map_v3.0.0_011221.merge_BE',
          'delimiter': None,
          'ignore_white_space': True,
          'padding': True,
          'device':'cpu'
     }
     config = config_302_be
     text2seq = Text2Seq(config['g2p_model_path'], 
                         g2p_config=config['g2p_config'], 
                         phone_id_list_file=config['phone_id_list_file'], 
                         delimiter=config['delimiter'],
                         ignore_white_space=config['ignore_white_space']
                         )
     sequence, src_pos = prepare_data(text2seq, texts, padding=config['padding'], device=config['device'])
     return sequence

if __name__ == "__main__":
     input_file = sys.argv[1]
     output_dir = sys.argv[2]

     config_302_be = {
          'g2p_model_path': 'g2p_v3/05_vn_g2p_model.fst',
          'g2p_config': 'g2p_v3/config_phonetisaurus.vn.v3.south.yml',
          'phone_id_list_file': 'g2p_v3/Phone_ID_Map.v3.0.0/vn_xsampa_phoneID_map_v3.0.0_011221.merge_BE',
          'delimiter': None,
          'ignore_white_space': True,
          'padding': True,
          'device':'cpu'
     }
     config = config_302_be
     text2seq = Text2Seq(config['g2p_model_path'], 
                         g2p_config=config['g2p_config'], 
                         phone_id_list_file=config['phone_id_list_file'], 
                         delimiter=config['delimiter'],
                         ignore_white_space=config['ignore_white_space']
                         )

     with open(input_file, "r", encoding="utf-8") as f:
          lines = f.read().splitlines()
          for line in lines:
               p, s = line.split("|")
               sequence, src_pos = prepare_data(text2seq, s, padding=config['padding'], device=config['device'])
               save_file = output_dir + "/" + p.split("/")[-1] + ".npy"
               np.save(save_file.replace(".wav",".npy"), sequence.cpu().detach().numpy())