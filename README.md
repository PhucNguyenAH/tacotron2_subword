# Tacotron 2

## Environment

1. Clone this repo: `git clone https://github.com/PhucNguyenAH/tacotrin2_subword.git`

2. CD into this repo: `cd tacotron2`

3. Create conda env: 

`conda create -n taco`

`conda activate taco`

Install pip in conda:

`conda install pip`

4. (Cuda 11.1) Install Pytorch and Cudnn: 

`python -m pip install torch==1.8.0+cu111 torchvision==0.9.0+cu111 torchaudio==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html`

For another version of CUDA, check this: https://pytorch.org/get-started/previous-versions/

5. Install python requirements: `python -m pip install -r requirements.txt`

## Prepare data

Using ForceAlign Result for training, require two folder:

- <wav folder> include audios wav in sample rate 22050

- <durations folder> which is the force alignment results using an external aligner such as MTA

- <training file> with structure: <wav file path> | <durations file path>

- <validation file> with structure the same as training file, example: 

```/data/S_F02_HoaiAnh_v3.0.2/0007_02.wav|/data/S_F02_HoaiAnh_v3.0.2/PhoneID_BE/durations/0007_02.npy```
## Training

1. `python train.py --output_directory=Outdir --log_directory=logdir --train_file <training file> --val_file <validation file> --n_symbols <phones numbers>`

2. `tensorboard --logdir Outdir/logdir`

Example: (For HoaiAnh 3.0.2 n_symbols is 313 for BE and 165 for merge all)

`python train.py --output_directory=Outdir --log_directory=logdir --train_file DATA/train.txt --val_file DATA/val.txt --n_symbols 313`

## Inference from text script

1. Prepare a <testing file> with structure:

<script_id> <tab> <script>

2. Config G2P path in * function "text2sequence" * in inference.py file

Example

```
config_302_be = {
          'g2p_model_path': '13k_foreign_checked_011121.multi_pronunciation.g2p_model.fst',
          'g2p_config': 'config_phonetisaurus_v1.0.3_map3.0.0.yml',
          'phone_id_list_file': 'Phone_ID_Map.v3.0.0/vn_xsampa_phoneID_map_v3.0.0_011221.merge_BE',
          'delimiter': None,
          'ignore_white_space': True,
          'padding': True,
          'device':'cpu'
     }

config = config_302_be
```

3. Inference:

`python inference.py <testing file> <taco_checkpoint> <hifigan_checkpoint>`

Output melspectrograms, audios, alignments, stored at:

["Outdir/demo/alignment","Outdir/demo/mels","Outdir/demo/audio"]

Example:

`python inference.py text_hoaianh.norm 06_tacotron/checkpoint_101000 05_hifigan/finetune/g_02500000`

## Extract GTA for Vocoder finetuning

`python GTA.py <training file> <taco_checkpoint> <folder_path_to_save_mel>`

- <training file> with structure: <wav file path> | <durations file path>

Example:

`python GTA.py DATA/train.txt 06_tacotron/checkpoint_101000 DATA/mel`