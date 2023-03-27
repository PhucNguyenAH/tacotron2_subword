class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def create_hparams(hparams_string=None, verbose=False):
    """Create model hyperparameters. Parse nondefault from given string."""

    hparams = AttrDict({
        ################################
        # Experiment Parameters        #
        ################################
        "epochs":1500,
        "iters_per_checkpoint":1000,
        "seed":1234,
        "dynamic_loss_scaling":True,
        "fp16_run":False,
        "distributed_run":False,
        "dist_backend":"nccl",
        "dist_url":"tcp://localhost:14897",
        "cudnn_enabled":True,
        "cudnn_benchmark":False,
        "ignore_layers":['embedding.weight'],
        # freeze_layers":['encoder'], # Freeze tacotron2 layer for finetuning

        ################################
        # Data Parameters             #
        ################################
        "load_mel_from_disk":False,
        "load_phone_from_disk":True,

        "datafiles": "data/vi_dataset",
        "training_files":'data/vi_dataset/script/train.txt',
        "validation_files":'data/vi_dataset/script/val.txt',
        "training_preprocess":'data/vi_dataset/preprocess/train.txt',
        "validation_preprocess":'data/vi_dataset/preprocess/val.txt',

        "bert_embeddings_train_path":"bert_embeddings/train",
        "bert_embeddings_val_path":"bert_embeddings/val",

        "bert_embeddings_cls_train_path":"bert_embeddings_cls/train",
        "bert_embeddings_cls_val_path":"bert_embeddings_cls/val",

        "text_cleaners":['basic_cleaners'],

        ################################
        # Audio Parameters             #
        ################################
        "max_wav_value":32768.0,
        "sampling_rate":22050,
        "filter_length":1024,
        "hop_length":256,
        "win_length":1024,
        "n_mel_channels":80,
        "mel_fmin":0.0,
        "mel_fmax":8000.0,

        ################################
        # Model Parameters             #
        ################################
        "n_symbols": 313,
        "sub_n_symbols": 5500,
        # "n_symbols": len(symbols),
        "symbols_embedding_dim":512,
        "alignloss": "",
        "attention": "StepwiseMonotonicAttention",

        # Encoder parameters
        "encoder_kernel_size":5,
        "encoder_n_convolutions":3,
        "encoder_embedding_dim":512,
        "BERT_embedding_dim": 768,

        # Decoder parameters
        "n_frames_per_step":1,  # currently only 1 is supported
        "decoder_rnn_dim":1024,
        "prenet_dim":256,
        "max_decoder_steps":1000,
        "gate_threshold":0.001,
        "p_attention_dropout":0.1,
        "p_decoder_dropout":0.1,

        # Attention parameters
        "attention_rnn_dim":1024,
        "attention_dim":128,

        # Location Layer parameters
        "attention_location_n_filters":32,
        "attention_location_kernel_size":31,

        # Mel-post processing network parameters
        "postnet_embedding_dim":512,
        "postnet_kernel_size":5,
        "postnet_n_convolutions":5,

        ################################
        # Optimization Hyperparameters #
        ################################
        "use_saved_learning_rate":True,
        "learning_rate":1e-3,
        "weight_decay":1e-6,
        "grad_clip_thresh":1.0,
        "batch_size":8, # each gpus
        "mask_padding":True  # set model's padded outputs to padded values
    })

    if hparams_string:
        hps = hparams_string[1:-2].split("-")
        for hp in hps:
            k,v = hp.split(":")
            if k in hparams:
                hparams[k] = v
                print("Set hparam: " + k + " to " + v)

    return hparams
