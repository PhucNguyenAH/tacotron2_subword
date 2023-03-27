
class Hyperparams:
    '''Hyperparameters'''


    ''' 20k vnmese v2'''
    # source_train = 'g2p_transformer/dataset/train_val_no_space.src.txt'
    # target_train = 'g2p_transformer/dataset/train_val_no_space.tgt.txt'
    # source_test = 'g2p_transformer/dataset/test_no_space.src.txt'
    # target_test = 'g2p_transformer/dataset/test_no_space.tgt.txt'

    # src_vocab = 'g2p_transformer/vocab/src_no_space.vocab.tsv'
    # tgt_vocab = 'g2p_transformer/vocab/tgt_no_space.vocab.tsv'
    # model_dir = 'g2p_transformer/model_no_space'
    # eval_result = 'g2p_transformer/result_no_space'

    src_vocab = 'resources/g2p_vocab_no_space/src_no_space.vocab.tsv'
    tgt_vocab = 'resources/g2p_vocab_no_space/tgt_no_space.vocab.tsv'
    model_dir = 'resources/g2p_model_no_space/model_epoch_84_best.pth'

    # src_vocab = 'resources/g2p_vocab/g2p_vnmese_withtone_localization.src.vocab.tsv'
    # tgt_vocab = 'resources/g2p_vocab/g2p_vnmese_withtone_localization.tgt.vocab.tsv'
    # model_dir = 'resources/g2p_model/model_epoch_21.pth'
    # eval_result = 'task_g2p_vnmese_withtone_localization_sampa_v2/result'

    # training
    batch_size = 64 # alias = N
    lr = 0.0001 # learning rate. In paper, learning rate is adjusted to the global step.
    logdir = 'g2p_transformer/logdir_no_space' # log directory

    # model
    maxlen = 30 # Maximum number of words in a sentence. alias = T.
                # Feel free to increase this if you are ambitious.
    min_cnt = 1 # words whose occurred less than min_cnt are encoded as <UNK>.
    hidden_units = 512 # alias = C
    num_blocks = 6 # number of encoder/decoder blocks
    num_epochs = 100
    num_heads = 8
    dropout_rate = 0.1
    sinusoid = False # If True, use sinusoid. If false, positional embedding.
    eval_epoch = 84  # epoch of model for eval
    preload = None
    use_gpu = True
