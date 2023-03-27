import os, glob
import time
import argparse
import math
from numpy import finfo
from tqdm import tqdm
import torch
from distributed import apply_gradient_allreduce
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
import shutil
from model import BERT_Tacotron2
from data_utils import DataLoader, collate_fn
from data_utils import BERTTacotron2Dataset
from loss_function import Tacotron2Loss
from logger import Tacotron2Logger
from hparams import create_hparams

device = torch.device('cuda'if torch.cuda.is_available()else 'cpu')
minLoss = float('inf')

def reduce_tensor(tensor, n_gpus):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.reduce_op.SUM)
    rt /= n_gpus
    return rt


def init_distributed(hparams, n_gpus, rank, group_name):
    assert torch.cuda.is_available(), "Distributed mode requires CUDA."
    print("Initializing Distributed")

    # Set cuda device so everything is done on the right GPU.
    torch.cuda.set_device(rank % torch.cuda.device_count())

    # Initialize distributed communication
    dist.init_process_group(
        backend=hparams.dist_backend, init_method=hparams.dist_url,
        world_size=n_gpus, rank=rank, group_name=group_name)

    print("Done initializing distributed")


def prepare_dataloaders(hparams):
    # Get data, data loaders and collate function ready
    trainset = BERTTacotron2Dataset(dataset_path="train", text_path=hparams.training_preprocess, embedding_path=hparams.bert_embeddings_train_path, embedding_cls_path=hparams.bert_embeddings_cls_train_path)
    valset = BERTTacotron2Dataset(dataset_path="val", text_path=hparams.validation_preprocess, embedding_path=hparams.bert_embeddings_val_path, embedding_cls_path=hparams.bert_embeddings_cls_val_path)

    if hparams.distributed_run:
        train_sampler = DistributedSampler(trainset)
        shuffle = False
    else:
        train_sampler = None
        shuffle = True

    train_loader = DataLoader(trainset, num_workers=8, shuffle=shuffle,
                                sampler=train_sampler,
                                batch_size=hparams.batch_size, pin_memory=False,
                                drop_last=True, collate_fn=collate_fn)
    return train_loader, valset, collate_fn


def prepare_directories_and_logger(output_directory, log_directory, rank):
    if rank == 0:
        if not os.path.isdir(output_directory):
            os.makedirs(output_directory, exist_ok=True)
            os.chmod(output_directory, 0o775)
        logger = Tacotron2Logger(os.path.join(output_directory, log_directory))
    else:
        logger = None
    return logger


def load_model(hparams):
    model = BERT_Tacotron2(hparams).cuda()
    if hparams.fp16_run:
        model.decoder.attention_layer.score_mask_value = finfo('float16').min

    if hparams.distributed_run:
        model = apply_gradient_allreduce(model)

    return model


def warm_start_model(checkpoint_path, model, ignore_layers):
    assert os.path.isfile(checkpoint_path)
    print("Warm starting model from checkpoint '{}'".format(checkpoint_path))
    checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')
    model_dict = checkpoint_dict['state_dict']
    if len(ignore_layers) > 0:
        model_dict = {k: v for k, v in model_dict.items()
                      if k not in ignore_layers}
        dummy_dict = model.state_dict()
        dummy_dict.update(model_dict)
        model_dict = dummy_dict
    model.load_state_dict(model_dict)
    return model


def load_checkpoint(checkpoint_path, model, optimizer):
    assert os.path.isfile(checkpoint_path)
    print("Loading checkpoint '{}'".format(checkpoint_path))
    checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint_dict['state_dict'])
    optimizer.load_state_dict(checkpoint_dict['optimizer'])
    learning_rate = checkpoint_dict['learning_rate']
    iteration = checkpoint_dict['iteration']
    if "val_loss" in checkpoint_dict.keys():
        minLoss = checkpoint_dict['val_loss']
    print("Loaded checkpoint '{}' from iteration {}" .format(
        checkpoint_path, iteration))
    return model, optimizer, learning_rate, iteration


def save_checkpoint(model, optimizer, learning_rate, iteration, val_loss, filepath):
    print("Saving model and optimizer state at iteration {} to {}".format(
        iteration, filepath))
    torch.save({'iteration': iteration,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'val_loss': val_loss,
                'learning_rate': learning_rate}, filepath)


def validate(model, criterion, valset, iteration, batch_size, n_gpus,
             collate_fn, logger, distributed_run, rank):
    """Handles all the validation scoring and printing"""
    model.eval()
    with torch.no_grad():
        val_sampler = DistributedSampler(valset) if distributed_run else None
        val_loader = DataLoader(valset, sampler=val_sampler, num_workers=8,
                                shuffle=False, batch_size=batch_size,
                                pin_memory=False, collate_fn=collate_fn)

        val_loss = 0.0
        total_step = 0
        for i, batchs in enumerate(val_loader):
            for j, data_of_batch in enumerate(batchs):
                # Get Data
                character = torch.from_numpy(
                    data_of_batch["text"]).long().to(device)
                mel_target = torch.from_numpy(data_of_batch["mel_target"]).float().to(
                    device).contiguous().transpose(1, 2)
                stop_target = torch.from_numpy(
                    data_of_batch["stop_token"]).float().to(device)
                embeddings = torch.from_numpy(data_of_batch["bert_embeddings"]).long().to(
                    device)
                phoneme_embeddings_cls = data_of_batch["phoneme_embeddings_cls"].float().to(
                    device)
                bert_embeddings_cls = data_of_batch["bert_embeddings_cls"].float().to(
                    device)
                input_lengths = torch.from_numpy(
                    data_of_batch["length_text"]).long().to(device)
                input_lengths_bert = torch.from_numpy(
                    data_of_batch["length_bert"]).long().to(device)
                output_lengths = torch.from_numpy(
                    data_of_batch["length_mel"]).long().to(device)
                align = torch.from_numpy(
                    data_of_batch["align"]).long().to(device)

                # Forward
                batch = character, input_lengths, input_lengths_bert, mel_target, stop_target, output_lengths, embeddings, phoneme_embeddings_cls, bert_embeddings_cls, align
                x, y = model.parse_batch(batch)
                y_pred = model(x)
                loss, _, _, _, _ = criterion(y_pred, y, x)
                if distributed_run:
                    reduced_val_loss = reduce_tensor(loss.data, n_gpus).item()
                else:
                    reduced_val_loss = loss.item()
                val_loss += reduced_val_loss
            total_step += j
        # val_loss = val_loss / (i + 1)
        val_loss = val_loss / total_step

    model.train()
    if rank == 0:
        print("Validation loss {}: {:9f}  ".format(iteration, val_loss))
        logger.log_validation(val_loss, model, y, y_pred, iteration)
    return val_loss

def scan_checkpoint(cp_dir):
    cp_list = glob.glob(cp_dir+"/checkpoint_*")
    if len(cp_list) == 0:
        return None
    return sorted(cp_list, key=lambda x: (len(x),x))[-1]

def train(output_directory, log_directory, checkpoint_path, warm_start, n_gpus, rank, group_name, hparams):
    global minLoss
    scheduler_learningrate = False
    """Training and validation logging results to tensorboard and stdout

    Params
    ------
    output_directory (string): directory to save checkpoints
    log_directory (string) directory to save tensorboard logs
    checkpoint_path(string): checkpoint path
    n_gpus (int): number of gpus
    rank (int): rank of current gpu
    hparams (object): comma separated list of "name=value" pairs.
    """
    if hparams.distributed_run:
        init_distributed(hparams, n_gpus, rank, group_name)

    torch.manual_seed(hparams.seed)
    torch.cuda.manual_seed(hparams.seed)

    model = load_model(hparams)
    learning_rate = hparams.learning_rate
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,
                                 weight_decay=hparams.weight_decay)

    if hparams.fp16_run:
        from apex import amp
        model, optimizer = amp.initialize(
            model, optimizer, opt_level='O2')

    if hparams.distributed_run:
        model = apply_gradient_allreduce(model)

    criterion = Tacotron2Loss(hparams.alignloss)

    logger = prepare_directories_and_logger(
        output_directory, log_directory, rank)

    if hparams.distributed_run:
        # train_loader, valset, collate_fn = prepare_dataloaders(hparams)
        # Get data, data loaders and collate function ready
        trainset = BERTTacotron2Dataset(dataset_path="train", text_path=hparams.training_preprocess, embedding_path=hparams.bert_embeddings_train_path, embedding_cls_path=hparams.bert_embeddings_cls_train_path)
        valset = BERTTacotron2Dataset(dataset_path="val", text_path=hparams.validation_preprocess, embedding_path=hparams.bert_embeddings_val_path, embedding_cls_path=hparams.bert_embeddings_cls_val_path)
        if hparams.distributed_run:
            train_sampler = DistributedSampler(trainset)
            shuffle = False
        else:
            train_sampler = None
            shuffle = True
        train_loader = DataLoader(trainset, num_workers=16, shuffle=shuffle,
                                sampler=train_sampler,
                                batch_size=hparams.batch_size, pin_memory=False,
                                drop_last=True, collate_fn=collate_fn)
    else:
        train_loader, valset, collate_fn = prepare_dataloaders(hparams)

    # Load checkpoint if one exists
    iteration = 0
    epoch_offset = 0

    checkpoint_path = scan_checkpoint(output_directory)

    if checkpoint_path is not None:
        print("Use pretrain: ", checkpoint_path)
        if warm_start:
            model = warm_start_model(
                checkpoint_path, model, hparams.ignore_layers)
        else:
            model, optimizer, _learning_rate, iteration = load_checkpoint(
                checkpoint_path, model, optimizer)
            if hparams.use_saved_learning_rate:
                learning_rate = _learning_rate
            iteration += 1  # next iteration is iteration + 1
            train_loader_tmp = next(iter(train_loader))
            epoch_offset = max(0, int(iteration / (len(train_loader_tmp)*len(train_loader))))
            print("Start epoch:", epoch_offset)

    if scheduler_learningrate:
        try:
            scheduler_lr = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.999, last_epoch=epoch_offset)
        except:
            scheduler_lr = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.999, last_epoch=-1)


    model.train()
    is_overflow = False
    
    # ================ MAIN TRAINNIG LOOP! ===================
    for epoch in range(epoch_offset, hparams.epochs):

        if scheduler_learningrate:
            print("Current learning rate: ",scheduler_lr.get_last_lr())
            learning_rate = scheduler_lr.get_last_lr()[-1]

        print("Epoch: {}".format(epoch))

        if hparams.distributed_run:
            train_sampler.set_epoch(epoch)

        for i, batchs in enumerate(train_loader):
            for j, data_of_batch in enumerate(batchs):
                start = time.perf_counter()
                for param_group in optimizer.param_groups:
                    param_group['lr'] = learning_rate

                model.zero_grad()
                # Get Data
                character = torch.from_numpy(
                    data_of_batch["text"]).long().to(device)
                mel_target = torch.from_numpy(data_of_batch["mel_target"]).float().to(
                    device).contiguous().transpose(1, 2)
                stop_target = torch.from_numpy(
                    data_of_batch["stop_token"]).float().to(device)
                embeddings = torch.from_numpy(data_of_batch["bert_embeddings"]).long().to(
                    device)
                phoneme_embeddings_cls = data_of_batch["phoneme_embeddings_cls"].float().to(
                    device)
                bert_embeddings_cls = data_of_batch["bert_embeddings_cls"].float().to(
                    device)
                input_lengths = torch.from_numpy(
                    data_of_batch["length_text"]).long().to(device)
                input_lengths_bert = torch.from_numpy(
                    data_of_batch["length_bert"]).long().to(device)
                output_lengths = torch.from_numpy(
                    data_of_batch["length_mel"]).long().to(device)
                align = torch.from_numpy(
                    data_of_batch["align"]).long().to(device)
                # Forward
                batch = character, input_lengths, input_lengths_bert, mel_target, stop_target, output_lengths, embeddings, phoneme_embeddings_cls, bert_embeddings_cls, align
                x, y = model.parse_batch(batch)
                y_pred = model(x)

                loss, mel_loss, gate_loss, align_loss, align_bert_loss = criterion(y_pred, y, x, iteration)

                if hparams.distributed_run:
                    reduced_loss = reduce_tensor(loss.data, n_gpus).item()
                else:
                    reduced_loss = loss.item()
                if hparams.fp16_run:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()

                if hparams.fp16_run:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        amp.master_params(optimizer), hparams.grad_clip_thresh)
                    is_overflow = math.isnan(grad_norm)
                else:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        model.parameters(), hparams.grad_clip_thresh)

                optimizer.step()

                if not is_overflow and rank == 0:
                    duration = time.perf_counter() - start
                    print("Train loss {} {:.6f} Grad Norm {:.6f} {:.2f}s/it".format(
                        iteration, reduced_loss, grad_norm, duration))

                    if align_loss is not None and align_bert_loss is not None:
                        logger.log_training(reduced_loss, grad_norm, learning_rate, duration, iteration, mel_loss.item(), gate_loss.item(), align_loss.item(), align_bert_loss().item())
                    elif align_loss is not None and align_bert_loss is None:
                        logger.log_training(reduced_loss, grad_norm, learning_rate, duration, iteration, mel_loss.item(), gate_loss.item(), align_loss.item(), 0)
                    elif align_loss is not None and align_bert_loss is None:
                        logger.log_training(reduced_loss, grad_norm, learning_rate, duration, iteration, mel_loss.item(), gate_loss.item(), 0, align_bert_loss().item())
                    else:
                        logger.log_training(reduced_loss, grad_norm, learning_rate, duration, iteration, mel_loss.item(), gate_loss.item(), 0, 0)
                        
                if not is_overflow and (iteration % hparams.iters_per_checkpoint == 0):
                    val_loss = validate(model, criterion, valset, iteration,
                                hparams.batch_size, n_gpus, collate_fn, logger,
                                hparams.distributed_run, rank)
                    if rank == 0:
                        checkpoint_path = os.path.join(
                            output_directory, "checkpoint_{}".format(iteration))
                        save_checkpoint(model, optimizer, learning_rate, iteration, val_loss,
                                        checkpoint_path)

                        if float(val_loss) < minLoss:
                            minLoss = float(val_loss)
                            save_checkpoint(model, optimizer, learning_rate, iteration, val_loss, os.path.join(output_directory, "checkpoint_best"))

                iteration += 1

        if scheduler_learningrate:     
            scheduler_lr.step()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output_directory', type=str,
                        help='directory to save checkpoints')
    parser.add_argument('-l', '--log_directory', type=str,
                        help='directory to save tensorboard logs')
    parser.add_argument('-c', '--checkpoint_path', type=str, default=None,
                        required=False, help='checkpoint path')
    parser.add_argument('--warm_start', action='store_true',
                        help='load model weights only, ignore specified layers')
    parser.add_argument('--n_gpus', type=int, default=1,
                        required=False, help='number of gpus')
    parser.add_argument('--rank', type=int, default=0,
                        required=False, help='rank of current gpu')
    parser.add_argument('--group_name', type=str, default='group_name',
                        required=False, help='Distributed group name')
    parser.add_argument('--hparams', type=str,
                        required=False, help='comma separated name=value pairs')
    parser.add_argument('--tokenizer_vocab', type=int, help='tokenizer vocab size')
    parser.add_argument('--batch_size', required=False, type=int, help='tokenizer vocab size')
    # parser.add_argument('--train_file', type=str, required=True)
    # parser.add_argument('--val_file', type=str, required=True)
    # parser.add_argument('--n_symbols', type=int, default=313, required=True)
    parser.add_argument('--attention', type=str, required=False)

    args = parser.parse_args()
    hparams = create_hparams(args.hparams)

    args.output_directory = args.output_directory + "_" + str(args.tokenizer_vocab)
    hparams.bert_embeddings_train_path = hparams.bert_embeddings_train_path.split("/")[0] + "_" + str(args.tokenizer_vocab) + "/" + hparams.bert_embeddings_train_path.split("/")[1]
    hparams.bert_embeddings_val_path = hparams.bert_embeddings_val_path.split("/")[0] + "_" + str(args.tokenizer_vocab) + "/" + hparams.bert_embeddings_val_path.split("/")[1]
    hparams.bert_embeddings_cls_train_path = hparams.bert_embeddings_cls_train_path.split("/")[0] + "_" + str(args.tokenizer_vocab) + "/" + hparams.bert_embeddings_cls_train_path.split("/")[1]
    hparams.bert_embeddings_cls_val_path = hparams.bert_embeddings_cls_val_path.split("/")[0] + "_" + str(args.tokenizer_vocab) + "/" + hparams.bert_embeddings_cls_val_path.split("/")[1]

    # hparams.training_files = args.train_file
    # hparams.validation_files = args.val_file
    # hparams.n_symbols = args.n_symbols

    if args.attention:
        hparams.attention = args.attention
    print("Used Attention is ", hparams.attention)

    if args.batch_size:
        hparams.batch_size = args.batch_size
    print("Batch size is ", hparams.batch_size)

    hparams.sub_n_symbols = args.tokenizer_vocab

    if not os.path.isdir(os.getcwd() + "/" + args.output_directory):
        os.makedirs(os.getcwd() + "/" + args.output_directory, exist_ok=True)

    # shutil.copy("hparams.py", os.getcwd() + "/" + args.output_directory)

    torch.backends.cudnn.enabled = hparams.cudnn_enabled
    torch.backends.cudnn.benchmark = hparams.cudnn_benchmark

    print("FP16 Run:", hparams.fp16_run)
    print("Dynamic Loss Scaling:", hparams.dynamic_loss_scaling)
    print("Distributed Run:", hparams.distributed_run)
    print("cuDNN Enabled:", hparams.cudnn_enabled)
    print("cuDNN Benchmark:", hparams.cudnn_benchmark)

    train(args.output_directory, args.log_directory, args.checkpoint_path,
          args.warm_start, args.n_gpus, args.rank, args.group_name, hparams)
