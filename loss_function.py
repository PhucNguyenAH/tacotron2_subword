from torch import nn, log, sum, mean
from utils import plot_spectrogram
from ssim import SSIM


class Tacotron2Loss(nn.Module):
    def __init__(self, alignloss = ""):
        super(Tacotron2Loss, self).__init__()
        self.alignloss = alignloss
        # self.ssim = SSIM(window_size = 11, size_average = True)

    def forward(self, model_output, targets, x, iters = 0):
        mel_target, gate_target, align_target = targets[0], targets[1], targets[2] #
        mel_target.requires_grad = False
        gate_target.requires_grad = False
        align_target.requires_grad = False #
        gate_target = gate_target.view(-1, 1)

        mel_out, mel_out_postnet, gate_out, align_out, align_bert_out = model_output
        gate_out = gate_out.view(-1, 1)
        mel_loss = nn.MSELoss()(mel_out, mel_target) + nn.MSELoss()(mel_out_postnet, mel_target)
        gate_loss = nn.BCEWithLogitsLoss()(gate_out, gate_target)

        # ssim_loss = -self.ssim(mel_out.unsqueeze(1), mel_target.unsqueeze(1))

        if self.alignloss == "":
            align_loss = None
            align_bert_loss = None
        elif self.alignloss == "L2" and iters < 40000:
            align_loss = nn.MSELoss()(align_out, align_target)
            align_bert_loss = nn.MSELoss()(align_bert_out, align_target)
        elif self.alignloss == "KL" and iters < 40000:
            align_loss_kl = 0
            align_bert_loss_kl = 0
            align_out_mask = align_out.clone()
            align_bert_out_mask = align_bert_out.clone()
            align_out[align_out_mask==0] = 0.000001
            align_bert_out[align_bert_out_mask==0] = 0.000001

            align_target_mask = align_target.clone()
            align_target[align_target_mask==0] = 0.000001

            text_len = x[1]
            mel_len = x[4]
            for batch_idx in range(align_target.size(0)):
                aliout = align_out[batch_idx][:mel_len[batch_idx]-1][:text_len[batch_idx]-1]
                alibertout = align_bert_out[batch_idx][:mel_len[batch_idx]-1][:text_len[batch_idx]-1]
                alitar = align_target[batch_idx][:mel_len[batch_idx]-1][:text_len[batch_idx]-1]
                kl = mean(sum(alitar*(log(alitar)-log(aliout)), dim=-1 ))
                kl_bert = mean(sum(alitar*(log(alitar)-log(alibertout)), dim=-1 ))
                align_loss_kl += kl
                align_bert_loss_kl += kl_bert
            align_loss = align_loss_kl
            align_bert_loss = align_bert_loss_kl
        else:
            align_loss = None
            align_bert_loss = None

        if align_loss is not None and align_bert_loss is not None:
            return mel_loss + gate_loss + align_loss + align_bert_loss, mel_loss, gate_loss, align_loss, align_bert_loss # + ssim_loss
        elif align_loss is None and align_bert_loss is not None:
            return mel_loss + gate_loss + align_bert_loss, mel_loss, gate_loss, align_loss, align_bert_loss # + ssim_loss
        if align_loss is not None and align_bert_loss is None:
            return mel_loss + gate_loss + align_loss , mel_loss, gate_loss, align_loss, align_bert_loss # + ssim_loss
        else:
            return mel_loss + gate_loss, mel_loss, gate_loss, align_loss, align_bert_loss
