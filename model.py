from math import sqrt
import numpy as np
import torch
from torch.autograd import Variable
from torch import nn
from torch.nn import functional as F
from layers import ConvNorm, LinearNorm
from utils import to_gpu, get_mask_from_lengths
from attention import LocationSensitiveAttention, ForwardAttentionV2, DynamicConvolutionAttention, GMMAttention, StepwiseMonotonicAttention
import hparams


class Prenet(nn.Module):
    def __init__(self, in_dim, sizes):
        super(Prenet, self).__init__()
        in_sizes = [in_dim] + sizes[:-1]
        self.layers = nn.ModuleList(
            [LinearNorm(in_size, out_size, bias=False)
             for (in_size, out_size) in zip(in_sizes, sizes)])

    def forward(self, x):
        for linear in self.layers:
            x = F.dropout(F.relu(linear(x)), p=0.5, training=True)
        return x


class Postnet(nn.Module):
    """Postnet
        - Five 1-d convolution with 512 channels and kernel size 5
    """

    def __init__(self, hparams):
        super(Postnet, self).__init__()
        self.convolutions = nn.ModuleList()

        self.convolutions.append(
            nn.Sequential(
                ConvNorm(hparams.n_mel_channels, hparams.postnet_embedding_dim,
                         kernel_size=hparams.postnet_kernel_size, stride=1,
                         padding=int((hparams.postnet_kernel_size - 1) / 2),
                         dilation=1, w_init_gain='tanh'),
                nn.BatchNorm1d(hparams.postnet_embedding_dim))
        )

        for i in range(1, hparams.postnet_n_convolutions - 1):
            self.convolutions.append(
                nn.Sequential(
                    ConvNorm(hparams.postnet_embedding_dim,
                             hparams.postnet_embedding_dim,
                             kernel_size=hparams.postnet_kernel_size, stride=1,
                             padding=int((hparams.postnet_kernel_size - 1) / 2),
                             dilation=1, w_init_gain='tanh'),
                    nn.BatchNorm1d(hparams.postnet_embedding_dim))
            )

        self.convolutions.append(
            nn.Sequential(
                ConvNorm(hparams.postnet_embedding_dim, hparams.n_mel_channels,
                         kernel_size=hparams.postnet_kernel_size, stride=1,
                         padding=int((hparams.postnet_kernel_size - 1) / 2),
                         dilation=1, w_init_gain='linear'),
                nn.BatchNorm1d(hparams.n_mel_channels))
            )

    def forward(self, x):
        for i in range(len(self.convolutions) - 1):
            x = F.dropout(torch.tanh(self.convolutions[i](x)), 0.5, self.training)
        x = F.dropout(self.convolutions[-1](x), 0.5, self.training)

        return x


class Encoder(nn.Module):
    """Encoder module:
        - Three 1-d convolution banks
        - Bidirectional LSTM
    """
    def __init__(self, hparams):
        super(Encoder, self).__init__()

        convolutions = []
        for _ in range(hparams.encoder_n_convolutions):
            conv_layer = nn.Sequential(
                ConvNorm(hparams.encoder_embedding_dim,
                         hparams.encoder_embedding_dim,
                         kernel_size=hparams.encoder_kernel_size, stride=1,
                         padding=int((hparams.encoder_kernel_size - 1) / 2),
                         dilation=1, w_init_gain='relu'),
                nn.BatchNorm1d(hparams.encoder_embedding_dim))
            convolutions.append(conv_layer)
        self.convolutions = nn.ModuleList(convolutions)

        self.lstm = nn.LSTM(hparams.encoder_embedding_dim,
                            int(hparams.encoder_embedding_dim / 2), 1,
                            batch_first=True, bidirectional=True)

    def forward(self, x, input_lengths):
        for conv in self.convolutions:
            x = F.dropout(F.relu(conv(x)), 0.5, self.training)

        x = x.transpose(1, 2)

        # pytorch tensor are not reversible, hence the conversion
        input_lengths = input_lengths.cpu().numpy()
        x = nn.utils.rnn.pack_padded_sequence(
            x, input_lengths, batch_first=True, enforce_sorted=False)

        self.lstm.flatten_parameters()
        outputs, _ = self.lstm(x)

        outputs, _ = nn.utils.rnn.pad_packed_sequence(
            outputs, batch_first=True)

        return outputs

    def inference(self, x):
        for conv in self.convolutions:
            x = F.dropout(F.relu(conv(x)), 0.5, self.training)

        x = x.transpose(1, 2)

        self.lstm.flatten_parameters()
        outputs, _ = self.lstm(x)

        return outputs


class Decoder(nn.Module):
    def __init__(self, hparams):
        super(Decoder, self).__init__()
        self.n_mel_channels = hparams.n_mel_channels
        self.n_frames_per_step = hparams.n_frames_per_step
        self.encoder_embedding_dim = hparams.encoder_embedding_dim
        self.attention_rnn_dim = hparams.attention_rnn_dim
        self.decoder_rnn_dim = hparams.decoder_rnn_dim
        self.prenet_dim = hparams.prenet_dim
        self.max_decoder_steps = hparams.max_decoder_steps
        self.gate_threshold = hparams.gate_threshold
        self.p_attention_dropout = hparams.p_attention_dropout
        self.p_decoder_dropout = hparams.p_decoder_dropout

        self.prenet = Prenet(
            hparams.n_mel_channels * hparams.n_frames_per_step,
            [hparams.prenet_dim, hparams.prenet_dim])
        
        self.prenet_bert = Prenet(
            hparams.n_mel_channels * hparams.n_frames_per_step,
            [hparams.prenet_dim, hparams.prenet_dim])

        self.attention_rnn = nn.LSTMCell(
            hparams.prenet_dim + hparams.encoder_embedding_dim,
            hparams.attention_rnn_dim)

        self.attention_rnn_bert = nn.LSTMCell(
            hparams.prenet_dim + hparams.encoder_embedding_dim,
            hparams.attention_rnn_dim)

        if hparams.attention == "StepwiseMonotonicAttention":
            print("Use SMA")
            self.attention_layer = StepwiseMonotonicAttention(
            hparams.attention_rnn_dim, hparams.encoder_embedding_dim,
            hparams.attention_dim, hparams.attention_location_n_filters,
            hparams.attention_location_kernel_size)
            self.attention_layer_bert = StepwiseMonotonicAttention(
            hparams.attention_rnn_dim, hparams.encoder_embedding_dim,
            hparams.attention_dim, hparams.attention_location_n_filters,
            hparams.attention_location_kernel_size)
        elif hparams.attention == "DynamicConvolutionAttention":
            print("Use DCA")
            self.attention_layer = DynamicConvolutionAttention(
            hparams.attention_rnn_dim, hparams.encoder_embedding_dim,
            hparams.attention_dim, hparams.attention_location_n_filters,
            hparams.attention_location_kernel_size)
        elif hparams.attention == "ForwardAttentionV2":
            print("Use ForwardAttention")
            self.attention_layer = ForwardAttentionV2(
            hparams.attention_rnn_dim, hparams.encoder_embedding_dim,
            hparams.attention_dim, hparams.attention_location_n_filters,
            hparams.attention_location_kernel_size)
        elif hparams.attention == "GMMAttention":
            print("Use GMMA")
            self.attention_layer = GMMAttention(
            hparams.attention_rnn_dim, hparams.encoder_embedding_dim,
            hparams.attention_dim, hparams.attention_location_n_filters,
            hparams.attention_location_kernel_size)
        else:
            print("Use LSA")
            self.attention_layer = LocationSensitiveAttention(
            hparams.attention_rnn_dim, hparams.encoder_embedding_dim,
            hparams.attention_dim, hparams.attention_location_n_filters,
            hparams.attention_location_kernel_size)

        self.decoder_rnn = nn.LSTMCell(
            2*hparams.attention_rnn_dim + 2*hparams.encoder_embedding_dim,
            hparams.decoder_rnn_dim, 1)

        self.decoder_rnn_bert = nn.LSTMCell(
            hparams.attention_rnn_dim + hparams.encoder_embedding_dim,
            hparams.decoder_rnn_dim, 1)

        self.linear_projection = LinearNorm(
            hparams.decoder_rnn_dim + 2*hparams.encoder_embedding_dim,
            hparams.n_mel_channels * hparams.n_frames_per_step)

        self.gate_layer = LinearNorm(
            hparams.decoder_rnn_dim + 2*hparams.encoder_embedding_dim, 1,
            bias=True, w_init_gain='sigmoid')

    def get_go_frame(self, memory):
        """ Gets all zeros frames to use as first decoder input
        PARAMS
        ------
        memory: decoder outputs

        RETURNS
        -------
        decoder_input: all zeros frames
        """
        B = memory.size(0)
        decoder_input = Variable(memory.data.new(B, self.n_mel_channels * self.n_frames_per_step).zero_())
        return decoder_input

    def initialize_decoder_states(self, memory, embeddings, mask, mask_bert):
        """ Initializes attention rnn states, decoder rnn states, attention
        weights, attention cumulative weights, attention context, stores memory
        and stores processed memory
        PARAMS
        ------
        memory: Encoder outputs
        mask: Mask for padded data if training, expects None for inference
        """
        B = memory.size(0)
        MAX_TIME = memory.size(1)
        BB = embeddings.size(0)
        MAX_TIME_BERT = embeddings.size(1)

        self.attention_hidden = Variable(memory.data.new(B, self.attention_rnn_dim).zero_())
        self.attention_cell = Variable(memory.data.new(B, self.attention_rnn_dim).zero_())

        self.attention_hidden_bert = Variable(embeddings.data.new(BB, self.attention_rnn_dim).zero_())
        self.attention_cell_bert = Variable(embeddings.data.new(BB, self.attention_rnn_dim).zero_())

        self.decoder_hidden = Variable(memory.data.new(B, self.decoder_rnn_dim).zero_())
        self.decoder_cell = Variable(memory.data.new(B, self.decoder_rnn_dim).zero_())

        self.decoder_hidden_bert = Variable(embeddings.data.new(BB, self.decoder_rnn_dim).zero_())
        self.decoder_cell_bert = Variable(embeddings.data.new(BB, self.decoder_rnn_dim).zero_())

        self.attention_weights = Variable(memory.data.new(B, MAX_TIME).zero_())
        self.attention_weights_cum = Variable(memory.data.new(B, MAX_TIME).zero_())
        self.attention_context = Variable(memory.data.new(B, self.encoder_embedding_dim).zero_())

        self.attention_weights_bert = Variable(embeddings.data.new(BB, MAX_TIME_BERT).zero_())
        self.attention_weights_cum_bert = Variable(embeddings.data.new(BB, MAX_TIME_BERT).zero_())
        self.attention_context_bert = Variable(embeddings.data.new(BB, self.encoder_embedding_dim).zero_())
        
        self.memory = memory
        self.processed_memory = self.attention_layer.memory_layer(memory)
        self.attention_layer.init_attention(self.processed_memory)
        self.bert = embeddings
        self.processed_bert = self.attention_layer_bert.memory_layer(embeddings)
        self.attention_layer_bert.init_attention(self.processed_bert)
        self.mask = mask
        self.mask_bert = mask_bert

        self.log_alpha = memory.new_zeros(B, MAX_TIME).fill_(-float(1e4))
        self.log_alpha[:, 0].fill_(0.)

        self.log_alpha_bert = embeddings.new_zeros(BB, MAX_TIME_BERT).fill_(-float(1e4))
        self.log_alpha_bert[:, 0].fill_(0.)

    def parse_decoder_inputs(self, decoder_inputs):
        """ Prepares decoder inputs, i.e. mel outputs
        PARAMS
        ------
        decoder_inputs: inputs used for teacher-forced training, i.e. mel-specs

        RETURNS
        -------
        inputs: processed decoder inputs

        """
        # (B, n_mel_channels, T_out) -> (B, T_out, n_mel_channels)
        decoder_inputs = decoder_inputs.transpose(1, 2)
        decoder_inputs = decoder_inputs.view(decoder_inputs.size(0), int(decoder_inputs.size(1)/self.n_frames_per_step), -1)
        # (B, T_out, n_mel_channels) -> (T_out, B, n_mel_channels)
        decoder_inputs = decoder_inputs.transpose(0, 1)
        return decoder_inputs

    def parse_decoder_outputs(self, mel_outputs, gate_outputs, alignments, alignments_bert):
        """ Prepares decoder outputs for output
        PARAMS
        ------
        mel_outputs:
        gate_outputs: gate output energies
        alignments:
        alignments_bert:

        RETURNS
        -------
        mel_outputs:
        gate_outpust: gate output energies
        alignments:
        alignments_bert:
        """
        # (T_out, B) -> (B, T_out)
        alignments = torch.stack(alignments).transpose(0, 1)
        # (T_out, B) -> (B, T_out)
        alignments_bert = torch.stack(alignments_bert).transpose(0, 1)
        # (T_out, B) -> (B, T_out)
        gate_outputs = torch.stack(gate_outputs).transpose(0, 1)
        gate_outputs = gate_outputs.contiguous()
        # (T_out, B, n_mel_channels) -> (B, T_out, n_mel_channels)
        mel_outputs = torch.stack(mel_outputs).transpose(0, 1).contiguous()
        # decouple frames per step
        mel_outputs = mel_outputs.view(mel_outputs.size(0), -1, self.n_mel_channels)
        # (B, T_out, n_mel_channels) -> (B, n_mel_channels, T_out)
        mel_outputs = mel_outputs.transpose(1, 2)

        return mel_outputs, gate_outputs, alignments, alignments_bert

    def decode(self, decoder_char_input, decoder_bert_input):
        """ Decoder step using stored states, attention and memory
        PARAMS
        ------
        decoder_input: previous mel output

        RETURNS
        -------
        mel_output:
        gate_output: gate output energies
        attention_weights:
        """
        #The prenet output and attention context vector are concatenated and passed through a stack of 2 uni-directional LSTM layers with 1024 units

        # LSTM 1 with self.attention_context t-1
        cell_input_char = torch.cat((decoder_char_input, self.attention_context), -1) # 768 = 256 + 512
        cell_input_bert = torch.cat((decoder_bert_input, self.attention_context_bert), -1) # 768 = 256 + 512

        self.attention_hidden, self.attention_cell = self.attention_rnn(cell_input_char, (self.attention_hidden, self.attention_cell)) # 768 -> 1024
        self.attention_hidden = F.dropout(self.attention_hidden, self.p_attention_dropout, self.training)
        self.attention_cell = F.dropout(self.attention_cell, self.p_attention_dropout, self.training)

        self.attention_hidden_bert, self.attention_cell_bert = self.attention_rnn_bert(cell_input_bert, (self.attention_hidden_bert, self.attention_cell_bert)) # 768 -> 1024
        self.attention_hidden_bert = F.dropout(self.attention_hidden_bert, self.p_attention_dropout, self.training)
        self.attention_cell_bert = F.dropout(self.attention_cell_bert, self.p_attention_dropout, self.training)
        
        # Attention: at each time step t, calculate probability for each input phone

        ## [B,1,sequence_len] + [B,1,sequence_len] => [B,2,sequence_len] Attention of previous step and cumulative attention (long term previous step)
        attention_weights_cat = torch.cat((self.attention_weights.unsqueeze(1),self.attention_weights_cum.unsqueeze(1)), dim=1)
        attention_weights_cat_bert = torch.cat((self.attention_weights_bert.unsqueeze(1),self.attention_weights_cum_bert.unsqueeze(1)), dim=1)
        
        ## Query: self.attention_hidden -> Key: self.memory => F_Sum(AttentionScore * Value) : self.attention_context with Attention score here is self.attention_weights
        self.attention_context, self.attention_weights = self.attention_layer(self.attention_hidden, self.memory, self.processed_memory, attention_weights_cat, self.mask, self.log_alpha)
        self.attention_context_bert, self.attention_weights_bert = self.attention_layer_bert(self.attention_hidden_bert, self.bert, self.processed_bert, attention_weights_cat_bert, self.mask_bert, self.log_alpha_bert)
        
        self.attention_weights_cum += self.attention_weights #[B,sequence_len] + [B,sequence_len] => [B,sequence_len] 
        self.attention_weights_cum_bert += self.attention_weights_bert #[B,sequence_len] + [B,sequence_len] => [B,sequence_len] 

        # LSTM 2 with self.attention_context t
        decoder_input = torch.cat((self.attention_hidden, self.attention_context, self.attention_hidden_bert, self.attention_context_bert), -1) # [B,1536] = [B,1024] + [B,512]
        # decoder_input = torch.cat((self.attention_hidden, self.attention_hidden_bert, self.attention_context, self.attention_context_bert), -1) # [B,1536] = [B,1024] + [B,512]
        # print("self.attention_hidden:",self.attention_hidden.size())
        # print("self.attention_hidden_bert:",self.attention_hidden_bert.size())
        # print("self.attention_context:",self.attention_context.size())
        # print("self.attention_context_bert:",self.attention_context_bert.size())
        # print("torch.add(self.attention_hidden, self.attention_hidden_bert):",torch.add(self.attention_hidden, self.attention_hidden_bert).size())
        # print("torch.add(self.attention_context, self.attention_context_bert):",torch.add(self.attention_context, self.attention_context_bert).size())
        # decoder_input = torch.cat((torch.add(self.attention_hidden, self.attention_hidden_bert), torch.add(self.attention_context, self.attention_context_bert)), -1) # [B,1536] = [B,1024] + [B,512]
        self.decoder_hidden, self.decoder_cell = self.decoder_rnn(decoder_input, (self.decoder_hidden, self.decoder_cell)) # 1536 -> 1024
        self.decoder_hidden = F.dropout(self.decoder_hidden, self.p_decoder_dropout, self.training)
        self.decoder_cell = F.dropout(self.decoder_cell, self.p_decoder_dropout, self.training)

        # decoder_input = torch.cat((self.attention_hidden_bert, self.attention_context_bert), -1) # [B,1536] = [B,1024] + [B,512]
        # self.decoder_hidden_bert, self.decoder_cell_bert = self.decoder_rnn_bert(decoder_input, (self.decoder_hidden_bert, self.decoder_cell_bert)) # 1536 -> 1024
        # self.decoder_hidden_bert = F.dropout(self.decoder_hidden_bert, self.p_decoder_dropout, self.training)
        # self.decoder_cell_bert = F.dropout(self.decoder_cell_bert, self.p_decoder_dropout, self.training)

        # The concatenation of the LSTM output and the attention context vector is projected through a linear transform to predict the target spectrogram frame
        # decoder_hidden_attention_context = torch.cat((self.decoder_hidden, self.attention_context, self.decoder_hidden_bert, self.attention_context_bert), dim=1) # 1536 = 1024 + 512
        decoder_hidden_attention_context = torch.cat((self.decoder_hidden, self.attention_context, self.attention_context_bert), dim=1) # 1536 = 1024 + 512

        # Linear Mel Predict
        decoder_output = self.linear_projection(decoder_hidden_attention_context) # [B,1536] -> [B,80]

        # Linear Gate Predict
        gate_prediction = self.gate_layer(decoder_hidden_attention_context) # 1536 -> 1

        return decoder_output, gate_prediction, self.attention_weights, self.attention_weights_bert

    def forward(self, memory, embeddings, decoder_inputs, memory_lengths, bert_lengths):
        """ Decoder forward pass for training
        PARAMS
        ------
        memory: Encoder outputs
        decoder_inputs: Decoder inputs for teacher forcing. i.e. mel-specs
        memory_lengths: Encoder output lengths for attention masking.

        RETURNS
        -------
        mel_outputs: mel outputs from the decoder
        gate_outputs: gate outputs from the decoder
        alignments: sequence of attention weights from the decoder
        alignments_bert: sequence of attention bert weights from the decoder
        """
        decoder_input = self.get_go_frame(memory).unsqueeze(0)
        decoder_bert_input = self.get_go_frame(embeddings).unsqueeze(0)
        decoder_inputs = self.parse_decoder_inputs(decoder_inputs)
        decoder_inputs_char = torch.cat((decoder_input, decoder_inputs), dim=0) # Add initial frame: [T, B, mel_chanel] -> [T+1, B, mel_chanel]
        decoder_inputs_bert = torch.cat((decoder_bert_input, decoder_inputs), dim=0) # Add initial frame: [T, B, mel_chanel] -> [T+1, B, mel_chanel]
        decoder_inputs_char = self.prenet(decoder_inputs_char) # [T+1, B, mel_chanel] -> [T+1, B, 256]
        decoder_inputs_bert = self.prenet_bert(decoder_inputs_bert)
        self.initialize_decoder_states(memory, embeddings, mask=~get_mask_from_lengths(memory_lengths), mask_bert=~get_mask_from_lengths(bert_lengths))

        mel_outputs, gate_outputs, alignments, alignments_bert = [], [], [], []
        while len(mel_outputs) < decoder_inputs_char.size(0) - 1:
            decoder_char_input = decoder_inputs_char[len(mel_outputs)]
            decoder_bert_input = decoder_inputs_bert[len(mel_outputs)]
            mel_output, gate_output, attention_weights, attention_weights_bert = self.decode(decoder_char_input, decoder_bert_input)
            mel_outputs += [mel_output.squeeze(1)]
            gate_outputs += [gate_output.squeeze(1)]
            alignments += [attention_weights]
            alignments_bert += [attention_weights_bert]

        mel_outputs, gate_outputs, alignments, alignments_bert = self.parse_decoder_outputs(mel_outputs, gate_outputs, alignments, alignments_bert)

        return mel_outputs, gate_outputs, alignments, alignments_bert

    def inference(self, memory, embeddings):
        """ Decoder inference
        PARAMS
        ------
        memory: Encoder outputs

        RETURNS
        -------
        mel_outputs: mel outputs from the decoder
        gate_outputs: gate outputs from the decoder
        alignments: sequence of attention weights from the decoder
        alignments_bert: sequence of attention bert weights from the decoder
        """
        INFER_FLAG = True
        decoder_input = self.get_go_frame(memory)
        decoder_bert_input = self.get_go_frame(embeddings)
        self.initialize_decoder_states(memory, embeddings, mask=None, mask_bert=None)

        mel_outputs, gate_outputs, alignments, alignments_bert = [], [], [], []
        decoder_input = self.prenet(decoder_input)
        decoder_bert_input = self.prenet_bert(decoder_bert_input)
        
        mel_output, gate_output, alignment, alignment_bert = self.decode(decoder_input, decoder_bert_input)

        mel_outputs += [mel_output.squeeze(1)]
        gate_outputs += [gate_output]
        alignments += [alignment]
        alignments_bert += [alignment_bert]

        decoder_input = mel_output
        decoder_bert_input = mel_output
        if torch.sigmoid(gate_output.data) > self.gate_threshold:
            print("pass")
        elif len(mel_outputs) == self.max_decoder_steps:
            print("Warning! Reached max decoder steps")
            INFER_FLAG = False
        else:
            while True:

                # prediction from the previous time step is first passed through a small pre-net containing 2 fully connected layers of 256 hidden ReLU units
                decoder_input = self.prenet(decoder_input)
                decoder_bert_input = self.prenet_bert(decoder_bert_input)
                
                mel_output, gate_output, alignment, alignment_bert = self.decode(decoder_input, decoder_bert_input)

                mel_outputs += [mel_output.squeeze(1)]
                gate_outputs += [gate_output]
                alignments += [alignment]
                alignments_bert += [alignment_bert]

                if torch.sigmoid(gate_output.data) > self.gate_threshold:
                    break
                elif len(mel_outputs) == self.max_decoder_steps:
                    print("Warning! Reached max decoder steps")
                    INFER_FLAG = False
                    break
                
                decoder_input = mel_output
                decoder_bert_input = mel_output

        mel_outputs, gate_outputs, alignments, alignments_bert = self.parse_decoder_outputs(mel_outputs, gate_outputs, alignments, alignments_bert)

        return mel_outputs, gate_outputs, alignments, alignments_bert, INFER_FLAG

class BERT_Tacotron2(nn.Module):
    def __init__(self, hparams):
        super(BERT_Tacotron2, self).__init__()
        self.mask_padding = hparams.mask_padding
        self.fp16_run = hparams.fp16_run
        self.n_mel_channels = hparams.n_mel_channels
        self.n_frames_per_step = hparams.n_frames_per_step
        self.embedding = nn.Embedding(hparams.n_symbols, hparams.symbols_embedding_dim)
        self.embedding_sub = nn.Embedding(hparams.sub_n_symbols, hparams.symbols_embedding_dim)
        std = sqrt(2.0 / (hparams.n_symbols + hparams.symbols_embedding_dim))
        val = sqrt(3.0) * std  # uniform bounds for std
        self.embedding.weight.data.uniform_(-val, val)
        self.embedding_sub.weight.data.uniform_(-val, val)
        self.encoder = Encoder(hparams)
        self.encoder_sub = Encoder(hparams)
        # self.linear_converter_sub = LinearNorm(hparams.BERT_embedding_dim, hparams.encoder_embedding_dim)
        self.linear_converter = LinearNorm(
            hparams.encoder_embedding_dim+hparams.BERT_embedding_dim, hparams.encoder_embedding_dim)
        self.linear_converter_sub = LinearNorm(
            hparams.encoder_embedding_dim+hparams.BERT_embedding_dim, hparams.encoder_embedding_dim)
        self.decoder = Decoder(hparams)
        self.postnet = Postnet(hparams)

    def parse_batch(self, batch):
        text_padded, input_lengths, input_lengths_bert, mel_padded, gate_padded, output_lengths, embeddings, phoneme_embeddings_cls, bert_embeddings_cls, align_padded = batch
        text_padded = to_gpu(text_padded).long()
        input_lengths = to_gpu(input_lengths).long()
        input_lengths_bert = to_gpu(input_lengths_bert).long()
        max_input_len = torch.max(torch.cat((input_lengths,input_lengths_bert), 0).data).item()
        max_output_len = torch.max(output_lengths.data).item()
        mel_padded = to_gpu(mel_padded).float()
        gate_padded = to_gpu(gate_padded).float()
        output_lengths = to_gpu(output_lengths).long()
        align_padded = to_gpu(align_padded).float()

        return ((text_padded, input_lengths, input_lengths_bert, mel_padded, (max_input_len, max_output_len), output_lengths, embeddings, phoneme_embeddings_cls, bert_embeddings_cls), (mel_padded, gate_padded, align_padded))

    def parse_output(self, outputs, output_lengths=None):
        if self.mask_padding and output_lengths is not None:
            mask = ~get_mask_from_lengths(output_lengths)
            mask = mask.expand(self.n_mel_channels, mask.size(0), mask.size(1))
            mask = mask.permute(1, 0, 2)

            outputs[0].data.masked_fill_(mask, 0.0)
            outputs[1].data.masked_fill_(mask, 0.0)
            outputs[2].data.masked_fill_(mask[:, 0, :], 1e3)  # gate energies

        return outputs

    def forward(self, inputs):
        text_inputs, text_lengths, bert_lengths, mels, (max_input_len, max_output_len), output_lengths, embeddings, phoneme_embeddings_cls, bert_embeddings_cls = inputs
        text_lengths, bert_lengths, output_lengths = text_lengths.data, bert_lengths.data, output_lengths.data
        embedded_inputs = self.embedding(text_inputs).transpose(1, 2)
        encoder_outputs = self.encoder(embedded_inputs, text_lengths) # => [B, sequence_len, embedding_dim]
        decoder_inputs = torch.cat([encoder_outputs, phoneme_embeddings_cls], 2)
        decoder_inputs = self.linear_converter(decoder_inputs)

        embedded_inputs_sub = self.embedding_sub(embeddings).transpose(1, 2)
        encoder_sub_outputs = self.encoder_sub(embedded_inputs_sub, bert_lengths) # => [B, sequence_len, embedding_dim]
        decoder_sub_inputs = torch.cat([encoder_sub_outputs, bert_embeddings_cls], 2)
        decoder_sub_inputs = self.linear_converter_sub(decoder_sub_inputs)

        mel_outputs, gate_outputs, alignments, alignments_bert = self.decoder(decoder_inputs, decoder_sub_inputs, mels, text_lengths, bert_lengths) # mel_outputs => [B, mel_channel, T]
        mel_outputs_postnet = self.postnet(mel_outputs)
        mel_outputs_postnet = mel_outputs + mel_outputs_postnet

        return self.parse_output([mel_outputs, mel_outputs_postnet, gate_outputs, alignments, alignments_bert],output_lengths)

    def inference(self, inputs, embeddings, phoneme_embeddings_cls, bert_embeddings_cls):
        embedded_inputs = self.embedding(inputs).transpose(1, 2)
        encoder_outputs = self.encoder.inference(embedded_inputs)

        decoder_inputs = torch.cat([encoder_outputs, phoneme_embeddings_cls], 2)
        decoder_inputs = self.linear_converter(decoder_inputs)

        embedded_inputs_sub = self.embedding_sub(embeddings).transpose(1, 2)
        encoder_sub_outputs = self.encoder_sub.inference(embedded_inputs_sub) # => [B, sequence_len, embedding_dim]
        decoder_sub_inputs = torch.cat([encoder_sub_outputs, bert_embeddings_cls], 2)
        decoder_sub_inputs = self.linear_converter_sub(decoder_sub_inputs)

        mel_outputs, gate_outputs, alignments, alignments_bert, INFER_FLAG = self.decoder.inference(
            decoder_inputs, decoder_sub_inputs)

        mel_outputs_postnet = self.postnet(mel_outputs)
        mel_outputs_postnet = mel_outputs + mel_outputs_postnet

        outputs = self.parse_output([mel_outputs, mel_outputs_postnet, gate_outputs, alignments, alignments_bert, INFER_FLAG])

        return outputs

if __name__ == "__main__":
    model = BERT_Tacotron2(hparams.create_hparams())
    print(model)