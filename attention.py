import torch
from torch.nn import functional as F
from layers import ConvNorm, LinearNorm
from torch import nn
import numpy as np

class LocationLayer(nn.Module):
    def __init__(self, attention_n_filters, attention_kernel_size,
                 attention_dim):
        super(LocationLayer, self).__init__()
        padding = int((attention_kernel_size - 1) / 2)
        self.location_conv = ConvNorm(2, attention_n_filters,
                                      kernel_size=attention_kernel_size,
                                      padding=padding, bias=False, stride=1,
                                      dilation=1)
        self.location_dense = LinearNorm(attention_n_filters, attention_dim,
                                         bias=False, w_init_gain='tanh')

    def forward(self, attention_weights_cat):
        processed_attention = self.location_conv(attention_weights_cat)
        processed_attention = processed_attention.transpose(1, 2)
        processed_attention = self.location_dense(processed_attention)
        return processed_attention

class LocationSensitiveAttention(nn.Module):
    def __init__(self, attention_rnn_dim, embedding_dim, attention_dim,
                 attention_location_n_filters, attention_location_kernel_size):
        super(LocationSensitiveAttention, self).__init__()
        self.query_layer = LinearNorm(attention_rnn_dim, attention_dim,
                                      bias=False, w_init_gain='tanh')
        self.memory_layer = LinearNorm(embedding_dim, attention_dim, bias=False,
                                       w_init_gain='tanh')
        self.v = LinearNorm(attention_dim, 1, bias=False)
        self.location_layer = LocationLayer(attention_location_n_filters,
                                            attention_location_kernel_size,
                                            attention_dim)
        self.score_mask_value = -float("inf")

    def init_attention(self, processed_memory):
        return None

    def get_alignment_energies(self, query, processed_memory,
                               attention_weights_cat):
        """
        PARAMS
        ------
        query: decoder output (batch, n_mel_channels * n_frames_per_step)
        processed_memory: processed encoder outputs (B, T_in, attention_dim)
        attention_weights_cat: cumulative and prev. att weights (B, 2, max_time)

        RETURNS
        -------
        alignment (batch, max_time)
        """

        processed_query = self.query_layer(query.unsqueeze(1))
        processed_attention_weights = self.location_layer(attention_weights_cat)
        energies = self.v(torch.tanh(
            processed_query + processed_attention_weights + processed_memory))

        energies = energies.squeeze(-1)
        return energies

    def forward(self, attention_hidden_state, memory, processed_memory,
                attention_weights_cat, mask, log_alpha):
        """
        PARAMS
        ------
        attention_hidden_state: attention rnn last output
        memory: encoder outputs
        processed_memory: processed encoder outputs
        attention_weights_cat: previous and cummulative attention weights
        mask: binary mask for padded data
        """

        alignment = self.get_alignment_energies(attention_hidden_state, processed_memory, attention_weights_cat) # (batch, max_time)

        if mask is not None:
            alignment.data.masked_fill_(mask, self.score_mask_value)

        attention_weights = F.softmax(alignment, dim=1)
        attention_context = torch.bmm(attention_weights.unsqueeze(1), memory) # (batch, 1, dim)
        attention_context = attention_context.squeeze(1) # # (batch, dim)

        return attention_context, attention_weights

class ForwardAttentionV2(nn.Module):
    def __init__(self, attention_rnn_dim, embedding_dim, attention_dim,
                 attention_location_n_filters, attention_location_kernel_size):
        super(ForwardAttentionV2, self).__init__()
        self.query_layer = LinearNorm(attention_rnn_dim, attention_dim,
                                      bias=False, w_init_gain='tanh')
        self.memory_layer = LinearNorm(embedding_dim, attention_dim, bias=False,
                                       w_init_gain='tanh')
        self.v = LinearNorm(attention_dim, 1, bias=False)
        self.location_layer = LocationLayer(attention_location_n_filters,
                                            attention_location_kernel_size,
                                            attention_dim)
        self.score_mask_value = -float(1e20)
    
    def init_attention(self, processed_memory):
        return None

    def get_alignment_energies(self, query, processed_memory,
                               attention_weights_cat):
        """
        PARAMS
        ------
        query: decoder output (batch, n_mel_channels * n_frames_per_step)
        processed_memory: processed encoder outputs (B, T_in, attention_dim)
        attention_weights_cat:  prev. and cumulative att weights (B, 2, max_time)
        RETURNS
        -------
        alignment (batch, max_time)
        """

        processed_query = self.query_layer(query.unsqueeze(1))
        processed_attention_weights = self.location_layer(attention_weights_cat)
        energies = self.v(torch.tanh(
            processed_query + processed_attention_weights + processed_memory))

        energies = energies.squeeze(-1)
        return energies

    def forward(self, attention_hidden_state, memory, processed_memory,
                attention_weights_cat, mask, log_alpha):
        """
        PARAMS
        ------
        attention_hidden_state: attention rnn last output
        memory: encoder outputs
        processed_memory: processed encoder outputs
        attention_weights_cat: previous and cummulative attention weights
        mask: binary mask for padded data
        """
        log_energy = self.get_alignment_energies(
            attention_hidden_state, processed_memory, attention_weights_cat)

        if mask is not None:
            log_energy.data.masked_fill_(mask, self.score_mask_value)

        fwd_shifted_alpha = F.pad(log_alpha[:, :-1], [1, 0], 'constant', self.score_mask_value)
        biased = torch.logsumexp(torch.cat([log_alpha.unsqueeze(2), fwd_shifted_alpha.unsqueeze(2)], 2), 2)

        log_alpha_new = biased + log_energy

        attention_weights = F.softmax(log_alpha_new, dim=1)
        attention_context = torch.bmm(attention_weights.unsqueeze(1), memory)
        attention_context = attention_context.squeeze(1)

        return attention_context, attention_weights#, log_alpha_new

class ContentAttention(nn.Module):
    def __init__(self, query_dim, memory_dim, attention_dim):
        """
        :param query_dim: query dim
        :param memory_dim: key dim
        :param attention_dim: attention dim
        """
        super(ContentAttention, self).__init__()
        self.query_dim = query_dim
        self.memory_dim = memory_dim
        self.attention_dim = attention_dim
        self.query_layer = LinearNorm(query_dim, attention_dim,
                                      bias=False, w_init_gain='tanh')
        self.memory_layer = LinearNorm(memory_dim, attention_dim,
                                      bias=False, w_init_gain='tanh')
        self.v = LinearNorm(attention_dim, 1, bias=False)
        self.score_mask_value = -float("1e20")

    def init_attention(self, processed_memory):
        return None
    
    def forward(self, query, memory, mask=None):
        """
        :param query: decoder query [B, query_dim]
        :param memory: keys [B, seq_len, encoder_hidden_dim]
        :param mask: alignments mask [B, seq_len]
        :return: context [B, encoder_hidden_dim]
        """
        processed_query = self.query_layer(query.unsqueeze(1))
        processed_memory = self.memory_layer(memory)
        energies = self.v(torch.tanh(processed_query + processed_memory))
        alignment = energies.squeeze(2)
        if mask is not None:
            alignment.data.masked_fill_(mask, self.score_mask_value)
 
        attention_weights = F.softmax(alignment, dim=1)
        attention_context = torch.bmm(attention_weights.unsqueeze(1), memory)
        attention_context = attention_context.squeeze(1)

        return attention_context, attention_weights

from scipy.stats import betabinom
class DynamicConvolutionAttention(nn.Module):
    def __init__(self, attention_rnn_dim, embedding_dim, attention_dim,
                 attention_location_n_filters, attention_location_kernel_size):
        super(DynamicConvolutionAttention, self).__init__()
        self.memory_layer = LinearNorm(embedding_dim, attention_dim, bias=False, w_init_gain='tanh')
        self.score_mask_value = -float("inf")

        static_channels=8
        static_kernel_size=21
        dynamic_channels=8
        dynamic_kernel_size=21
        prior_length=11
        alpha=0.1
        beta=0.9

        self.prior_length = prior_length
        self.dynamic_channels = dynamic_channels
        self.dynamic_kernel_size = dynamic_kernel_size

        P = betabinom.pmf(np.arange(prior_length), prior_length - 1, alpha, beta)

        self.register_buffer("P", torch.FloatTensor(P).flip(0))
        self.W = nn.Linear(attention_rnn_dim, attention_dim)
        self.V = nn.Linear(
            attention_dim, dynamic_channels * dynamic_kernel_size, bias=False
        )
        self.F = nn.Conv1d(
            1,
            static_channels,
            static_kernel_size,
            padding=(static_kernel_size - 1) // 2,
            bias=False,
        )
        self.U = nn.Linear(static_channels, attention_dim, bias=False)
        self.T = nn.Linear(dynamic_channels, attention_dim)
        self.v = nn.Linear(attention_dim, 1, bias=False)
    
    def init_attention(self, processed_memory):
        b, t, _ = processed_memory.size()
        self.alignment_pre = F.one_hot(torch.zeros(b, dtype=torch.long), t).float().cuda()

    def get_energies(self, query, processed_memory):
        query = query.squeeze(1)
        p = F.conv1d(
            F.pad(self.alignment_pre.unsqueeze(1), (self.prior_length - 1, 0)), self.P.view(1, 1, -1)
        )
        p = torch.log(p.clamp_min_(1e-6)).squeeze(1)

        G = self.V(torch.tanh(self.W(query)))
        g = F.conv1d(
            self.alignment_pre.unsqueeze(0),
            G.view(-1, 1, self.dynamic_kernel_size),
            padding=(self.dynamic_kernel_size - 1) // 2,
            groups=query.size(0),
        )
        g = g.view(query.size(0), self.dynamic_channels, -1).transpose(1, 2)

        f = self.F(self.alignment_pre.unsqueeze(1)).transpose(1, 2)

        e = self.v(torch.tanh(self.U(f) + self.T(g))).squeeze(-1) + p

        return e

    def get_alignment_energies(self, query, processed_memory,
                               attention_weights_cat):

        alignment = self.get_energies(query, processed_memory)

        return alignment

    def forward(self, attention_hidden_state, memory, processed_memory,
                attention_weights_cat, mask, log_alpha):
        """
        PARAMS
        ------
        attention_hidden_state: attention rnn last output
        memory: encoder outputs
        processed_memory: processed encoder outputs
        attention_weights_cat: previous and cummulative attention weights
        mask: binary mask for padded data
        """

        alignment = self.get_alignment_energies(attention_hidden_state, processed_memory, attention_weights_cat) # (batch, max_time)

        if mask is not None:
            alignment.data.masked_fill_(mask, self.score_mask_value)

        attention_weights = F.softmax(alignment, dim=1)

        self.alignment_pre = attention_weights

        attention_context = torch.bmm(attention_weights.unsqueeze(1), memory) # (batch, 1, dim)
        attention_context = attention_context.squeeze(1) # # (batch, dim)

        return attention_context, attention_weights

class StepwiseMonotonicAttention(nn.Module):
    """
    StepwiseMonotonicAttention (SMA)
    This attention is described in:
        M. He, Y. Deng, and L. He, "Robust Sequence-to-Sequence Acoustic Modeling with Stepwise Monotonic Attention for Neural TTS,"
        in Annual Conference of the International Speech Communication Association (INTERSPEECH), 2019, pp. 1293-1297.
        https://arxiv.org/abs/1906.00672
    See:
        https://gist.github.com/mutiann/38a7638f75c21479582d7391490df37c
        https://github.com/keonlee9420/Stepwise_Monotonic_Multihead_Attention
    """
    def __init__(self, attention_rnn_dim, embedding_dim, attention_dim,
                 attention_location_n_filters, attention_location_kernel_size):
        super(StepwiseMonotonicAttention, self).__init__()
        self.memory_layer = LinearNorm(embedding_dim, attention_dim, bias=False, w_init_gain='tanh')
        self.score_mask_value = -float("inf")

        """
        Args:
            sigmoid_noise: Standard deviation of pre-sigmoid noise.
                           Setting this larger than 0 will encourage the model to produce
                           large attention scores, effectively making the choosing probabilities
                           discrete and the resulting attention distribution one-hot.
        """
        sigmoid_noise=2.0

        self.tanh = nn.Tanh()
        self.v = nn.Linear(attention_dim, 1, bias=False)
        self.query_layer = LinearNorm(attention_rnn_dim, attention_dim,
                                      bias=False, w_init_gain='tanh')
        self.alignment = None # alignment in previous query time step
        self.sigmoid_noise = sigmoid_noise

    def init_attention(self, processed_memory):
        # Initial alignment with [1, 0, ..., 0]
        b, t, c = processed_memory.size()
        self.alignment = processed_memory.new_zeros(b, t)
        self.alignment[:, 0:1] = 1

    def stepwise_monotonic_attention(self, p_i, prev_alignment):
        """
        Compute stepwise monotonic attention
            - p_i: probability to keep attended to the last attended entry
            - Equation (8) in section 3 of the paper
        """
        pad = prev_alignment.new_zeros(prev_alignment.size(0), 1)
        alignment = prev_alignment * p_i + torch.cat((pad, prev_alignment[:, :-1] * (1.0 - p_i[:, :-1])), dim=1)
        return alignment

    def get_selection_probability(self, e, std):
        """
        Compute selecton/sampling probability `p_i` from energies `e`
            - Equation (4) and the tricks in section 2.2 of the paper
        """
        # Add Gaussian noise to encourage discreteness
        if self.training:
            noise = e.new_zeros(e.size()).normal_()
            e = e + noise * std

        # Compute selecton/sampling probability p_i
        # (batch, max_time)
        return torch.sigmoid(e)

    def get_probabilities(self, energies):
        # Selecton/sampling probability p_i
        p_i = self.get_selection_probability(energies, self.sigmoid_noise)
        
        # Stepwise monotonic attention
        alignment = self.stepwise_monotonic_attention(p_i, self.alignment)

        # (batch, max_time)
        self.alignment = alignment
        return alignment

    def get_energies(self, query, processed_memory,
                               attention_weights_cat):

        processed_query = self.query_layer(query.unsqueeze(1))
        energies = self.v(torch.tanh(processed_query + processed_memory))
        energies = energies.squeeze(-1)

        return energies

    def forward(self, attention_hidden_state, memory, processed_memory,
                attention_weights_cat, mask, log_alpha):
        """
        PARAMS
        ------
        attention_hidden_state: attention rnn last output
        memory: encoder outputs
        processed_memory: processed encoder outputs
        attention_weights_cat: previous and cummulative attention weights
        mask: binary mask for padded data
        """

        alignment = self.get_energies(attention_hidden_state, processed_memory, attention_weights_cat) # (batch, max_time)

        if mask is not None:
            alignment.data.masked_fill_(mask, self.score_mask_value)

        alignment = self.get_probabilities(alignment)

        # print(alignment)
        attention_weights = alignment #F.softmax(alignment, dim=1)
        attention_context = torch.bmm(attention_weights.unsqueeze(1), memory) # (batch, 1, dim)
        attention_context = attention_context.squeeze(1) # # (batch, dim)

        return attention_context, attention_weights


class GMMAttention(nn.Module):
    def __init__(self, attention_rnn_dim, embedding_dim, attention_dim,
                 attention_location_n_filters, attention_location_kernel_size, version='2'):
        super(GMMAttention, self).__init__()
        self.memory_layer = LinearNorm(embedding_dim, attention_dim, bias=False, w_init_gain='tanh')
        self.score_mask_value = -float("inf")

        self.gmm_version = version
        self.K = 5  # num mixture: follow paper https://arxiv.org/pdf/1910.10288.pdf
        self.eps = 1e-5
        self.mlp = nn.Sequential(
            nn.Linear(attention_rnn_dim, attention_dim, bias=True),
            nn.Tanh(),
            nn.Linear(attention_dim, 3*self.K))
    
    def init_attention(self, processed_memory):
        # No need to initialize alignment 
        # because GMM Attention is purely location based 
        # it has nothing to do with memory and t-1's alignment

        # Initial mu_pre with all zeros
        b, t, c = processed_memory.size()
        self.mu_prev = processed_memory.data.new(b, self.K, 1).zero_()
        j = torch.arange(0, processed_memory.size(1)).to(processed_memory.device)
        self.j = j.view(1, 1, processed_memory.size(1))  # [1, 1, T]

    def get_energies(self, query, processed_memory):
        '''
         Args:
            query: (batch, dim)
            processed_memory: (batch, max_time, dim)
        Returns:
            alignment: [batch, max_time]
        '''
        # Intermediate parameters (in Table 1)
        interm_params = self.mlp(query).view(query.size(0), -1, self.K)  # [B, 3, K]
        omega_hat, delta_hat, sigma_hat = interm_params.chunk(3, dim=1)  # Tuple

        # Each [B, K]
        omega_hat = omega_hat.squeeze(1)
        delta_hat = delta_hat.squeeze(1)
        sigma_hat = sigma_hat.squeeze(1)

        # Convert intermediate parameters to final mixture parameters
        # Choose version V0/V1/V2
        # Formula from https://arxiv.org/abs/1910.10288
        if self.gmm_version == '0':
            sigma = (torch.sqrt(torch.exp(-sigma_hat) / 2) + self.eps).unsqueeze(-1)  # [B, K, 1]
            delta = torch.exp(delta_hat).unsqueeze(-1)  # [B, K, 1]
            omega = torch.exp(omega_hat).unsqueeze(-1)  # [B, K, 1]
            Z = 1.0
        elif self.gmm_version == '1':
            sigma = (torch.sqrt(torch.exp(sigma_hat)) + self.eps).unsqueeze(-1)
            delta = torch.exp(delta_hat).unsqueeze(-1)
            omega = F.softmax(omega_hat, dim=-1).unsqueeze(-1)
            Z = torch.sqrt(2 * np.pi * sigma**2)
        elif self.gmm_version == '2':
            sigma = (F.softplus(sigma_hat) + self.eps).unsqueeze(-1)
            delta = F.softplus(delta_hat).unsqueeze(-1)
            omega = F.softmax(omega_hat, dim=-1).unsqueeze(-1)
            Z = torch.sqrt(2 * np.pi * sigma**2)

        mu = self.mu_prev + delta  # [B, K, 1]

        # Get alignment(phi in mathtype)
        alignment = omega / Z * torch.exp(-(self.j - mu)**2 / (sigma**2) / 2)  # [B, K ,T]
        alignment = torch.sum(alignment, 1)  # [B, T]

        # Update mu_prev
        self.mu_prev = mu

        return alignment
    
    def get_probabilities(self, energies):
        return energies

    def get_alignment_energies(self, query, processed_memory,
                               attention_weights_cat):

        alignment = self.get_energies(query, processed_memory)
        alignment = self.get_probabilities(alignment)

        return alignment

    def forward(self, attention_hidden_state, memory, processed_memory,
                attention_weights_cat, mask, log_alpha):
        """
        PARAMS
        ------
        attention_hidden_state: attention rnn last output
        memory: encoder outputs
        processed_memory: processed encoder outputs
        attention_weights_cat: previous and cummulative attention weights
        mask: binary mask for padded data
        """

        alignment = self.get_alignment_energies(attention_hidden_state, processed_memory, attention_weights_cat) # (batch, max_time)

        if mask is not None:
            alignment.data.masked_fill_(mask, self.score_mask_value)

        attention_weights = F.softmax(alignment, dim=1)
        attention_context = torch.bmm(attention_weights.unsqueeze(1), memory) # (batch, 1, dim)
        attention_context = attention_context.squeeze(1) # # (batch, dim)

        return attention_context, attention_weights