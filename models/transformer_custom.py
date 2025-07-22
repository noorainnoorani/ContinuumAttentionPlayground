import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k, dropout=0.1):
        super(ScaledDotProductAttention, self).__init__()
        # Scale factor for dot-product attention: sqrt(d_k)
        # Used to prevent large dot products that push softmax into regions with small gradients
        self.scale = nn.Parameter(torch.sqrt(torch.FloatTensor([d_k])), requires_grad=False)
        self.dropout = nn.Dropout(dropout)

    def custom_softmax(self, x, coords=None, dim=-1):
        # Numerically stable softmax: subtract max for stability
        # exp_x = exp(x - max(x))
        exp_x = torch.exp(x - x.max(dim=dim, keepdim=True)[0])
        # If coords is provided, apply custom reweighting along sequence dimension
        # Otherwise, standard softmax: softmax_x = exp_x / sum(exp_x)
        if coords is not None:
            # Custom denominator: sum over 0.5 * coords * (exp_x[...,1:] + exp_x[...,:-1])
            # This reweights the softmax by the distances between coordinates
            softmax_x = exp_x / (0.5*coords*(exp_x[...,1:]+exp_x[...,:-1])).sum(dim=dim, keepdim=True)
        else:
            softmax_x = exp_x / exp_x.sum(dim=dim, keepdim=True)
        return softmax_x

    def forward(self, query, key, value, coords, key_padding_mask=None):
        """
        query: (batch, n_heads, seq_len, d_k)
        key:   (batch, n_heads, seq_len, d_k)
        value: (batch, n_heads, seq_len, d_k)
        coords: (seq_len, domain_dim, ...) or similar
        key_padding_mask: (batch, seq_len)
        """
        # Compute scaled dot-product attention scores
        # scores_{l,s} = (Q_l · K_s^T) / sqrt(d_k)
        # scores shape: (batch, n_heads, seq_len, seq_len)
        scores = torch.einsum("bhld,bhsd->bhls", query, key) / self.scale

        # If domain_dim == 1, compute pairwise distances between coordinates
        # coords: (seq_len, 1, ...) -> (1, 1, 1, seq_len-1) after processing
        if coords.shape[1]==1:
            # Compute absolute differences between consecutive coordinates
            coords = torch.abs(coords[1:,...] - coords[:-1,...])
            # Rearrange for broadcasting with scores[...,1:]
            coords = coords.permute(1,2,0).unsqueeze(0)
        else:
            coords = None

        # Apply key padding mask: set masked positions to -inf so softmax ~ 0
        if key_padding_mask is not None:
            scores = scores.masked_fill(key_padding_mask.unsqueeze(1).unsqueeze(2), float('-inf'))

        # Compute (custom) softmax over last dimension (sequence)
        # attention_weights_{l,s} = softmax(scores_{l,s})
        attention_weights = self.custom_softmax(scores, coords=coords, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # Weighted sum of values:
        # If coords is not None, reweight value using coords and average two directions
        if coords is not None:
            # value1: coords * value[...,1:,:], output1: sum over s (attention_weights[...,1:] * value1)
            value1 = coords.permute(0,1,3,2)*value[...,1:,:]
            output1 = torch.einsum("bhls,bhsd->bhld", attention_weights[...,1:], value1)
            # value2: coords * value[...,:-1,:], output2: sum over s (attention_weights[...,:-1] * value2)
            value2 = coords.permute(0,1,3,2)*value[...,:-1,:]
            output2 = torch.einsum("bhls,bhsd->bhld", attention_weights[...,:-1], value2)
            # Average the two outputs
            output = 0.5*(output1 + output2)
        else:
            # Standard attention output: sum_s (attention_weights_{l,s} * value_s)
            output = torch.einsum("bhls,bhsd->bhld", attention_weights, value)

        return output
    
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % nhead == 0, "d_model must be divisible by nhead"

        self.d_model = d_model
        self.nhead = nhead
        self.d_k = d_model // nhead  # Each head's dimension

        # Linear projections for queries, keys, and values
        # W_q, W_k, W_v: (d_model, nhead * d_k)
        self.W_q = nn.Linear(d_model, nhead*self.d_k)
        self.W_k = nn.Linear(d_model, nhead*self.d_k)
        self.W_v = nn.Linear(d_model, nhead*self.d_k)
        self.scaled_dot_product_attention = ScaledDotProductAttention(self.d_k, dropout=dropout)
        # Output linear layer to combine heads
        self.W_o = nn.Linear(nhead*self.d_k, d_model)

    def split_heads(self, x):
        """
        Splits the last dimension into (nhead, d_k) and transposes for attention.
        Input:  x: (batch_size, seq_len, d_model)
        Output: (batch_size, nhead, seq_len, d_k)
        """
        batch_size = x.shape[0]
        # Reshape: (batch_size, seq_len, nhead, d_k), then transpose to (batch_size, nhead, seq_len, d_k)
        return x.view(batch_size, -1, self.nhead, self.d_k).transpose(1, 2)

    def combine_heads(self, x):
        """
        Combines the heads back to a single vector per position.
        Input:  x: (batch_size, nhead, seq_len, d_k)
        Output: (batch_size, seq_len, nhead * d_k)
        """
        batch_size = x.shape[0]
        # Transpose to (batch_size, seq_len, nhead, d_k), then reshape
        return x.transpose(1, 2).contiguous().view(batch_size, -1, self.nhead * self.d_k)

    def forward(self, x, coords_x, mask=None):
        """
        x:        (batch_size, seq_len, d_model)
        coords_x: (seq_len, domain_dim, ...)
        mask:     (batch_size, seq_len)
        """
        # Linear projections: project input to queries, keys, values for all heads
        # Q = x @ W_q, K = x @ W_k, V = x @ W_v
        # Shapes: (batch_size, seq_len, nhead * d_k)
        Q = self.split_heads(self.W_q(x))  # (batch_size, nhead, seq_len, d_k)
        K = self.split_heads(self.W_k(x))
        V = self.split_heads(self.W_v(x))

        # Compute multi-head attention using scaled dot-product attention
        # For each head, computes:
        #   Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V
        attn_output = self.scaled_dot_product_attention(Q, K, V, coords_x, mask)  # (batch_size, nhead, seq_len, d_k)

        # Concatenate heads and project
        # Output: (batch_size, seq_len, d_model)
        output = self.W_o(self.combine_heads(attn_output))
        return output
    
class FeedForward(nn.Module):
    def __init__(self, d_model, dim_feedforward, activation):
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, dim_feedforward)
        self.fc2 = nn.Linear(dim_feedforward, d_model)
        self.activation = getattr(F, activation)

    def forward(self, x):
        return self.fc2(self.activation(self.fc1(x)))
    
class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1, activation="relu", norm_first=True, do_layer_norm=True, dim_feedforward=2048, batch_first=True):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, nhead)
        self.feed_forward = FeedForward(d_model, dim_feedforward, activation)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, coords_x, mask=None):
        attn_output = self.self_attn(x, coords_x)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers=1):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(num_layers)])

    def forward(self, x, coords_x, mask=None):
        for layer in self.layers:
            x = layer(x, coords_x, mask=mask)
        return x
