import os
import torch
import math
import numpy as np

from torch import nn
import json
import torch.nn.functional as F

from einops import rearrange, repeat
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

from torch.autograd import Variable, Function

import math
from dataclasses import dataclass
from typing import Union

class Round3(Function):
    @staticmethod
    def forward(ctx, input, training=False, inplace=False):
        output = torch.round(input)
        ctx.input = input
        return output

    @staticmethod
    def backward(ctx, grad_output):
        mask = ~(ctx.input==0)
        mask = Variable(mask).cuda().float()
        grad_output = grad_output * mask
        return grad_output, None, None

def __call_trunc_normal_(tensor, mean, std, a, b):
    def norm_cdf(x):
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                                                              "The distribution of values may be incorrect.",
                                                             stacklevel=2)

    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1)

        tensor.erfinv_()

        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        tensor.clamp_(min=a, max=b)
        return tensor

def trunc_normal_(tensor, mean=0., std=1.):
    __call_trunc_normal_(tensor, mean=mean, std=std, a=-std, b=std)



class WSCH(nn.Module):
    def __init__(self, hashcode_size=64, input_size=2048, hidden_size=1024, decode_size=None, cfg=None):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.hashcode_size = hashcode_size
        self.cfg = cfg
        
        self.En = Encoder(input_size, hidden_size)
        if decode_size==None:
            self.decode_size = hidden_size
        self.De = Decoder(input_size, decode_size, hashcode_size)
        

        self.hash = HashLayer(hidden_size, hashcode_size)
        
        # cluster centers
        self.register_buffer(
            "cluster_centers", 
            torch.zeros(cfg.n_clusters, hashcode_size)
        )
        self.initialized = False

    def forward(self, I, mask=None):
        B, L, D = I.shape
        if mask is None:
            mask = torch.zeros((B, L), dtype=torch.bool, device=I.device)
        
        I = self.En(I, mask)
        I = self.hash(I)

        # reconstruction
        recon = self.De(I, mask, L)
        
        I = torch.mean(I, dim=1)
        
        return recon, F.normalize(I, dim=-1)


    def inference(self, I, mask=None):
        batch = I.size(0)
        L = I.size(1)
        if mask==None:
            mask = torch.zeros((batch, L), dtype=torch.bool, device=I.device)

        I = self.En(I, mask)
        I = self.hash(I)

        I = torch.mean(I, dim=1)

        return F.normalize(I, dim=1)
    

    def compute_cluster_probability(self, features):
        
        similarity = F.cosine_similarity(
            features.unsqueeze(1),
            self.cluster_centers.unsqueeze(0),
            dim=2
        )
    
        similarity = similarity / self.cfg.clus_temperature
        pos_idx = similarity.argmax(dim=1)
        cluster_loss = F.cross_entropy(similarity, pos_idx)

        return cluster_loss 

        
class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.Norm = nn.LayerNorm(self.input_size)
        self.Norm1 = RMSNorm(self.hidden_size)
 
        self.in_proj = nn.Linear(self.input_size, self.hidden_size, bias=False)
        
        self.expand_f = 2
        self.expand_proj = nn.Linear(self.hidden_size, self.hidden_size * 2 * self.expand_f, bias=False)

        self.d_conv = 4
        self.conv = nn.Conv1d(in_channels=self.hidden_size * self.expand_f, out_channels=self.hidden_size * self.expand_f,
                              kernel_size=self.d_conv, bias=False, 
                              groups=self.hidden_size,
                              padding=self.d_conv - 1)

        self.ssm = SSD(hidden_size*2)
        self.out_proj = nn.Linear(self.hidden_size * self.expand_f, self.hidden_size, bias=True)


        self.norm1 = nn.LayerNorm(self.hidden_size)

        self.att1 = Self_Attention(self.hidden_size)


    def forward(self, x, mask):

        B, N, C = x.shape
        x = x[~mask].reshape(B, -1, C) # ~mask means visible

        x = self.Norm(x)
        x = self.in_proj(x)

        t = x
        x = self.Norm1(x)

        x, z = self.expand_proj(x).chunk(2, dim=-1)

        _, L, _ = x.shape
        x = self.conv(x.transpose(1, 2))[:,:,:L].transpose(1, 2)

        x = self.ssm(x)
        
        z = F.silu(z)

        x = x * z
        
        x = self.out_proj(x)

        x = x + t

        x = x + self.att1(self.norm1(x))
        
        return x

class Decoder(nn.Module):
    def __init__(self, output_size, hidden_size, hashcode_size):
        super().__init__()
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.hashcode_size = hashcode_size

        self.mask_token = nn.Parameter(torch.zeros(1, 1, hashcode_size)) 

        self.encoder_to_decoder = nn.Linear(hashcode_size, hidden_size, bias=False)

        self.norm = nn.LayerNorm(self.hidden_size)

        self.att = Self_Attention(self.hidden_size)

        self.out_proj = nn.Linear(hidden_size, output_size, bias=True)

    def forward(self, x, mask, L):
        B = x.size(0)

        padding = torch.zeros(B, L, self.hashcode_size).type_as(x).to(x.device).detach()            
        padding_vis = padding[~mask].reshape(B, -1, self.hashcode_size)
        padding_mask = padding[mask].reshape(B, -1, self.hashcode_size)
        x = torch.cat([x + padding_vis, self.mask_token + padding_mask], dim=1)

        x = self.encoder_to_decoder(x)
 
        x = x + self.att(self.norm(x))
        
        x = self.out_proj(x)
        x = x[:, -padding_mask.shape[1]:]

        return x


class CalculateAttention(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, Q, K, V):
        attention = torch.matmul(Q, torch.transpose(K, -1, -2))
        # use mask
        attention = torch.softmax(attention / torch.sqrt(torch.tensor(K.size(-1))), dim=-1)
        attention = torch.matmul(attention, V)
        return attention

class Self_Attention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.all_head_size = hidden_size
        self.num_heads = 4
        self.h_size = self.all_head_size // self.num_heads

        self.linear_q = nn.Linear(self.hidden_size, self.all_head_size, bias=False)
        self.linear_k = nn.Linear(self.hidden_size, self.all_head_size, bias=False)
        self.linear_v = nn.Linear(self.hidden_size, self.all_head_size, bias=False)
        self.linear_output = nn.Linear(self.all_head_size, self.hidden_size)

    def forward(self, x):

        batch_size = x.size(0)
        q_s = self.linear_q(x).view(batch_size, -1, self.num_heads, self.h_size).transpose(1, 2)
        k_s = self.linear_k(x).view(batch_size, -1, self.num_heads, self.h_size).transpose(1, 2)
        v_s = self.linear_v(x).view(batch_size, -1, self.num_heads, self.h_size).transpose(1, 2)

        attention = CalculateAttention()(q_s, k_s, v_s)
        attention = attention.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.h_size)

        return self.linear_output(attention)

    def cross(self, x, y):

        batch_size = x.size(0)
        q_s = self.linear_q(x).view(batch_size, -1, self.num_heads, self.h_size).transpose(1, 2)
        k_s = self.linear_k(y).view(batch_size, -1, self.num_heads, self.h_size).transpose(1, 2)
        v_s = self.linear_v(y).view(batch_size, -1, self.num_heads, self.h_size).transpose(1, 2)

        attention = CalculateAttention()(q_s, k_s, v_s)
        attention = attention.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.h_size)

        return self.linear_output(attention)



from math import sqrt
import torch
import torch.nn as nn

class HashLayer(nn.Module):

    def __init__(self, hidden_size, hashcode_size):
        super(HashLayer, self).__init__()
        self.linear = nn.Linear(hidden_size, hashcode_size)
        self.ln = nn.LayerNorm(hashcode_size)

    def forward(self, x):
        x = self.binary_tanh_unit(self.ln(self.linear(x)))
        return x

    def binary_tanh_unit(self, x):
        y = self.hard_sigmoid(x)
        out = 2. * Round3.apply(y) - 1.
        return out
    
    def hard_sigmoid(self, x):
        y = (x + 1.) / 2.
        y[y > 1] = 1
        y[y < 0] = 0
        return y


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()

        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight

        return output


class SSD(nn.Module):
    def __init__(self,
        d_model,
        d_state=64,
        conv_init=None,
        expand=2,
        headdim=128,
        ngroups=1,
        A_init_range=(1, 16),
        dt_min=0.001,
        dt_max=0.1,
        dt_init_floor=1e-4,
        dt_limit=(0.0, float("inf")),
        chunk_size=25,):
        super().__init__()

        self.d_model = d_model
        self.d_state = d_state
        self.headdim = headdim
        self.ngroups = ngroups
        self.nheads = d_model // self.headdim
        self.chunk_size = chunk_size
        self.dt_limit = dt_limit

        self.x_proj = nn.Linear(d_model, self.nheads + 2 * self.d_state, bias=False)

        dt = torch.exp(
            torch.zeros(self.nheads) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        )

        self.dt_bias = nn.Parameter(torch.zeros(self.nheads))

        A = torch.empty(self.nheads, dtype=torch.float32).uniform_(*A_init_range)
        A_log = torch.log(A).to(dtype=torch.float32)
        self.A_log = nn.Parameter(A_log)

        self.D = nn.Parameter(torch.ones(self.nheads))
    
    def forward(self, x):
        batch, seqlen, dim = x.shape

        dtBC = self.x_proj(x)  # (B, L, d_in_proj)
        A = -torch.exp(self.A_log)  # (nheads) or (d_inner, d_state)

        dt, B, C = torch.split(dtBC, [self.nheads, self.ngroups * self.d_state, self.ngroups * self.d_state], dim=-1)

        dt = F.softplus(dt + self.dt_bias)  # (B, L, nheads)

        y = ssd_minimal_discrete(
            rearrange(x, "b l (h p) -> b l h p", p=self.headdim) * dt.unsqueeze(-1),
            A * dt,
            rearrange(B, "b l (g n) -> b l g n", g=self.ngroups),
            rearrange(C, "b l (g n) -> b l g n", g=self.ngroups),
            seqlen,
        )

        y = rearrange(y, "b l h p -> b l (h p)")
        D = repeat(self.D, "h -> (h p)", p=self.headdim)
        y = y + x * D

        return y

def segsum(x):
    """More stable segment sum calculation."""
    T = x.size(-1)
    x = repeat(x, "... d -> ... d e", e=T)
    mask = torch.tril(torch.ones(T, T, device=x.device, dtype=bool), diagonal=-1)
    x = x.masked_fill(~mask, 0)
    x_segsum = torch.cumsum(x, dim=-2)
    mask = torch.tril(torch.ones(T, T, device=x.device, dtype=bool), diagonal=0)
    x_segsum = x_segsum.masked_fill(~mask, -torch.inf)
    return x_segsum

def ssd_minimal_discrete(X, A, B, C, L, initial_states=None):
    X, A, B, C = [rearrange(x, "b (c l) ... -> b c l ...", l=L) for x in (X, A, B, C)]

    A = rearrange(A, "b c l h -> b h c l")
    A_cumsum = torch.cumsum(A, dim=-1)

    # 1. Compute the output for each intra-chunk (diagonal blocks)
    L = torch.exp(segsum(A))
    Y_diag  = torch.einsum("bclhn,bcshn,bhcls,bcshp->bclhp", C, B, L, X)

    # 2. Compute the state for each intra-chunk
    # (right term of low-rank factorization of off-diagonal blocks; B terms)
    decay_states = torch.exp((A_cumsum[:, :, :, -1:] - A_cumsum))
    states = torch.einsum("bclhn,bhcl,bclhp->bchpn", B, decay_states, X)

    # 3. Compute the inter-chunk SSM recurrence; produces correct SSM states at chunk boundaries
    # (middle term of factorization of off-diag blocks; A terms)
    if initial_states is None:
        initial_states = torch.zeros_like(states[:, :1])
    states = torch.cat([initial_states, states], dim=1)
    decay_chunk = torch.exp(segsum(F.pad(A_cumsum[:, :, :, -1], (1, 0))))
    new_states = torch.einsum("bhzc,bchpn->bzhpn", decay_chunk, states)
    states = new_states[:, :-1]

    # 4. Compute state -> output conversion per chunk
    # (left term of low-rank factorization of off-diagonal blocks; C terms)
    state_decay_out = torch.exp(A_cumsum)
    Y_off = torch.einsum('bclhn,bchpn,bhcl->bclhp', C, states, state_decay_out)

    # Add output of intra-chunk and inter-chunk terms (diagonal and off-diagonal blocks)
    Y = rearrange(Y_diag+Y_off, "b c l h p -> b (c l) h p")

    return Y

