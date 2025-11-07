import torch
import torch.nn as nn
import math


class Linear(nn.Module):
    '''
    Construct a linear transformation module. This function should accept the following parameters:
    in_features: int final dimension of the input
    out_features: int final dimension of the output
    device: torch.device | None = None Device to store the parameters on
    dtype: torch.dtype | None = None Data type of the parameters
    '''
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features, device=device, dtype=dtype))
        self._reset_parameters()
    
    def _reset_parameters(self):
        std = math.sqrt(2.0 / (self.in_features + self.out_features))
        nn.init.trunc_normal_(self.weight, mean=0.0, std=std, a=-3*std, b=3*std)
       
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.weight.T


class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = nn.Parameter(torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype))
        self._reset_parameters()
    
    def _reset_parameters(self):
        nn.init.trunc_normal_(self.weight, mean=0.0, std=1.0, a=-3.0, b=3.0)
       
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.weight[token_ids]


class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-5, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.weight = nn.Parameter(torch.empty(d_model, device=device, dtype=dtype))
        self._reset_parameters()
    
    def _reset_parameters(self):
        nn.init.ones_(self.weight)
       
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        rms = torch.sqrt(variance+self.eps)
        normalized = x / rms
        res = normalized * self.weight 
        return res.to(in_dtype)

class SwiGLU(nn.Module):
    def __init__(self, d_model, d_ff, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)
        self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype)
    
       
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w1_x = self.w1(x)
        w3_x = self.w3(x)
        silu_x1_x = w1_x * torch.sigmoid(w1_x)
        gated = silu_x1_x * w3_x
        output = self.w2(gated)

        return output


def make_swiglu_d_ff(d_model):
    target_d_ff = int(8*d_model/3)
    d_ff = ((target_d_ff+31)//64)*64
    return d_ff


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, device, dtype):
        super().__init__()
        self.d_model = d_model
        self.d_ff = make_swiglu_d_ff(d_model)
        self.swiglu = SwiGLU(d_model, self.d_ff, device, dtype)
    
    def forward(self, x):
        return self.swiglu(x)