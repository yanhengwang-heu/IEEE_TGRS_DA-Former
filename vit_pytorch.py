import torch
import torch.nn as nn
import numpy as np
from einops import rearrange, repeat

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads, dim_head, dropout):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x, mask = None):
        # x:[b,n,dim]
        b, n, _, h = *x.shape, self.heads

        # get qkv tuple:([b,n,head_num*head_dim],[...],[...])
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        # split q,k,v from [b,n,head_num*head_dim] -> [b,head_num,n,head_dim]
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)
        # transpose(k) * q / sqrt(head_dim) -> [b,head_num,n,n]
        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        mask_value = -torch.finfo(dots.dtype).max

        # mask value: -inf
        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value = True)
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = mask[:, None, :] * mask[:, :, None]
            dots.masked_fill_(~mask, mask_value)
            del mask

        # softmax normalization -> attention matrix
        attn = dots.softmax(dim=-1)
        # value * attention matrix -> output
        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        # cat all output -> [b, n, head_num*head_dim]
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out

class Encoder_unshare(nn.Module):
    def __init__(self, channel, dim_c, dropout):
        super().__init__()
        self.band_to_embedding = nn.Linear(channel, dim_c)
        self.dropout = nn.Dropout(dropout)
    def forward(self,x):
        
        x = self.band_to_embedding(x)
        x = self.dropout(x)
        return x




class Encoder_share(nn.Module):
    def __init__(self, image_size, dim, depth, heads, dim_c, depth_c, heads_c, dim_head = 16, dim_head_c = 16, dropout=0.):
        super().__init__()
        
        patch_size = image_size**2

        self.layers = nn.ModuleList([])
        self.k_layers = nn.ModuleList([])
        self.patch_to_embedding = nn.Linear(patch_size, dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout))),
                Residual(PreNorm(dim, FeedForward(dim, 8, dropout = dropout)))
            ]))
        for _ in range(depth_c):
            self.k_layers.append(nn.ModuleList([
                Residual(PreNorm(dim_c, Attention(dim=dim_c, heads=heads_c, dim_head=dim_head_c, dropout = dropout))),
                Residual(PreNorm(dim_c, FeedForward(dim_c, 8, dropout = dropout)))
            ]))

        # self.mlp_head = nn.Sequential(
        #     nn.LayerNorm(dim),
        #     nn.Linear(dim, dim//2)
        # )
        
    def forward(self, x, mask = None):
        
        # transformer: x[b,n,dim] -> x[b,n,dim]
        # classification: using cls_token output
        for attn, ff in self.k_layers:
            x = attn(x, mask = mask)
            x = ff(x)
        x = rearrange(x, 'b n d -> b d n')
        x = self.patch_to_embedding(x)
        b,_,_=x.shape
        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        x = torch.cat((cls_tokens, x), dim = 1)
        for attn, ff in self.layers:
            x = attn(x, mask = mask)
            x = ff(x)
        
        cla_token= x[:,0,:]
        feature = x[:,1:,:]

        # MLP classification layer
        return cla_token,feature

class Decoder_share(nn.Module):
    def __init__(self, dim, dim_c, heads, dim_head, image_size, dropout):
        super().__init__()
        self.mlp_wh = nn.Sequential(
            Residual(PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout))),
            nn.LayerNorm(dim),
            nn.Linear(dim,image_size**2)
        )

        self.mlp_c = nn.Sequential(
            Residual(PreNorm(dim_c, FeedForward(dim_c, dim_c//2, dropout = dropout)))
            # nn.LayerNorm(dim),
            # nn.Linear(dim,patch_size)
        )

    def forward(self, x):
        x = self.mlp_wh(x)
        x = rearrange(x, 'b c n -> b n c')
        x = self.mlp_c(x)
        return x

class Decoder_unshare(nn.Module):
    def __init__(self,dim_c,channel):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.LayerNorm(dim_c),
            nn.Linear(dim_c,channel)
        )        
    def forward(self,x):
        x = self.mlp(x)
        return x


class Change_detection(nn.Module):
    def __init__(self,dim,dim_c,dropout):
        super().__init__()
        self.ln1 = nn.Linear(dim*dim_c,64)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.ln2 = nn.Linear(64,2)

    def forward(self, x1, x2):
        x = abs(x1-x2)
        x = rearrange(x,'b c d ->b (c d)')
        x = self.relu(self.ln1(x))
        x = self.ln2(x)
        return x


