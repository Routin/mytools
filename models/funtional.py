import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_len=512):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=0.1)

        # 计算位置编码
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class RotaryPositionEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE)模块
    
    该模块根据给定的序列长度和模型维度，生成对应的RoPE编码。
    
    Args:
        dim (int): 模型维度，即每个token的特征数。
        max_seq_len (int, optional): 最大序列长度，默认为512。
    """
    
    def __init__(self, dim, max_seq_len=512):
        super(RotaryPositionEmbedding, self).__init__()
        # 计算频率
        self.dim = dim
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        self.max_seq_len = max_seq_len
    
    def forward(self, positions, batch_first=True):
        """
        Args:
            positions (torch.Tensor): 位置索引的张量，大小为 [batch_size, seq_len] 或 [seq_len, batch_size]。
            batch_first (bool, optional): 如果为True，输入输出的张量形状为[batch_size, seq_len, dim]，
                                          否则为[seq_len, batch_size, dim]。默认为True。
        
        Returns:
            torch.Tensor: 经过RoPE编码后的张量。
        """
        seq_len = positions.size(1) if batch_first else positions.size(0)
        positions = positions.view(-1)  # 展平位置信息以便计算
        
        # 创建维度上的位置编码
        sinusoid_inp = torch.ger(positions, self.inv_freq)
        pos_emb = torch.cat((sinusoid_inp.sin(), sinusoid_inp.cos()), dim=-1)
        
        # 重塑为原始的维度
        if batch_first:
            pos_emb = pos_emb.view(-1, seq_len, self.dim)
        else:
            pos_emb = pos_emb.view(seq_len, -1, self.dim)
        
        return pos_emb

def apply_rotary_pos_emb(x, sin_cos_emb):
    """应用RoPE编码到特征上。
    
    Args:
        x (torch.Tensor): 输入特征张量，形状为 [batch_size, seq_len, dim] 或 [seq_len, batch_size, dim]。
        sin_cos_emb (torch.Tensor): 由RoPE模块生成的编码张量，形状需与输入特征匹配。
    
    Returns:
        torch.Tensor: 应用了RoPE编码的特征张量。
    """
    half_dim = sin_cos_emb.shape[-1] // 2
    sin_emb = sin_cos_emb[..., :half_dim]
    cos_emb = sin_cos_emb[..., half_dim:]
    
    # 将sin和cos编码应用到特征上
    x_cos = x * cos_emb
    x_sin = torch.roll(x, shifts=1, dims=-1) * sin_emb
    return x_cos + x_sin
        

        
    
    
