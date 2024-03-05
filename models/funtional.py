import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()

        # 创建一个足够长的位置编码
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0).transpose(0, 1)
        # 注册为常量，不需要梯度
        self.register_buffer('pe', pe)

    def forward(self, x):
        # 将位置编码添加到输入的嵌入中
        x = x + self.pe[:x.size(0), :]
        return x

class RoPEPositionalEncoding(nn.Module):
    
    def __init__(self, d_model, max_len=512):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        cos_pos = []
        sin_pos = []
        for i in range(max_len):
            cos_pos.append(self.cos_m_theta(i,d_model))
            sin_pos.append(self.sin_m_theta(i,d_model))

        print(len(cos_pos))
        self.cos_pos = torch.stack(cos_pos)
        self.sin_pos = torch.stack(sin_pos)

    def forward(self, input):
        # input_shape = (B, N, D)
        new_input = torch.zeros_like(input)
        B = input.shape[0]

        cos_pos = self.cos_pos.unsqueeze(0)
        cos_pos = cos_pos.repeat(B,1,1)
        sin_pos = self.sin_pos.unsqueeze(0)
        sin_pos = sin_pos.repeat(B,1,1)
        # 对于张量的最后一个维度，我们要交换奇数和偶数索引位置的元素并改变符号
        # 偶数索引位置（从0开始计数），在新张量中对应的是奇数索引位置
        new_input[..., 1::2] = input[..., :-1:2]

        # 奇数索引位置，在新张量中对应的是偶数索引位置
        new_input[..., 0::2] = -input[..., 1::2]
        
        out = input*cos_pos+new_input*sin_pos
        return out

    def cos_m_theta(self,m,d_model):
        cos_tensor = torch.ones(d_model)
        for i in range(len(cos_tensor)):
            cos_tensor[i] = math.cos(m*self.theta(i))
        return cos_tensor
    def sin_m_theta(self,m,d_model):
        sin_tensor = torch.ones(d_model)
        for i in range(len(sin_tensor)):
            sin_tensor[i] = math.sin(m*self.theta(i))
        return sin_tensor

    def theta(self,t):
        return 10000**(-2*t/self.d_model)
