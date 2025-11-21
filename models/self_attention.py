import torch
import torch.nn as nn
import numpy as np

class SelfAttention(nn.Module):
    """ 
    Self Attention Layer adapted from GACN to PyTorch.
    Input shape: [Batch, Channel, Time/Sequence]
    """
    def __init__(self, in_dim):
        super(SelfAttention, self).__init__()
        self.chanel_in = in_dim
        
        # Sử dụng Spectral Norm giống code GACN gốc để ổn định huấn luyện
        self.f_conv = nn.utils.spectral_norm(nn.Conv1d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1))
        self.g_conv = nn.utils.spectral_norm(nn.Conv1d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1))
        self.h_conv = nn.utils.spectral_norm(nn.Conv1d(in_channels=in_dim, out_channels=in_dim, kernel_size=1))
        
        # Tham số gamma trainable, khởi tạo bằng 0
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
        inputs :
            x : input feature maps [Batch, Channel, Time]
        returns :
            out : self attention value + input feature
            attention: [Batch, Time, Time]
        """
        m_batchsize, C, width = x.size()
        
        # Tính f(x), g(x), h(x)
        # Permute để đưa về dạng [Batch, Time, Channel] cho phép nhân ma trận
        f = self.f_conv(x).permute(0, 2, 1) # [B, T, C']
        g = self.g_conv(x).permute(0, 2, 1) # [B, T, C']
        h = self.h_conv(x).permute(0, 2, 1) # [B, T, C]
        
        # Tính Energy (Scores) s = g * f^T
        s = torch.bmm(g, f.permute(0, 2, 1)) # [B, T, T]
        
        # Tính Attention Map (beta)
        beta = self.softmax(s) # Đây là cái bạn cần lấy ra
        
        # Tính Output o = beta * h
        o = torch.bmm(beta, h) # [B, T, C]
        o = o.permute(0, 2, 1) # [B, C, T]
        
        # Kết quả: gamma * o + x
        out = self.gamma * o + x
        
        return out, beta