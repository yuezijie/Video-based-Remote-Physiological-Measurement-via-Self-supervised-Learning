import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class CalculateNormPSD(nn.Module):
    def __init__(self, Fs, high_pass, low_pass):
        super().__init__()
        self.Fs = Fs
        self.high_pass = high_pass
        self.low_pass = low_pass

    def forward(self, x, zero_pad=0):
        x = x - torch.mean(x)
        if zero_pad > 0:
            L = x.shape[-1]
            x = F.pad(x, (int(zero_pad/2*L), int(zero_pad/2*L)), 'constant', 0)

        # Get PSD
        x = torch.view_as_real(torch.fft.rfft(x, dim=-1, norm='forward'))
        x = torch.add(x[:, :, 0] ** 2, x[:, :, 1] ** 2)

        # Normalize PSD
        x = x / torch.sum(x, dim=-1, keepdim=True)
        return x
