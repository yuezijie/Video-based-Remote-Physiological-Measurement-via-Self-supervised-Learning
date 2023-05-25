import os
import torch.nn as nn
import torch.optim as optim
from base_networks import *
import torch.nn.functional as F
import torch
import numpy as np
import collections


class Net(nn.Module):
    def __init__(self, base_filter,video_length):
        super(Net, self).__init__()

        self.conv1=ConvBlock3D(3, base_filter, 3, 1,1, activation='lrelu', norm=None)

        res1= [
            ResnetBlock3D(base_filter, kernel_size=3, stride=1, padding=1, bias=True, activation='lrelu', norm=None) \
            for _ in range(2)]
        self.res1= nn.Sequential(*res1)

        self.downsample_1=nn.Upsample(scale_factor=(1,0.5,0.5))
        self.downsample_2=nn.Upsample(scale_factor=(1,0.25,0.25))

        self.upsample_1 = nn.Upsample(scale_factor=(1,2,2))
        self.upsample_2 = nn.Upsample(scale_factor=(1,4,4))

        self.mod_gap=nn.AdaptiveAvgPool3d((video_length,1,1))
        self.mod_conv1d_1=nn.Conv1d(in_channels=base_filter, out_channels=1,kernel_size=3,stride=1, padding=1)
        self.mod_res1d_1=ResnetBlock1D(2, kernel_size=3, stride=1, padding=1, bias=True, activation='relu', norm=None)
        self.mod_lstm_1=nn.LSTM(2, 1, num_layers=1, batch_first=True)
        # , bidirectional = True,
        self.mod_conv1d_2=nn.Conv1d(in_channels=base_filter, out_channels=1,kernel_size=3,stride=1, padding=1)
        self.mod_res1d_2=ResnetBlock1D(2, kernel_size=3, stride=1, padding=1, bias=True, activation='relu', norm=None)
        self.mod_lstm_2=nn.LSTM(2, 1, num_layers=1, batch_first=True)

        self.mod_conv1d_3=nn.Conv1d(in_channels=base_filter, out_channels=1,kernel_size=3,stride=1, padding=1)
        self.mod_res1d_3=ResnetBlock1D(2, kernel_size=3, stride=1, padding=1, bias=True, activation='relu', norm=None)
        self.mod_lstm_3=nn.LSTM(2, 1, num_layers=1, batch_first=True )

        self.finalconv1=ConvBlock3D(base_filter*3, 3, 1, 1,padding=0, norm=None)


    def forward(self, input,ratio):


        B, T, C, H, W = input.size()
        input=input.transpose(1,2)
        feat = self.conv1(input)
        feat=self.res1(feat)
        featdown1=self.downsample_1(feat)
        featdown2 =self.downsample_2(feat)

        mod1feat=self.mod_gap(feat)
        mod1feat=mod1feat.squeeze(3)
        mod1feat=mod1feat.squeeze(3)

        mod1feat=self.mod_conv1d_1(mod1feat)
        ratio= ratio.unsqueeze(1)

        fuse1=torch.cat((mod1feat, ratio), 1)

        mod1feat = self.mod_res1d_1(fuse1)
        mod1feat=mod1feat.transpose(1,2)
        mod1feat = self.mod_lstm_1(mod1feat)[0]
        mod1feat=mod1feat.view(B,1,T,1,1)
        aftermod_1=torch.mul(feat,mod1feat)

        mod2feat=self.mod_gap(featdown1)
        mod2feat=mod2feat.squeeze(3)
        mod2feat=mod2feat.squeeze(3)
        mod2feat=self.mod_conv1d_2(mod2feat)
        fuse2=torch.cat((mod2feat, ratio), 1)
        mod2feat = self.mod_res1d_2(fuse2)
        mod2feat=mod2feat.transpose(1,2)
        mod2feat = self.mod_lstm_2(mod2feat)[0]
        mod2feat=mod2feat.view(B,1,T,1,1)
        aftermod_2=torch.mul(featdown1,mod2feat)
        aftermod_2=self.upsample_1(aftermod_2)

        mod3feat=self.mod_gap(featdown2)
        mod3feat=mod3feat.squeeze(3)
        mod3feat=mod3feat.squeeze(3)
        mod3feat=self.mod_conv1d_3(mod3feat)
        fuse3=torch.cat((mod3feat, ratio), 1)
        mod3feat = self.mod_res1d_3(fuse3)
        mod3feat=mod3feat.transpose(1,2)
        mod3feat = self.mod_lstm_3(mod3feat)[0]
        mod3feat=mod3feat.view(B,1,T,1,1)
        aftermod_3=torch.mul(featdown2,mod3feat)
        aftermod_3=self.upsample_2(aftermod_3)

        finalfeat=torch.cat((aftermod_1, aftermod_2,aftermod_3), 1)
        finalfeat=self.finalconv1(finalfeat)

        return finalfeat