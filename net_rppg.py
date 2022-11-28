import os
import torch.nn as nn
import torch.optim as optim
from base_networks import *
import torch.nn.functional as F
from torch.autograd import Variable
import torch
from resnet import resnet10 as Encoder
from collections import OrderedDict

class E1(nn.Module):
    def __init__(self, base_filter,video_length):
        super(E1, self).__init__()
        self.conv1=ConvBlock3D(64, base_filter, 3, 1,1, activation='relu', norm=None)
        self.res1= ResnetBlock3D(base_filter, kernel_size=3, stride=1, padding=1, bias=True, activation='relu', norm=None)
        self.ra=RABlock(base_filter, kernel_size=3, stride=1, padding=1, bias=True)
        self.res2= ResnetBlock3D(base_filter, kernel_size=3, stride=1, padding=1, bias=True, activation='relu', norm=None)
        self.gap=nn.AdaptiveAvgPool3d((video_length,1,1))
        self.cov1d=nn.Conv1d(in_channels=base_filter, out_channels=1,kernel_size=3,stride=1, padding=1)

    def forward(self, input):

        feat = self.conv1(input)
        feat=self.res1(feat)
        feat =self.ra(feat)
        feat=self.res2(feat)
        feat=self.gap(feat)
        feat=feat.squeeze(3)
        feat=feat.squeeze(3)
        feat=self.cov1d(feat)
        return feat

class E2(nn.Module):
    def __init__(self, base_filter,video_length):
        super(E2, self).__init__()
        self.conv1 = ConvBlock3D(64, base_filter, 3, 1, 1, activation='relu', norm=None)
        self.res1 = ResnetBlock3D(base_filter, kernel_size=3, stride=1, padding=1, bias=True, activation='relu',
                                  norm=None)
        self.ra = RABlock(base_filter, kernel_size=3, stride=1, padding=1, bias=True)
        self.res2 = ResnetBlock3D(base_filter, kernel_size=3, stride=1, padding=1, bias=True, activation='relu',
                                  norm=None)
        self.gap = nn.AdaptiveAvgPool3d((video_length, 1, 1))
        self.cov1d = nn.Conv1d(in_channels=base_filter, out_channels=1, kernel_size=3, stride=1, padding=1)

    def forward(self, input):
        feat = self.conv1(input)
        feat = self.res1(feat)
        feat = self.ra(feat)
        feat = self.res2(feat)
        feat = self.gap(feat)
        feat = feat.squeeze(3)
        feat = feat.squeeze(3)
        feat = self.cov1d(feat)
        return feat

class E3(nn.Module):
    def __init__(self, base_filter,video_length):
        super(E3, self).__init__()
        self.conv1 = ConvBlock3D(64, base_filter, 3, 1, 1, activation='relu', norm=None)
        self.res1 = ResnetBlock3D(base_filter, kernel_size=3, stride=1, padding=1, bias=True, activation='relu',
                                  norm=None)
        self.ra = RABlock(base_filter, kernel_size=3, stride=1, padding=1, bias=True)
        self.res2 = ResnetBlock3D(base_filter, kernel_size=3, stride=1, padding=1, bias=True, activation='relu',
                                  norm=None)
        self.gap = nn.AdaptiveAvgPool3d((video_length, 1, 1))
        self.cov1d = nn.Conv1d(in_channels=base_filter, out_channels=1, kernel_size=3, stride=1, padding=1)

    def forward(self, input):
        feat = self.conv1(input)
        feat = self.res1(feat)
        feat = self.ra(feat)
        feat = self.res2(feat)
        feat = self.gap(feat)
        feat = feat.squeeze(3)
        feat = feat.squeeze(3)
        feat = self.cov1d(feat)
        return feat

class E4(nn.Module):
    def __init__(self, base_filter,video_length):
        super(E4, self).__init__()
        self.conv1 = ConvBlock3D(64, base_filter, 3, 1, 1, activation='relu', norm=None)
        self.res1 = ResnetBlock3D(base_filter, kernel_size=3, stride=1, padding=1, bias=True, activation='relu',
                                  norm=None)
        self.ra = RABlock(base_filter, kernel_size=3, stride=1, padding=1, bias=True)
        self.res2 = ResnetBlock3D(base_filter, kernel_size=3, stride=1, padding=1, bias=True, activation='relu',
                                  norm=None)
        self.gap = nn.AdaptiveAvgPool3d((video_length, 1, 1))
        self.cov1d = nn.Conv1d(in_channels=base_filter, out_channels=1, kernel_size=3, stride=1, padding=1)

    def forward(self, input):
        feat = self.conv1(input)
        feat = self.res1(feat)
        feat = self.ra(feat)
        feat = self.res2(feat)
        feat = self.gap(feat)
        feat = feat.squeeze(3)
        feat = feat.squeeze(3)
        feat = self.cov1d(feat)
        return feat

class Gating(nn.Module):
    def __init__(self, base_filter,video_length):
        super(Gating, self).__init__()
        self.conv1=ConvBlock3D(base_filter, base_filter, 3, 1,1, activation='relu', norm=None)
        self.conv2=ConvBlock3D(base_filter, base_filter, 3, 1,1, activation='relu', norm=None)

        self.gap=nn.AdaptiveAvgPool3d((video_length,1,1))
        self.cov1d=nn.Conv1d(in_channels=base_filter, out_channels=4,kernel_size=3,stride=1, padding=1)
        self.act = nn.Softmax(dim=1)

    def forward(self, input):
        feat = self.conv1(input)
        feat=self.conv2(feat)
        feat=self.gap(feat)
        feat=feat.squeeze(3)
        feat=feat.squeeze(3)
        feat=self.cov1d(feat)
        feat = self.act(feat)
        return feat

class REA(nn.Module):
    def __init__(self, base_filter,video_length):
        super(REA, self).__init__()
        self.E1=E1(base_filter,video_length)
        self.E2=E2(base_filter,video_length)
        self.E3=E3(base_filter,video_length)
        self.E4=E4(base_filter,video_length)
        self.Gating=Gating(base_filter,video_length)
        self.encoder=Encoder()
        self.conv1d=nn.Conv1d(in_channels=1, out_channels=1,kernel_size=3,stride=1, padding=1)

    def freeze_model(self, model):
        for child in model.children():
            for param in child.parameters():
                param.requires_grad = False

    def forward(self, input):

        feat = self.encoder(input)

        B, C,T, H, W = input.size()
        cut_1 = feat[:, :, :, :int(H / 2), :int(W / 2)]
        cut_2 = feat[:, :, :, int(H / 2):, :int(W / 2)]
        cut_3 = feat[:, :, :, :int(H / 2), int(W / 2):]
        cut_4 = feat[:, :, :, int(H / 2):, int(W / 2):]

        cut_1 = self.E1(cut_1)
        cut_2 = self.E2(cut_2)
        cut_3 = self.E3(cut_3)
        cut_4 = self.E4(cut_4)
        gates = self.Gating(feat)

        cut_rppg1 = torch.mul(cut_1, gates[:, 0, :])
        cut_rppg2 = torch.mul(cut_2, gates[:, 1, :])
        cut_rppg3 = torch.mul(cut_3, gates[:, 2, :])
        cut_rppg4 = torch.mul(cut_4, gates[:, 3, :])
        rppg_fuse = cut_rppg1 + cut_rppg2 + cut_rppg3 + cut_rppg4
        rppg_fuse = self.conv1d(rppg_fuse)
        return rppg_fuse
