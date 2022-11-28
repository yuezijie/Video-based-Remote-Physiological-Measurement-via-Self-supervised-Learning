import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import torch.fft
from torch_utils import CalculateNormPSD
import math
from scipy.interpolate import Akima1DInterpolator


def compute_power_spectrum(signal, Fs, zero_pad=None):
    if zero_pad is not None:
        L = len(signal)
        signal = np.pad(signal, (int(zero_pad/2*L), int(zero_pad/2*L)), 'constant')
    freqs = np.fft.fftfreq(len(signal), 1 / Fs) * 60  # in bpm
    ps = np.abs(np.fft.fft(signal))**2
    cutoff = len(freqs)//2
    freqs = freqs[:cutoff]
    ps = ps[:cutoff]
    return freqs, ps


def predict_heart_rate(signal, Fs, min_hr=40., max_hr=180., method='fast_ideal'):

    if method == 'ideal':
        """ Zero-pad in time domain for ideal interp in freq domain
        """
        signal = signal - np.mean(signal)
        freqs, ps = compute_power_spectrum(signal, Fs, zero_pad=100)
        cs = Akima1DInterpolator(freqs, ps)
        max_val = -np.Inf
        interval = 0.1
        min_bound = max(min(freqs), min_hr)
        max_bound = min(max(freqs), max_hr) + interval
        for bpm in np.arange(min_bound, max_bound, interval):
            cur_val = cs(bpm)
            if cur_val > max_val:
                max_val = cur_val
                max_bpm = bpm
        return max_bpm

    elif method == 'fast_ideal':
        """ Zero-pad in time domain for ideal interp in freq domain
        """
        signal = signal - np.mean(signal)
        freqs, ps = compute_power_spectrum(signal, Fs, zero_pad=100)
        freqs_valid = np.logical_and(freqs >= min_hr, freqs <= max_hr)
        freqs = freqs[freqs_valid]
        ps = ps[freqs_valid]
        max_ind = np.argmax(ps)
        if 0 < max_ind < len(ps)-1:
            inds = [-1, 0, 1] + max_ind
            x = ps[inds]
            f = freqs[inds]
            d1 = x[1]-x[0]
            d2 = x[1]-x[2]
            offset = (1 - min(d1,d2)/max(d1,d2)) * (f[1]-f[0])
            if d2 > d1:
                offset *= -1
            max_bpm = f[1] + offset
        elif max_ind == 0:
            x0, x1 = ps[0], ps[1]
            f0, f1 = freqs[0], freqs[1]
            max_bpm = f0 + (x1 / (x0 + x1)) * (f1 - f0)
        elif max_ind == len(ps) - 1:
            x0, x1 = ps[-2], ps[-1]
            f0, f1 = freqs[-2], freqs[-1]
            max_bpm = f0 + (x1 / (x0 + x1)) * (f1 - f0)
        return max_bpm

class FRCL(nn.Module):
    def __init__(self,Fs,min_hr ,max_hr):
        super(FRCL, self).__init__()
        self.Fs=Fs
        self.min_hr=min_hr
        self.max_hr=max_hr

    def forward(self, neg_rppgarr,pos_rppg1,pos_rppg2,ratio_array):
        loss=0
        count=0

        poshr1= predict_heart_rate(pos_rppg1[0].detach().cpu().numpy(),self.Fs,self.min_hr,self.max_hr)

        poshr2= predict_heart_rate(pos_rppg2[0].detach().cpu().numpy(),self.Fs,self.min_hr,self.max_hr)
        for i in range(len(neg_rppgarr)):
            neghr=predict_heart_rate(neg_rppgarr[i][0].detach().cpu().numpy(),self.Fs,self.min_hr,self.max_hr)

            loss+=np.abs(neghr/poshr1-ratio_array[0][i][0].detach().cpu().numpy())+\
                  np.abs(neghr/poshr2-ratio_array[0][i][0].detach().cpu().numpy())
            count+=2
        loss=loss/count
        return loss

class CFAL(nn.Module):
    def __init__(self, Fs, high_pass=2.5, low_pass=0.4):
        super(CFAL, self).__init__()
        #PSD_MSE
        self.norm_psd = CalculateNormPSD(Fs, high_pass, low_pass)
        self.distance_func = nn.MSELoss()

    def forward(self, pos_rppg1,pos_rppg2,neighbor_rppg1,neighbor_rppg2,neighbor_rppg3):
        posfre1= self.norm_psd(pos_rppg1)
        posfre2= self.norm_psd(pos_rppg2)
        neifre1=self.norm_psd(neighbor_rppg1)
        neifre2=self.norm_psd(neighbor_rppg2)
        neifre3=self.norm_psd(neighbor_rppg3)

        loss = self.distance_func(posfre1, neifre1)+self.distance_func(posfre1, neifre2)+self.distance_func(posfre1, neifre3)\
               +self.distance_func(posfre2, neifre1)+self.distance_func(posfre2, neifre2)+self.distance_func(posfre2, neifre3)
        loss=loss/6
        return loss

class FCL(nn.Module):
    def __init__(self, Fs, high_pass=2.5, low_pass=0.4,tau=0.08):
        super(FCL, self).__init__()
        #PSD_MSE
        self.norm_psd = CalculateNormPSD(Fs, high_pass, low_pass)
        self.distance_func = nn.MSELoss()
        self.tau=tau

    def forward(self, neg_rppgarr,pos_rppg1,pos_rppg2):

        posfre1= self.norm_psd(pos_rppg1)
        posfre2= self.norm_psd(pos_rppg2)
        pos_dis=torch.exp(self.distance_func(posfre1, posfre2)/self.tau)
        neg_dis_total=0
        for i in range(len(neg_rppgarr)):
            negfre=self.norm_psd(neg_rppgarr[i])
            neg_dis = torch.exp(self.distance_func(posfre1, negfre) / self.tau)+torch.exp(self.distance_func(posfre2, negfre) / self.tau)
            neg_dis_total+=neg_dis

        loss = torch.log10(pos_dis/neg_dis_total+1)
        return loss

