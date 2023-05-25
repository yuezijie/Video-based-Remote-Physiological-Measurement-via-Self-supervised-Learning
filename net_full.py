import os
import torch.nn as nn
import torch.optim as optim
from base_networks import *
import torch.nn.functional as F
import torch
import numpy as np
from net_LFA import Net as LFA
from net_REA import REA


class Mynet(nn.Module):
    def __init__(self, base_filter,num_negative,video_length,num_expert):
        super(Mynet, self).__init__()

        self.LFA=LFA(base_filter,video_length)
        self.REA=REA(base_filter,video_length,num_expert)
        self.num_negative=num_negative


    def forward(self, input, positive1, positive2,neighbor1,neighbor2,neighbor3,ratio_array):
        neg_rppgarr=[]
        negative_arr=[]
        for i in range(self.num_negative):
            negative=self.LFA(input,ratio_array[:,i])
            #BCTHW

            neg_rppg=self.REA(negative).squeeze(1)
            neg_rppgarr.append(neg_rppg)
            negative_arr.append(negative)

        positive1=positive1.transpose(1,2)
        positive2=positive2.transpose(1,2)
        neighbor1=neighbor1.transpose(1,2)
        neighbor2=neighbor2.transpose(1,2)
        neighbor3=neighbor3.transpose(1,2)

        pos_rppg1=self.REA(positive1).squeeze(1)
        pos_rppg2=self.REA(positive2).squeeze(1)
        neighbor_rppg1=self.REA(neighbor1).squeeze(1)
        neighbor_rppg2=self.REA(neighbor2).squeeze(1)
        neighbor_rppg3=self.REA(neighbor3).squeeze(1)

        negative_arr=[negative.transpose(1, 2) for negative in negative_arr]

        return neg_rppgarr,pos_rppg1,pos_rppg2,neighbor_rppg1,neighbor_rppg2,neighbor_rppg3,negative_arr
