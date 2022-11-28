import argparse
from math import log10
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import models
import time
from data_rppg import *
import cv2
import math
import numpy as np
from net_full import Mynet
from loss import FRCL,CFAL,FCL

# Training settings
parser = argparse.ArgumentParser(description='PyTorch Super Res Example')
parser.add_argument('--batchSize', type=int, default=1, help='training batch size')
parser.add_argument('--start_epoch', type=int, default=0, help='Starting epoch for continuing training')
parser.add_argument('--nEpochs', type=int, default=100, help='number of epochs to train for')

parser.add_argument('--lr', type=float, default=1e-5, help='Learning Rate. Default=0.01')
parser.add_argument('--gpu_mode', type=bool, default=True)
parser.add_argument('--threads', type=int, default=0, help='number of threads for data loader to use')
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
parser.add_argument('--num_negative', type=int, default=4, help='number of negative samples')
parser.add_argument('--video_length', type=int, default=150, help='video length')
parser.add_argument('--num_expert', type=int, default=4, help='number of experts')
parser.add_argument('--gpus', default=1, type=int, help='number of gpu')
parser.add_argument('--file_list', type=str, default='trainlist.txt')
parser.add_argument('--pretrained_path', type=str, default='premodel.pth')
parser.add_argument('--save_folder', default='weights/', help='Location to save checkpoint models')

opt = parser.parse_args()
gpus_list = range(opt.gpus)
cudnn.benchmark = True
print(opt)


def train(epoch):
    epoch_loss = 0
    model.train()
    for iteration, batch in enumerate(training_data_loader, 1):
        input, positive1, positive2,neighbor1,neighbor2,neighbor3,ratio_array= batch[0], batch[1], batch[2], batch[3], \
                                                                         batch[4], batch[5],batch[6]

        if cuda:
            input = Variable(input).cuda(gpus_list[0]).float()
            positive1 = Variable(positive1).cuda(gpus_list[0]).float()
            positive2 = Variable(positive2).cuda(gpus_list[0]).float()
            neighbor1 = Variable(neighbor1).cuda(gpus_list[0]).float()

            neighbor2 = Variable(neighbor2).cuda(gpus_list[0]).float()
            neighbor3 = Variable(neighbor3).cuda(gpus_list[0]).float()
            ratio_array = Variable(ratio_array).cuda(gpus_list[0]).float()


        optimizer.zero_grad()
        t0 = time.time()
        neg_rppgarr,pos_rppg1,pos_rppg2,neighbor_rppg1,neighbor_rppg2,neighbor_rppg3,negative_arr \
            = model(input, positive1, positive2,neighbor1,neighbor2,neighbor3,ratio_array)

        l_rec = mse_loss(input, negative_arr)
        l_frc=criterion2(neg_rppgarr,pos_rppg1,pos_rppg2,ratio_array)
        l_cfa=criterion3(pos_rppg1,pos_rppg2,neighbor_rppg1,neighbor_rppg2,neighbor_rppg3)
        l_fc=criterion4(neg_rppgarr,pos_rppg1,pos_rppg2)

        loss = l_rec + l_frc+l_cfa+l_fc

        t1 = time.time()

        epoch_loss += loss.data
        loss.backward()
        optimizer.step()
        print(
            "===> Epoch[{}]({}/{}): Loss: {:.4f} || Timer: {:.4f} sec.".format(
                epoch, iteration, len(training_data_loader), loss.data, (t1 - t0)))

    print("===> Epoch {} Complete: Avg. Loss: {:.4f}".format(epoch, epoch_loss / len(training_data_loader)))



def mse_loss(input, negative_arr):
    l_mse = 0
    for i in range(len(negative_arr)):
        l_mse_negative = criterion(input, negative_arr[i])
        l_mse += l_mse_negative
    return l_mse

def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    # print(net)
    print('Total number of parameters: %d' % num_params)


def checkpoint(epoch):
    model_out_path = opt.save_folder + "rppg_model_epoch_{}.pth".format(epoch)
    torch.save(model.state_dict(), model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))


cuda = opt.gpu_mode
if cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")

torch.manual_seed(opt.seed)
if cuda:
    torch.cuda.manual_seed(opt.seed)

print('===> Loading datasets')
train_set = get_dataset(opt.file_list,opt.num_negative,opt.video_length)
print('total_epoch',opt.nEpochs)
training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=True,
                                  drop_last=True)

print('===> Building model ')
model = Mynet(base_filter=64,num_negative=opt.num_negative,video_length=opt.video_length,num_expert=opt.num_expert)

model = torch.nn.DataParallel(model, device_ids=gpus_list)

criterion = nn.MSELoss()

criterion2=FRCL(Fs=30,min_hr = 40, max_hr = 180)
criterion3=CFAL(Fs=30, high_pass=2.5, low_pass=0.4)
criterion4=FCL(Fs=30, high_pass=2.5, low_pass=0.4)


print('---------- Networks architecture -------------')
print_network(model)
print('----------------------------------------------')


if cuda:
    model = model.cuda(gpus_list[0])
    criterion = criterion.cuda(gpus_list[0])
    criterion2= criterion2.cuda(gpus_list[0])
    criterion3= criterion3.cuda(gpus_list[0])
    criterion4= criterion4.cuda(gpus_list[0])


optimizer = optim.Adamax(model.parameters(), lr=opt.lr, betas=(0.9, 0.999), eps=1e-8)

for epoch in range(opt.start_epoch, opt.nEpochs):
    train(epoch)

    # learning rate is decayed by a factor of 2 every half of total epochs
    if (epoch + 1) % (opt.nEpochs/ 2) == 0:
        for param_group in optimizer.param_groups:
            param_group['lr'] /= 2.0
        print('Learning rate decay: lr={}'.format(optimizer.param_groups[0]['lr']))

checkpoint(total_epochs)