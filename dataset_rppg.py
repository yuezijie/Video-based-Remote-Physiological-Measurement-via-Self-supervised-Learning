import torch.utils.data as data
import torch
import numpy as np
import os
import random
import cv2


def load_img(filepath, scale,video_length):
    list = os.listdir(filepath)
    imglen=len(list)
    begin=random.randrange(0,int(imglen-(video_length*4)))
    whole_video=[]
    count=[0,1,2,3]
    for i in range(begin, begin + (video_length*4)):
        img=cv2.imread(os.path.join(filepath,'face_{0}.png'.format(i)))
        img=cv2.resize(img,(scale, scale))
        whole_video.append(img)
    whole_video=np.array(whole_video)

    whole_video=whole_video.reshape((4,video_length,whole_video.shape[1],whole_video.shape[2],whole_video.shape[3]))
    a = random.choice(count)
    input=whole_video[a]
    count.remove(a)
    neighbor1=whole_video[count[0]]
    neighbor2=whole_video[count[1]]
    neighbor3=whole_video[count[2]]
    return input, neighbor1,neighbor2,neighbor3


def augment(img_in):
    info_aug = {'flip_h': False, 'flip_v': False, 'rot0': False, 'rot90': False,'rot180': False,'rot270': False}
    randomaug=random.sample(['flip_h', 'flip_v','rot0','rot90','rot180','rot270'],1)
    info_aug[randomaug[0]]=True

    if info_aug['flip_v']:
        img_in = [cv2.flip(j,0) for j in img_in]

    if info_aug['flip_h']:
        img_in = [cv2.flip(j,1) for j in img_in]

    if info_aug['rot90']:
        img_in = [cv2.rotate(j,cv2.ROTATE_90_CLOCKWISE) for j in img_in]
    if info_aug['rot270']:
        img_in = [cv2.rotate(j,cv2.ROTATE_90_COUNTERCLOCKWISE) for j in img_in]
    if info_aug['rot180']:
        img_in = [cv2.rotate(j,cv2.ROTATE_180) for j in img_in]

    return img_in

def frequency_ratio():
  ratio_interval1 = np.random.uniform(0.3,0.8)
  ratio_interval2 = np.random.uniform(1.2,1.7)
  random_ratio = np.random.choice([ratio_interval1, ratio_interval2])
  return random_ratio


class DatasetFromFolder(data.Dataset):
    def __init__(self, file_list,num_negative,video_length,transform=None,scale=64):
        super(DatasetFromFolder, self).__init__()
        self.video_list =[line.rstrip() for line in open(file_list)]
        self.transform = transform
        self.scale=scale
        self.num_negative=num_negative
        self.video_length=video_length


    def __getitem__(self, index):
        input, neighbor1,neighbor2,neighbor3= load_img(self.video_list[index],self.scale,self.video_length)

        positive1= augment(input)
        positive2= augment(input)

        if self.transform:
            input = torch.tensor([self.transform(j).detach().numpy() for j in input])
            positive1 = torch.tensor([self.transform(j).detach().numpy() for j in positive1])
            positive2 = torch.tensor([self.transform(j).detach().numpy() for j in positive2])
            neighbor1 = torch.tensor([self.transform(j).detach().numpy() for j in neighbor1])
            neighbor2 = torch.tensor([self.transform(j).detach().numpy() for j in neighbor2])
            neighbor3 = torch.tensor([self.transform(j).detach().numpy() for j in neighbor3])
        ratio_array=[]
        for i in range(self.num_negative):
            ratio=np.repeat(frequency_ratio(),self.video_length)
            ratio_array.append(ratio)
        ratio_array=torch.tensor(np.array(ratio_array))

        return input, positive1, positive2,neighbor1,neighbor2,neighbor3,ratio_array

    def __len__(self):
        return len(self.video_list)
