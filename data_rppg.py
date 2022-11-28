from torchvision.transforms import Compose, ToTensor, Normalize
from dataset_rppg import *

def transform():
    return Compose([
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
    ])

##LOADER TRAINING
def get_dataset(file_list,num_negative,video_length):
    return DatasetFromFolder(file_list,num_negative,video_length,transform=transform())

