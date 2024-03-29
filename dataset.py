import glob
import torch
from imageio import imread
import cv2
import numpy as np
import os
from torch.utils.data import Dataset
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import random

def img_normalize(image):
    if len(image.shape)==2:
        channel = (image[:, :, np.newaxis] - 0.485) / 0.229
        image = np.concatenate([channel,channel,channel], axis=2)
    else:
        # print('normalize!')
        image = (image-np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape((1, 1, 3)))\
                /np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape((1, 1, 3))
    return image

class TrainDataset(Dataset):
    def __init__(self, pathes, size):
        self.fwflow_list=[]
        self.img_list=[]
        self.label_list=[]
        self.size=size
        
        for path in pathes:
            
            # DUTS
            if path.split('/')[-1]=='DUTS-TR':
                self.img_list+=sorted(glob.glob(os.path.join(path, "Image", "*.jpg")))
                self.fwflow_list+=sorted(glob.glob(os.path.join(path, "Image", "*.jpg")))
                self.label_list+=sorted(glob.glob(os.path.join(path, "Mask", "*.png")))
            
            # davis16
            elif path.split('/')[-1]=='DAVIS':
                with open('dataset/DAVIS/train_seqs.txt') as f:
                    seqs = f.readlines()
                    seqs = [seq.strip() for seq in seqs]
                # print(seqs)
                for i in seqs:
                    self.img_list+=sorted(glob.glob(os.path.join(path, "JPEGImages/480p", i, "*.jpg")))[:-1]
                    self.label_list+=sorted(glob.glob(os.path.join(path, "Annotations/480p", i, "*.png")))[:-1]
                    self.fwflow_list+=sorted(glob.glob(os.path.join(path, "raft_flow", i, "*.png")))
            else:
                raise Exception("Invalid dataset!")
            
        self.dataset_len = len(self.img_list)
    def __len__(self):
        return len(self.img_list)
    def __getitem__(self, item):
        video = imread(self.img_list[item])
        fw = imread(self.fwflow_list[item])
        label = imread(self.label_list[item])
        label = np.minimum(label, 1)*255
        
        # get flip param
        h_flip = False
        if random.random() > 0.5:
            h_flip = True
        v_flip = False
        if random.random() > 0.5:
            v_flip = True
        
        # joint flip
        if h_flip:
            video = cv2.flip(video, 1) # 水平翻转
            fw = cv2.flip(fw, 1)
            label = cv2.flip(label, 1)
        if v_flip:
            video = cv2.flip(video, 0) # 垂直翻转
            fw = cv2.flip(fw, 0)
            label = cv2.flip(label, 0)
            
        if len(label.shape)==3:
            label=label[:,:,0]
        label=label[:, :, np.newaxis]
        video = img_normalize(video.astype(np.float32)/255.)
        label = label.astype(np.float32)/255.
        fwflow = img_normalize(fw.astype(np.float32)/255.)
  
        video = torch.from_numpy(video).permute(2,0,1)
        label = torch.from_numpy(label).permute(2,0,1)
        fwflow = torch.from_numpy(fwflow).permute(2,0,1)

        if self.size is None:
            return {'video': video,
                    'label': label,
                    'fwflow': fwflow}
        else:
            return {'video': F.interpolate(video.unsqueeze(0), (self.size,self.size), mode='bilinear', align_corners=True).squeeze(0),
                    'label': F.interpolate(label.unsqueeze(0), (self.size,self.size), mode='bilinear', align_corners=True).squeeze(0),
                    'fwflow': F.interpolate(fwflow.unsqueeze(0), (self.size,self.size), mode='bilinear', align_corners=True).squeeze(0)}

class ValDataset(Dataset):
    def __init__(self, pathes, size):
        self.fwflow_list=[]
        self.img_list=[]
        self.label_list=[]
        self.size=size
        
        for path in pathes:
            # davis16
            if path.split('/')[-1]=='DAVIS':
                with open('dataset/DAVIS/val_seqs.txt') as f:
                    seqs = f.readlines()
                    seqs = [seq.strip() for seq in seqs]
                print(seqs)
                for i in seqs:
                    self.img_list+=sorted(glob.glob(os.path.join(path, "JPEGImages/480p", i, "*.jpg")))[:-1]
                    self.label_list+=sorted(glob.glob(os.path.join(path, "Annotations/480p", i, "*.png")))[:-1]
                    self.fwflow_list+=sorted(glob.glob(os.path.join(path, "raft_flow", i, "*.png")))
            else:
                raise Exception("Invalid dataset!")

        self.dataset_len = len(self.img_list)
    def __len__(self):
        return len(self.img_list)
    def __getitem__(self, item):
        video = imread(self.img_list[item])
        fw = imread(self.fwflow_list[item])
        label = imread(self.label_list[item])
        # label = np.minimum(label, 1)*255
            
        if len(label.shape)==3:
            label=label[:,:,0]
        label=label[:, :, np.newaxis]
        video = img_normalize(video.astype(np.float32)/255.)
        label = label.astype(np.float32)/255.
        fwflow = img_normalize(fw.astype(np.float32)/255.)
        # print(video.shape, label.shape, fwflow.shape)
        
        video = torch.from_numpy(video).permute(2,0,1)
        label = torch.from_numpy(label).permute(2,0,1)
        fwflow = torch.from_numpy(fwflow).permute(2,0,1)
        
        if self.size is None:
            return {'video': video,
                    'label': label,
                    'fwflow': fwflow}
        else:
            return {'video': F.interpolate(video.unsqueeze(0), (self.size,self.size), mode='bilinear', align_corners=True).squeeze(0),
                    'label': F.interpolate(label.unsqueeze(0), (self.size,self.size), mode='bilinear', align_corners=True).squeeze(0),
                    'fwflow': F.interpolate(fwflow.unsqueeze(0), (self.size,self.size), mode='bilinear', align_corners=True).squeeze(0)}
