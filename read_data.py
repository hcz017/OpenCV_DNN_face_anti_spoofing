from PIL import Image
import numpy as np
import os
from torch.utils.data import Dataset
import math
import cv2
import torchvision
import torch
# 训练集路径 
depth_dir_train_file = ["datasets/real_data_train_list.txt","datasets/fake_data_train_list.txt"]
label_dir_train_file = ["datasets/real_labels_train_list.txt","datasets/fake_labels_train_list.txt"]
 
# 验证集路径
depth_dir_val_file = ["datasets/real_data_val_list.txt","datasets/fake_data_val_list.txt"]
label_dir_val_file = ["datasets/real_labels_val_list.txt","datasets/fake_labels_val_list.txt"]
# 测试集路径
depth_dir_test_file = ["datasets/real_data_test_list.txt","datasets/fake_data_test_list.txt"]
label_dir_test_file = ["datasets/real_labels_test_list.txt","datasets/fake_labels_test_list.txt"]

class Data(Dataset):
    def __init__(self, transform=None, phase_train=True, data_dir=None,phase_test=False):
        self.phase_train = phase_train
        self.phase_test = phase_test
        self.transform = transform
        # 处理训练集
        self.depth_dir_train = []
        self.label_dir_train = []
        for file_list in depth_dir_train_file:
            print('train file list: {}'.format(file_list))
            with open(file_list, 'r') as f:
                self.depth_dir_train += f.read().splitlines()
        for file_list in label_dir_train_file:
            print('train label list: {}'.format(file_list))
            with open(file_list, 'r') as f:
                self.label_dir_train += f.read().splitlines()
        print('{} train files, {} train labels'.format(len(self.depth_dir_train), len(self.label_dir_train)))
        # 处理验证集
        self.depth_dir_val = []
        self.label_dir_val = []
        for file_list in depth_dir_val_file:
            with open(file_list, 'r') as f:
                self.depth_dir_val += f.read().splitlines()
        for file_list in label_dir_val_file:
            with open(file_list, 'r') as f:
                self.label_dir_val += f.read().splitlines()
        print('{} val files, {} val labels'.format(len(self.depth_dir_val), len(self.label_dir_val)))
        # 处理测试集
        self.depth_dir_test = []
        self.label_dir_test = []
        if self.phase_test:
            for file_list in depth_dir_test_file:
                with open(file_list, 'r') as f:
                    self.depth_dir_test += f.read().splitlines()
            for file_list in label_dir_test_file:
                with open(file_list, 'r') as f:
                    self.label_dir_test += f.read().splitlines()
        print('{} test files, {} test labels'.format(len(self.depth_dir_test), len(self.label_dir_test)))
    # 定义文件内样本个数函数len（）
    def __len__(self):
        if self.phase_train:
            return len(self.depth_dir_train)
        else:
            if self.phase_test:
                return len(self.depth_dir_test)
            else:
                return len(self.depth_dir_val)
     # 定义获取item函数getitem（）
    def __getitem__(self, idx):
        if self.phase_train:
            depth_dir = self.depth_dir_train
            label_dir = self.label_dir_train
            label = int(label_dir[idx])
            label = np.array(label)
        else:
            if self.phase_test:
                depth_dir = self.depth_dir_test
                label_dir = self.label_dir_test
                label = int(label_dir[idx])
                label = np.array(label)
            else:
                depth_dir = self.depth_dir_val
                label_dir = self.label_dir_val
                label = int(label_dir[idx])
                label = np.array(label)
        depth = Image.open(depth_dir[idx])
        depth = depth.convert('RGB')
        # 获取数据集中的depth图像和对应label
        if self.transform:
            depth = self.transform(depth)
        if self.phase_train:
            return depth,label
        else:
            return depth,label,depth_dir[idx]
