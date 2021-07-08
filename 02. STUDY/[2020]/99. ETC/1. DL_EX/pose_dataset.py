import torch
import torchvision
import os
import glob
import numpy as np
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import random

TRAIN_COUNT = 900
TEST_COUNT = 180

def makeOneHotTensor(index, tensor_size):
  tensor = torch.zeros(1, tensor_size)
  tensor[0][index] = 1
  return tensor

def makeOneHotArray(index, array_size):
  one_array = [0] * array_size
  one_array[index] = 1
  return [one_array]

class PoseDataset(Dataset):


    def __init__(self, is_train=True, cuda=True, is_augment=True):

      self.is_train = is_train
      self.cuda = cuda
      self.is_augment = is_augment

      files = [
        "./pose_data/clap.dat",
        "./pose_data/left_wave.dat",
        "./pose_data/nod.dat",
        "./pose_data/right_wave.dat",
        "./pose_data/shake.dat",
      ]

      train_data = []
      train_label = []
      test_data = []
      test_label = []
      
      for (i, file) in enumerate(files):
        with open(file) as f:
          content = f.readlines()

        for (j, oneLine) in enumerate(content):
          content[j] = oneLine.split(",")

        for j in range(int(TRAIN_COUNT/10)-4):
          train_data.append(content[j*10:j*10+50])
          train_label = train_label + makeOneHotArray(i, 5)

        for j in range(int(TEST_COUNT/10)-4):
          test_data.append(content[TRAIN_COUNT+j*10:TRAIN_COUNT+j*10+50])
          test_label = test_label + makeOneHotArray(i, 5)

      self.train_data = np.array(train_data, dtype='f')
      self.train_label = np.array(train_label, dtype='f')
      self.test_data = np.array(test_data, dtype='f')
      self.test_label = np.array(test_label, dtype='f')
      print(self.train_data.shape)
      print(self.train_label.shape)
      print(self.test_data.shape)
      print(self.test_label.shape)

    def __len__(self):

      if self.is_train:
        return self.train_data.shape[0]

      return self.test_data.shape[0]

    def __getitem__(self, idx):

      if self.is_train:
        data = self.train_data[idx]
        label = self.train_label[idx]
      else:
        data = self.test_data[idx]
        label = self.test_label[idx]

      if self.is_augment:
#         print("augmented")
        random_factor = np.random.normal(1, 0.007, data.shape)
        data = data * random_factor
          
      if self.cuda:
        data = torch.from_numpy(data).type('torch.cuda.FloatTensor')
        label = torch.tensor(label).type('torch.cuda.FloatTensor')
      else:
        data = torch.from_numpy(data).type('torch.FloatTensor')
        label = torch.tensor(label).type('torch.FloatTensor')

#       print(sample.max())
#       print(sample.min())
#       print(sample.mean())
#       print(sample.shape)
#       print(label.shape)
      return data, label


