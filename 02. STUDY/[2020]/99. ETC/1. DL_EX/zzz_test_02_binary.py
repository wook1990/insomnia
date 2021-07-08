import warnings
import getopt
import os
import platform
import sys
import datetime
import uuid
import pandas
import numpy
import multiprocessing
import glob
import codecs
import random
import time

import torch
import torchvision
import torchvision.models as models
import torch.nn as nn

from torchvision.models import mobilenet_v2
from torch.utils.data import Dataset, DataLoader

from multiprocessing import *
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C


# 상수 설정
def get_constants(p_name):
    dict_constants = {
                        'epoch_begin': 0,
                        'epoch_end': 1000,
                        'batch_size': 43,
                        'lr': 0.001,
                        'cuda': False,
                        'device': torch.device("cpu"),
                        # 'device':  torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
                        'hidden_size': 128,
                        'output_size': 5,
                        'num_layers': 2,
                        'infer_batch_size': 10,
                    }
    return dict_constants[p_name]


# 모델 구조 임포트
class BinaryDNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(BinaryDNN, self).__init__()
        self.dnn = nn.DNN(input_size, hidden_size, get_constants('num_layers'), batch_first=True)
        self.output_fc = nn.Linear(hidden_size, get_constants('output_size'))
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input, hidden):
        output, _ = self.rnn(input, hidden)
        output = output[:,-1,:]
        # output = output.contiguous().view(-1, hidden_size)
        output = self.output_fc(output)
        output = self.softmax(output)
        return output

    def initHidden(self):
        return torch.zeros(1, self.hidden_size)


# 초기화 함수
def infer(p_recommendNet):
    p_recommendNet.eval()
    loss_max = 0.
    worst_item = 0.
    loss_count = 0
    for i, sample in enumerate(infer_dataloader):
        samples = sample[0]
        label = sample[1]
        predict = p_recommendNet(samples, torch.zeros(get_constants('num_layers'), get_constants('infer_batch_size'), get_constants('hidden_size')).to(get_constants('device')))
    check_labels = label.max(1)[1]
    check_predict = predict.max(1)[1]
    check_eq = torch.eq(check_labels, check_predict).sum().item()
    loss = get_loss_function(predict, label)
    # print("infer loss: ", loss.item(), "wrong count: %d" % (label.shape[0]-check_eq))
    p_recommendNet.train()


# 옵티마이저 설정
def get_optimizer(p_recommendNet):
    adam = torch.optim.Adam(p_recommendNet.parameters(), lr=lr, betas=(0.5, 0.999))
    return adam


# 손실함수 설정
def get_loss_function(p_predict, p_label):
    # bce = torch.nn.BCELoss()
    bce = torch.nn.MSELoss()
    # bce = torch.nn.CrossEntropyLoss()
    # bce = torch.nn.BCEWithLogitsLoss()

    loss = bce(p_predict, p_label)
    return loss_value


# 조기 중단점 설정
def get_early_stopping_score():
    return 0.85


def main(p_args):
    # 모델 선언
    recommendNet = BinaryDNN(54, get_constants('hidden_size'))
    recommendNet = recommendNet.to(get_constants('device'))
    # print(poseNet)

    # 데이터 로딩부분 설정
    train_dataset = PoseDataset(cuda=get_constants('cuda'))
    infer_dataset = PoseDataset(is_train=False, cuda=get_constants('cuda'))
    train_dataloader = DataLoader(train_dataset, batch_size=get_constants('batch_size'), shuffle=True)
    infer_dataloader = DataLoader(infer_dataset, batch_size=get_constants('infer_batch_size'), shuffle=False)

    # 옵티마이저 획득
    optimizer = get_optimizer(recommendNet)

    epoch_begin = get_constants('epoch_begin')
    epoch_end = get_constants('epoch_end')
    # 모델 학습
    for e in range(epoch_begin, epoch_end):
        if e % 10 == 0 and e > 0:
            infer(recommendNet)

        #   if e%20 == 0 and e>0:
        #     torch.save(mobileV2.state_dict(), "pose.torch_%04d" % e)

        for i, sample in enumerate(train_dataloader):
            samples = sample[0]
            label = sample[1]

            optimizer.zero_grad()

            predict = p_recommendNet(samples, torch.zeros(get_constants('num_layers'), get_constants('batch_size'), get_constants('hidden_size')).to(get_constants('device')))

            loss = get_loss_function(predict, label)

            loss.backward()
            optimizer.step()

            check_labels = label.max(1)[1]
            check_predict = predict.max(1)[1]
            check_eq = torch.eq(check_labels, check_predict).sum().item()

            passed = time.time() - start_time
            log_format = "Epoch: [%04d], [%04d/%04d] time: %.4f, loss: %.5f, wrong count: %d, predict1: %.4f"
            print(log_format % (e, i, len(train_dataloader) , passed, loss.item(), label.shape[0]-check_eq, predict[0][0].item()))

    # 모델 저장
    # ????????????????


if __name__ == "__main__":
    warnings.filterwarnings('ignore')  # "error", "ignore", "always", "default", "module" or "once"
    main(sys.argv)




