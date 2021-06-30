# Class를 활용한 모델 객체 생성 하여
# torch Model 활용
# python class 구성과 init, self, super 등의
# oop (객체지향프로그래밍) 개념 정리하기
import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1,1)

    def forward(self,x):
        return self.linear(x)



class MultivariateLinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3,1) #

    def forward(self,x):
        return self.linear(x)


model = MultivariateLinearRegressionModel()