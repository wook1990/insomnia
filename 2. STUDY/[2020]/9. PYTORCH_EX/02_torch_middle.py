import torch
import numpy as np
import pandas as pd


# Linear Regression use torch
# 1. 훈련데이터 셋의 구성
x_train = torch.FloatTensor([[1],[2],[3]])
y_train = torch.FloatTensor([[2],[4],[6]])

# 2. 가설 수립(Hypothesis)
# 선형 회귀는 데이터와 가장 잘맞는 하나의 직선을 찾음
# H(x) = Wx + b (W 가중치 (weight), b를 편향 (Bais))

# 3. 비용함수(cost function)에 대한 이해
# 비용함수(cost function) = 손실함수(loss function) = 오차함수(error function) = 목적함수(objective function)
# 적절한 가중치오아 편향을 찾기위해 최적화된 식을 비용 함수로 정의할 수 있다
# Cost(W,b)를 최소가 되게 만드는 W와 b를 구하면 훈련데이터를 가장 잘나타내는 직선을 구할 수 있음

# 4. optimizer - 경사하강법(Gradient Descent)
# Cost(W,b)의 값을 최소화하는 W,b를 찾는 방법을 옵티마이저(Optimizer) 알고리즘이라 하고,
# 최적화알고리즘이라고도 함
# 옵티마이저 알고리즘을 통해 적절한 W,b를 찾는 과정은 ML의 학습(Training)이라 함
# cost가 최소화가 되는 지점은 접선의 기울기가 0이되는 지점, 또한 미분값이 0이되는 지점
# 경사하강법의 아이디어는 Cost Function을 미분하여 현재 W에서 접선의 기울기를 구하고,
# 접선의 기울기가 낮은 방향으로 W의 값을 변경하는 작업을 반복
# 기울기 = acost(W)/aW (특정숫자 a = learning rate)
# learning rate 는 W의 값을 변경할 때, 얼마만큼의 크기로 변경할지를 결정하는 상수


# 구현
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# 시드값 고정
torch.manual_seed(1)

# 변수 선언
x_train = torch.FloatTensor([[1],[2],[3]])
y_train = torch.FloatTensor([[2],[4],[6]])
print(x_train)
print(x_train.shape)
print(y_train)
print(y_train.shape)

# 가중치와 편향의 초기화
W = torch.zeros(1,requires_grad=True)
b = torch.zeros(1, requires_grad=True)
# optimizer 설정
optimizer = optim.SGD([W,b], lr = 0.01)

np_epochs = 2000 # 원하는 만큼 경사 하강법 반복

for epoch in range(np_epochs +1):
    # H(x) 계산
    hypothesis = x_train * W + b
    # 비용 함수 선언
    cost = torch.mean((hypothesis - y_train)**2)
    # 경사 하강법 구하기
    # gradient를 0으로 초기화
    optimizer.zero_grad()
    # 파이토치는 미분을 통해 얻은 기울기를 이전에 계산된 기울기 값에 누적시키는 특징
    # 기울기를 0으로 초기화 해주어야함
    # 비용함수를 미분하여 gradient 계산
    cost.backward()
    # W와 b를 업데이터
    optimizer.step()
    # 100번 마다 로그 출력
    if epoch % 100 == 0:
        print('Epoch {:4d}/{} W: {:.3f}, b: {:.3f} Cost: {:.6f}'.format(
            epoch, np_epochs, W.item(), b.item(), cost.item()))



# requires_grad = True, backward() 파이토치에서 제공하는 자동미분 기능 수행(Autograd)
# 비용함수를 손실 함수, 오차함수라고도 부르므로 비용이 최소화 되는 방향이라는 표현 대신
# 손실이 최소화 되는 방향 또는 오차를 최소화하는 방향이라고도 설명 가능


# 자동 미분 Autograd
import torch
# require_grad 텐서에 기울기를 저장하는 옵션
W = torch.tensor(2.0, requires_grad=True)

y = W**2
z = 2*y + 5
z.backward()
print( '수식을  w로 미분한 값 : {}'.format(W.grad))

# 다중선형회귀
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# 시드 고정
torch.manual_seed(1)

# H(x) = w1x1 + w2x2 + w3x3 + b
x1_train = torch.tensor([[73],[93],[89],[96],[73]])
x2_train = torch.tensor([[80],[88],[91],[98],[66]])
x3_train = torch.tensor([[75],[93],[90],[100],[70]])
y_train = torch.tensor([[152],[185],[180],[196],[142]])

# 가중치 w 와 편향 b를 선언,
w1 = torch.zeros(1, requires_grad=True)
w2 = torch.zeros(1, requires_grad=True)
w3 = torch.zeros(1, requires_grad=True)
b = torch.zeros(1, requires_grad=True)

# 가설, 비용함수, 옵티마이저 선헝후 경사하강법 1000회 반복
optimizer = optim.SGD([w1,w2,w3,b], lr = 1e-5)

nb_epochs = 10000
for epoch in range(nb_epochs + 1):

    # H(x) 계산
    hypothesis = x1_train * w1 + x2_train * w2 + x3_train * w3 + b

    # cost 계산
    cost = torch.mean((hypothesis - y_train) ** 2)

    # cost로 H(x) 개선
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    # 100번마다 로그 출력
    if epoch % 100 == 0:
        print('Epoch {:4d}/{} W: {:.3f}, b: {:.3f} Cost: {:.6f}'.format(
            epoch, nb_epochs, W.item(), b.item(), cost.item()))


# 벡터와 행렬 연산으로 변경하여 각 행렬의 곱센 연산으로 데이터 선언
# 행렬의 곱셈과정에서 일우어지는 벡터 연산을 벡터의 내적(Dot product)라 함

x_train = torch.FloatTensor([[73,80,75],
                             [93,88,93],
                             [89,91,90],
                             [96,998,100],
                             [73,66,70]])

y_train = torch.FloatTensor([[152],[185],[180],[196],[142]])

print(x_train.shape)
print(y_train.shape)

# 가중치 W와 편향 b 선언
W = torch.zeros((3,1),requires_grad=True)
b = torch.zeros(1, requires_grad=True)

# 행렬의 곱셈연산이므로 matmul함수 사용
hypothesis = x_train.matmul(W) + b
# hypothesis = x_train * W + b(같은 의미?)

optimizer = optim.SGD([W,b], lr = 1e-5)

nb_epochs = 2000
for epoch in range(nb_epochs + 1):
    # H(x) 계산
    # 편향 b 는 브로드캐스팅되어 각 샘플에 더해짐
    hypothesis = x_train.matmul(W) + b

    # cost계산
    cost = torch.mean((hypothesis - y_train) ** 2)

    # cost로 H(x) 계산
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()
    # 100번마다 로그 출력
    if epoch % 5 == 0:
        print('Epoch {:4d}/{} hypothesis: {} Cost: {:.6f}'.format(
            epoch, nb_epochs, hypothesis.squeeze().detach(), cost.item()
        ))


# 04. nn.Modul로 구현하는 선형 회귀
# 파이토치에서는 선형 회귀 모델이 nn.Linear() 함수로
# 평균 제곱오차가 nn.functional.mse_loss() 함수로 구현
# import torch.nn as nn
# model = nn.Linear(input_dim, ouput_dim)
# import torch.nn.functional as F
# cost = F.mse_loss(prediction, y_train)

# 단순 선형 회귀 구현
import torch
import torch.nn as nn
import torch.nn.functional as F

# 데이터
x_train = torch.FloatTensor([[1],[2],[3]])
y_train = torch.FloatTensor([[2],[4],[6]])


# 모델 선언 및 초기화, 단순 선형회귀 input_dim = 1 , ouput_dim = 1
model = nn.Linear(1,1)
print(list(model.parameters()))
# 두 파라미터는 가중치 w와 편향 b를 랜덤 초기화 되어 있는 값이며
# 두 대상 모두 학습의 대상이므로 require_grad=True로 초기화 되어있음

# optimizer 설정, 경사하강법 SGD를 사용하고 learning rate 를 의미하는 lr은 0.01
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# 전체 훈련 데이터에 대한 경사하강법 2000회 반복
nb_epochs = 2000
for epoch in range(nb_epochs + 1):
    #H(x) 선언
    prediction = model(x_train)

    # cost 계산(파이토치 제공 평균 제곱 오차 함수)
    cost = F.mse_loss(prediction, y_train)

    # cost로 H(x) 개선
    # gradient를 0으로 초기화
    optimizer.zero_grad()
    # 비용함수를 미분하여 gradient 계산
    cost.backward()
    # W와 b를 업데이트
    optimizer.step()

    if epoch % 100 == 0:
    # 100번마다 로그 출력
      print('Epoch {:4d}/{} Cost: {:.6f}'.format(
          epoch, nb_epochs, cost.item()
      ))

# 임의의 입력 4선언
new_var = torch.FloatTensor([[4.0]])
# 입력한 값 4에 대해서 예측값 y를 리턴받아서 pred_y에 저장
pred_y = model(new_var)
print("훈련후 입력이 4일때 예측값 : ", pred_y)
print(list(model.parameters()))

# H(x) 식에 입력 x로 부터 예측된 y를 얻는 것을 forward 연산
# 학습 전, prediction = model(x_train)은 x_train으로 부터 예측값을 리턴하므로 forward 연산
# 학습 후, pred_y = model(new_var)는 임의의 값 new_var로 부터 예측값을 리턴하므로 forward 연산
# 학습과정에서 비용 함수를 미분하여 기울기를 구하는 것을 backward 연산
# cost.backward()는 비용 함수로부터 기울기를 구하라는 의미이며 backward 연산


# 다중회귀
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(1)

model = torch.nn.Linear(3,1)
list(model.parameters())

optimizer = optim.SGD(model.parameters(), lr = 1e-5)

nb_epochs = 2000
for epoch in range(nb_epochs + 1):
    # H(x)
    prediction = model(x_train)
    # model(x_train) 은 model.forward(x_train)과 동일

    # cost 계산
    cost = F.mse_loss(prediction, y_train)

    # cosg로 H(x) 최적화
    # gradient 초기화
    optimizer.zero_grad()
    # 비용함수를 미분하여 gradient 계산
    cost.backward()
    # W 와 b를 업데이트
    optimizer.step()

    if epoch % 100 == 0:
        print('Epocck : {:4d}/{} Cost : {:.6f}'.format(epoch, nb_epochs, cost.item()))


# 임의의 값 [73, 80, 75] 선언
new_var = torch.FloatTensor([[73, 80 ,75]])
# 입력한 값에 대한 예측값 y 리턴
pred_y = model(new_var)
print(" 훈련후 입력이 [73,80,75] 일 때 예측값 :",pred_y)
print(list(model.parameters()))



# 미니 배치와 데이터 로드
# 에포크(Epoch)는 전체 훈련 데이터가 학습에 한번 사용된 주기
# 미니 배치 학습을 하게 되면, 분할된 미니 배치사이즈 만큼 데이터를 가져가
# 미니 배치에 대한 비용(cost)를 계산하고, 경사하강법을 수행 후
# 다음 미니배치애 대하여 비용과 경사하강법계산을 마지막 미니 배치
# 까지 반복을 해 전체 데이터에 대한 학습이 끝날경우 1 epoch라고 함

# 미니 배치의 개수는 미니 배치의 크기를 몇으로 하느냐에 다라 달라지고
# 미니 배치의 크기를 배치 크기(Batch size)라 함
# 경사하강법시 전체 데이터를 사용하는 경우 가중치 값이 최적값에 수렴하는 과정이
# 매우 안정적이지만 계산량이 많음
# 미니 배치 경사 하강법은 최적값으로 수렴하는 과정에서 값이 조금
# 방향성을 잃기도 하지만 훈력 속도가 빠름
# 배치의 크기는 보통 2의 제곱수를 사용, CPU와 GPU의 메모리가
# 2의 배수이므로 배치크기가 2인 제곱수일 경우에 데이터 송수신효과를
# 높힐수 있다고 함

# 이터레이션 : 이터레이션은 한번의 epoch 내에서 이루어지는 매개변수인
# 가중치 W와 b의 업데이트 횟수를 의미.


# 파이토치에서는 Dataset과 DataLoader를 제공하여
# 미니 배치 학습, 데이터 셔플, 병렬 처리까지 간단히 수행할 수 있음
# 기본적으로 Dataset을 정의하고, 이를 DataLoader에 전달
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

x_train  =  torch.FloatTensor([[73,  80,  75],
                               [93,  88,  93],
                               [89,  91,  90],
                               [96,  98,  100],
                               [73,  66,  70]])
y_train  =  torch.FloatTensor([[152],  [185],  [180],  [196],  [142]])

dataset = TensorDataset(x_train, y_train)
# Dataloder는 기본적으로 2개의 파라미터를 받음
# 데이터셋, 미니배치크기, 추가사용되는 파라미터로 shuffle이 있으며
# True 옵션을 주면 Epoch마다 데이터셋을 섞어서 데이터가 학습되는 순서를 변경
dataloader = DataLoader(dataset, batch_size = 2, shuffle = True)

model  = nn.Linear(3,1)
optimizer = optim.SGD(model.parameters(), lr = 1e-5)

nb_epochs = 20
for epoch in range(nb_epochs + 1):
    for batch_idx, samples in enumerate(dataloader):
        print(batch_idx)
        print(samples)
        x_train, y_train = samples

        # H(x) 계산
        prediciton = model(x_train)

        # cost 계산
        cost = F.mse_loss(prediciton, y_train)

        # cost로 h(x) 계산
        optimizer.zero_grad()
        cost.backward()
        optimizer.step()

        print('Epoch {:4d}/{} Batch {}/{} Cost: {:.6f}'.format(epoch, nb_epochs,
               batch_idx + 1, len(dataloader), cost.item()))

# 커스텀 데이터셋(Custom Dataset)으로 선형 회귀 구현
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
class CustomDataset(TensorDataset):
    def __init__(self):
        self.x_data = [[73, 80, 75],
                   [93, 88, 93],
                   [89, 91, 90],
                   [96, 98, 100],
                   [73, 66, 70]]
        self.y_data = [[152], [185], [180], [196], [142]]

    # 총 데이터의 개수를 리턴
    def __len__(self):
        return len(self.x_data)

    # 인덱스를 입력받아 그에 맵핑되는 입출력 데이터를 파이토치의 tensor 형태로 리턴
    def __getitem__(self, idx):
        x = torch.FloatTensor(self.x_data[idx])
        y = torch.FloatTensor(self.y_data[idx])
        return x, y


dataset = CustomDataset()
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
model = torch.nn.Linear(3,1)
optimizer = optim.SGD(list(model.parameters()), lr=1e-5)

nb_epochs = 20
for epoch in range(nb_epochs + 1):
    for batch_idx, samples in enumerate(dataloader):
        print(batch_idx)
        print(samples)
        x_train, y_train = samples
        # H(x) 계산
        prediction = model(x_train)
        # cost 계산
        cost = F.mse_loss(prediction, y_train)

        # cost로 h(x) 계산
        optimizer.zero_grad()
        cost.backward()
        optimizer.step()

        print('Epoch {:4d}/{} Batch {}/{} Cost: {:.6f}'.format(
            epoch, nb_epochs, batch_idx + 1, len(dataloader),
            cost.item()
        ))

# 임의의 값 선언
new_var = torch.FloatTensor([[73, 80,75]])
pred_y = model(new_var)
print(pred_y)
print(list(model.parameters()))


