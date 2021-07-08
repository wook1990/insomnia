# 이진분류의 대표적인 알고리즘 Logistic Regression
# 시그모이드 함수(Sigmoid Function)
# H(x) = sigmoid(Wx+b) = 1/(1+e^(-(Wx+b) = σ(Wx+b)
# 선형회귀에서는 W가 직선의 기울기, b가 y절편
# 시그모이드 함수의 W,b는 어떠한 영향을 줄까?

import numpy as np
import matplotlib.pyplot as plt

# sigmoid function 정의
def sigmoid(x):
    return 1/(1+np.exp(-x))

x = np.arange(-5.0, 5.0, 1.0)
y = sigmoid(x)

plt.plot(x,y, 'g')
plt.plot([0,0],[1.0,0.0], ':')
plt.title('Sigimoid Function')
plt.show()

# 시그모이드 함수는 출력값을 0과 1사이로 변환
# x가 0일때 0.5의 값을 가지고 x가 매우커지면 1에수렴, 매우작아지면 0에 수렴

# W값의 변화에 따른 경사도의 변화
x = np.arange(-5.0, 5.0, 1.0)
y1 = sigmoid(0.5*x)
y2 = sigmoid(x)
y3 = sigmoid(2*x)
plt.plot(x, y1, 'r', linestyle = '--')
plt.plot(x, y2, 'g')
plt.plot(x, y3, 'b', linestyle = '--')
plt.plot([0,0],[1.0,0.0], ':')
plt.title('Sigimoid Function')

# W의 값에 따라 그래프의 경사도가 변함
# 선형회귀의 가중치 W는 직선의 기울기를 의미했지만
# 시그모이드 함수에서는 그래프의 경사도를 결정
# W의 값이 커지면 경사가 커지고 W의 값이 작아지면 경사가 작아짐

# b값의 변화에 따른 좌,우 이동
x = np.arange(-5.0, 5.0, 1.0)
y1 = sigmoid(x+0.5)
y2 = sigmoid(x+1)
y3 = sigmoid(x+1.5)

plt.plot(x, y1, 'r', linestyle = '--')
plt.plot(x, y2, 'g')
plt.plot(x, y3, 'b', linestyle = '--')
plt.plot([0,0],[1.0,0.0], ':')
plt.title('Sigimoid Function')

# 시그모이드 함수를 이용한 분류
# 시그모이드 함수의 출력값은 0과 1사이의 값을 가지는데 이특성을 이용하여 분류 작업에 활용
# 확률값으로 계산하여 판단 기준 0.5가 일반적

# 비용함수(cost function)
# 로지스틱 회귀의 가설 : H(x) = sigmoid(Wx+b)
# W,b를 찾을 수 있는 cost function 정의
# 선형회귀의 평균제곱오차를 그대로 사용할 시
# 비볼록(non-convex)형태의 그래프가 출력되어, 경사가 최소가되는 구간이
# 실제 오차가 완전히 최소가 되는 부분이 아닐수 있음
# 전체 함수에 걸쳐 최소값인 글로벌 미니멈(Global Minimum)이 아닌
# 특정 구역에서의 최소값인 로컬 미니멈(Local Minimum)에 도달할 경우
# cost가 최소가 되는 가중치 W를 찾는 비용함수 목적에 적합하지 않음
# if y=1 -> cost(H(x),y) = -log(H(x))
# if y=0 -> cost(H(x),y) = -log(1-H(x))
# cost(W)=−1n∑i=1n[y(i)logH(x(i))+(1−y(i))log(1−H(x(i)))]


# pytorch로 구현
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)

x_data = [[1,2],[2,3],[3,1],[4,3],[5,3],[6,2]]
y_data = [[0],[0],[0],[1],[1],[1]]
x_train = torch.FloatTensor(x_data)
y_train = torch.FloatTensor(y_data)
print(x_train.shape)
print(y_train.shape)

# W와 b선언
W = torch.zeros((2,1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)

# H(x)
hypothesis = 1/(1 + torch.exp(-(x_train.matmul(W)+b)))
# hypothesis = torch.sigmoid(x_train.matmul(W)+b)
# 두방법을 사용하여 정의 가능
print(hypothesis)
-(y_train[0] * torch.log(hypothesis[0]) +   (1 - y_train[0]) * torch.log(1 - hypothesis[0]))
# 전체에 대한 오차
losses = -(y_train * torch.log(hypothesis) + (1 - y_train) * torch.log(1 - hypothesis))
print(losses)
# 오차에 대한 평균
cost = losses.mean()
print(cost)

# 직접 비용 함수를 구현하였지만 파이토치에서는 로지스틱회귀분석의
# 비용함수를 이미 구현 제공
# torch.nn.functional 의 binary_cross_entropy(예측값, 실제값)
cost = F.binary_cross_entropy(hypothesis, y_train)


# 최종 cost를 줄여가는 코드
x_data = [[1,2],[2,3],[3,1],[4,3],[5,3],[6,2]]
y_data = [[0],[0],[0],[1],[1],[1]]
x_train = torch.FloatTensor(x_data)
y_train = torch.FloatTensor(y_data)

# W와 b선언
W = torch.zeros((2,1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)

# optimizer 선언
optimizer = optim.SGD([W,b], lr=1)

nb_epochs = 1000
for epoch in range(nb_epochs + 1):

    # H(x) 선언
    hypothesis = torch.sigmoid(x_train.matmul(W)+b)
    # cost 선언
    cost = -(y_train * torch.log(hypothesis) +
             (1 - y_train) * torch.log(1 - hypothesis)).mean()

    # cost로 H(x) 개선
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    # 100번마다 로그 출력
    if epoch % 100 == 0:
        print('Epoch {:4d}/{} Cost: {:.6f}'.format(
            epoch, nb_epochs, cost.item()
        ))

hypothesis = torch.sigmoid(x_train.matmul(W) + b)
print(hypothesis)
prediction = hypothesis >= torch.FloatTensor([0.5])
print(prediction)
print(W)
print(b)

# nn 모듈로 구현하는 로지스틱 회귀
# 1. nn.Linear와 nn.Sigmoid로 로지스틱 회귀 구현
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


#동일 값 학습을 위한 시드 고정
torch.manual_seed(1)

x_data = [[1, 2], [2, 3], [3, 1], [4, 3], [5, 3], [6, 2]]
y_data = [[0], [0], [0], [1], [1], [1]]
x_train = torch.FloatTensor(x_data)
y_train = torch.FloatTensor(y_data)

# nn.Sequential()은 nn.Module 층을 차례로 쌓을수 있도록 해준다
# nn.Sequintail()은 Wx + b 와 같은 수식과 시그모이드 함수 등과 같은 여러함수를 연결

model = nn.Sequential(
    nn.Linear(2,1), # input_dim  = 2 ,output_dim = 1
    nn.Sigmoid() # 출력은 시그모이드 함수를 거친다
)

model(x_train)

# W와 b는 임의의 값을 가지므로 예측의 의미 없음

# optimizer 설정
optimizer = optim.SGD(model.parameters(), lr = 1)

nb_epochs = 1000
for epoch in range(nb_epochs+1):
    # H(x) 계산
    hypothesis = model(x_train)

    # cost 계산
    cost = F.binary_cross_entropy(hypothesis, y_train)

    # cost로 H(x)계산
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    # 20번 마다 로그 출력
    if epoch % 10 == 0:
        prediction = hypothesis >= torch.FloatTensor([0.5]) # 예측값이 0.5를 넘으면 True로 간주
        correct_prediction = prediction.float() == y_train # 실제값과 일치하는 경우만 True로 간주
        accuracy = correct_prediction.sum().item() / len(correct_prediction) # 정확도를 계산
        print('Epoch {:4d}/{} Cost: {:.6f} Accuracy {:2.2f}%'.format( # 각 에포크마다 정확도를 출력
            epoch, nb_epochs, cost.item(), accuracy * 100,
        ))

model(x_train)
print(list(model.parameters()))
