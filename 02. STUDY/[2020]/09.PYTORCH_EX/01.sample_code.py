import torch
import numpy as np
import pandas as pd


# pytorch version(1.5.1+cpu, Not cuda)
torch.__version__

# tensor type
# Numpy의 배열(ndarray)와 유사
# 자료형을 만드는 방법
# 1. 리스트나 numpy 배열을 텐서로 변환
# 2. 0 도는 1등의 특정한 값을 가진 텐서를 생성
# 3. 랜덤한 값을 가지는 텐서 생성

# 1. 배열과 리스트를 텐서 자료형으로 변환
# torch.tensor(), torch.as_tensor(), torch.from_numpy() 명령어 사용
# torch.tensor() : 값 복사(value copy)를 사용하여 새로운 텐서 자료형 인스턴스 생성
# torch.as_tensro() : 리스트나 ndarray 객체를 받아, 값 참조(reference)를 사용하여 텐서 자료형 뷰 생성
# torch.from_numpy() : ndarray 객체를 받아, 값 참조(reference)를 사용하여 텐서 자료형 뷰 생성

li = np.array([[1,2],[3,4]])
li_tensor = torch.tensor(li)
li_as_tensor = torch.as_tensor(li)

print(li)
print(li_tensor)
print(li_as_tensor)

# NumPy 배열ㅇ을 텐서로 ㅏ꿀 때대는 torch.from_numpy() 사용

arr = np.array([[1,2],[3,4]])
arr_tensor = torch.tensor(arr)
arr_as_tensor = torch.as_tensor(arr)
arr_from_numpy = torch.from_numpy(arr)

print(arr_tensor, arr_tensor.dtype)
print(arr_as_tensor, arr_as_tensor.dtype)
print(arr_from_numpy, arr_from_numpy.dtype)


# 반대로 tensor를 Numpy 배열로 바꿀 때에는 torch.numpy()
print(arr_tensor.numpy())
print(arr_as_tensor.numpy())
print(arr_from_numpy.numpy())

# tensor는 ndarray의 값을 참조하므로 ndarray 객체의 값을 바꾸면 텐서 자료형의 값도 바뀌고
# 반대로 tensor 자료형에서 원소의 값을 바꾸면 원래 ndarray 객체의 값도 바뀜

arr_as_tensor[0,0]= 1000
arr
# 같은 객체를 참조
arr_from_numpy

arr[0,1] = 2000
arr_as_tensor


# random한 값을 가지는 tensor 생성
# torch.rand() : 0 과 1 사이의 숫자를 균등하게 생성
# torch.rand_like() : 사이즈를 튜플로 입력하지 않고 기존의 텐서로 정의
# torch.randn() : 평균이 0 이고 표준편차가 1인 가우시안 정규분포를 이용해 생성
# torch.randn_like() : 사이즈를 튜플로 입력하지 않고 기존의 텐서로 정의
# torch.randint() : 주어진 범위 내의 정수를 균등하게 생성, 자료형은 torch.float32
# torch.randint_like() : 사이즈를 튜플로 입력하지 않고 기존의 텐서로 정의
# torch.randperm() : 주어진 범위 내의 정수를 랜덤하게 생성
# 랜덤 생성에 사용되는 시드(seed)는 torch.manual_seed() 명령으로 설정

torch.manual_seed(0)
a = torch.rand(5)
b = torch.randn(5)
c = torch.randint(10, size=(5,))
d = torch.randperm(5)
print(a)
print(b)
print(c)
print(d)

# 특정한 값으로 초기화를 하지 않는 행렬을 만들때
# torch.empty() 사용
# 랜덤한 밸류를 가지는 텐서 생성
torch.empty(3,4)

# 특정한 값의 텐서 생성하기
# torch.arrange() : 주어진 범위내의 정수를 순서대로 생성
# torch.ones() : 주어진 사이즈의 1로 이루어진 텐서 생성
# torch.zeros() : 주어진 사이즈의 0 으로 이루어진 텐서 생성
# torch.ones_like() : 사이즈를 튜플로 입력하지 않고 기존의 텐서로 정의
# torch.zeors_like() : 사이즈를 튜플로 입력하지 않고 기존의 텐서로 정의
# torch.linspace() : 시작점과 끝점을 주어진 갯수만큼 균등하게 나눈 간격점을 행벡터로 출력
# torch.logspace() : 시작점과 끝점을 주어진 갯수만큼 로그간격으로 나눈 간격점을 행벡터로 출력


torch.arange(1,10)
torch.ones((2,5)) # 사이즈가 선언된 공동분산행렬
torch.zeros((3,5)) # 사이즈가 선언된 영행렬
torch.linspace(0,10,5) # 0에서 10사이를 5등분한 텐서 생성


arr_tensor.dtype
arr_tensor.type(dtype=torch.int32)

# tensor의 형상을 변환(reshape)하려면 .view() 함수 사용
# 차원을 늘리거나 줄일때에도 사용
t1 = torch.ones(4,3)
t2 = t1.view(3,4)
t3 = t1.view(12)
print(t1)
print(t2)


print(t3)
# 차원의 증가 view()사용 하기도하며, sqeeze(), unsqeeze()를 사용
# squeeze()는 차원의 원소가 1인 차원을 제거
# unsqeeze()함수는 인수로 받는 위치에 새로운 차원 삽입
t1.view(1,3,4)

t4 = torch.rand(1,3,3,)
t4.shape
# 원소가 1인 차원을 제거
# 1 * 3 * 3 의 구조인 텐서의 1인 차원 축소
t4.squeeze()
t5 = torch.rand(3,3)
t5.shape
torch.Size([3,3])
t5.unsqueeze(0).shape
torch.Size([1,3,3])
t5.unsqueeze(1).shape



# 복수의 텐서를 결합
# torch.cat() 함수 사용
# 위아래 병합 행렬의 원리 이해
a = torch.ones(2,3)
b = torch.zeros(3,3)
a.shape
b.shape
c = torch.cat([a,b], dim = 0)
c.shape


# 텐서 분할
# torch.chunk(), torch.split() 사용
d = torch.rand(3,6)
# 1차원의 3덩어리로 (3,2) * 3 개 텐서
c1, c2, c3 = torch.chunk(d, 3, dim=1)
print(d)
print(c1)
print(c2)
print(c3)

# 3개로 분할 (3,3) 2개
d1, d2 = torch.split(d, 3, dim = 1)
print(d1)
print(d2)


# 텐서연산
# 사칙연산기호를 사용하거나 torch의 함수를 사용
# torch.add, torch.sub, torch.mul, torch.div, torch.mm(내적)

x = torch.from_numpy(np.array([[1,2],[3,4]]))
y = torch.from_numpy(np.array([[5,6],[7,8]]))
# 덧셈
print(x+y)
print(torch.add(x,y))
print(x.add(y))
# 곱셈
print( x * y)
print(torch.mul(x,y))
print(x.add(y))

# 내적
print(torch.mm(x,y))

# 인플레이스연산
# 명령어뒤에 _를 붙이면 자기 자신의 값을 바꾸는 inplace 명령이 됨
# 연산결과를 반환하면서 동시에 자기 자신의 데이터를 수정
x = torch.arange(0,5)
z = torch.arange(1,6)

print(x)
print(x.add_(z))
print(x)

# 1개의 원소를 가진 tensor를 python의 scalar로 만들때 .item()함수 사용
scl = torch.tensor(1)
print(scl.item())