import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
import os
import warnings
warnings.filterwarnings(action='ignore')
import datetime
# 데이터 입력
df = pd.read_csv("E:\Dacon\study\LSTM\equity_output_005490_H.csv", index_col=False, low_memory = False)
df.eod_date = pd.to_datetime(df.eod_date, format='%Y%m%d')
df = df.set_index("eod_date")

# 트레인, 검증 데이터 분리
from sklearn.model_selection import train_test_split
# 데이터, 레이블 분리
df_X = df.iloc[:,0:99]  # feature
df_Y = df.iloc[:,100:]  # label

X_train, X_test, Y_train, Y_test = train_test_split(df_X, df_Y, test_size = 0.3, shuffle=False)

# LSTM 모델(기본)
# ndarray 형식 변환
X_train = X_train.values
X_test = X_test.values
Y_train = Y_train.values
Y_test = Y_test.values

# timestep에 관련된 시퀀스의 정의가 없음#
print(X_train.shape, Y_train.shape) # timestep = 99, size = 917
print(X_test.shape, Y_test.shape) # timestpe = 99, size = 393
#  Keras의 RNN 계열은 3차원 데이터형식필요 각 차원은( size, timestep, feature) 순으로 reshape 필요
X_train_t = X_train.reshape(X_train.shape[0],99,1)
X_test_t = X_test.reshape(X_test.shape[0],99,1)
print(X_train_t.shape, Y_train.shape)
print(X_test_t.shape, Y_test.shape)
# LSTM 모델 만들기

# 기초 하이퍼파라미터

model = tf.keras.Sequential()
model.add(keras.layers.LSTM(
    units=1,
    input_shape=[X_train_t.shape[1], X_train_t.shape[2]],
    activation = 'relu',
    return_sequences=False
    )
)
model.add(keras.layers.Dense(units =1))

model.compile(
    loss = 'mean_squared_error',
    optimizer='adam'
)
early_stop = keras.callbacks.EarlyStopping(monitor='var_loss', patience=5)
model_path = "E:\\Dacon\\study\\LSTM\\"
filename = os.path.join(model_path, 'test_checkpoint.h5')
checkpoint = keras.callbacks.ModelCheckpoint(filename, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
history = model.fit(X_train_t, Y_train,
                    epochs=200,
                    batch_size = 16,
                    validation_split= 0.1 ,
                    verbose= 1,
                    shuffle=False,
                    callbacks = [early_stop, checkpoint])

model.load_weights(filename)
pred = model.predict(X_test_t)
Y_test
# Standardization
def data_standardization(x):
    x_np = np.asarray(x)
    return (x_np - x_np.mean()) / x_np.std()


# 너무 작거나 너무 큰 값이 학습을 방해하는 것을 방지하고자 정규화한다
# x가 양수라는 가정하에 최소값과 최대값을 이용하여 0~1사이의 값으로 변환
# Min-Max scaling
def min_max_scaling(x):
    x_np = np.asarray(x)
    return (x_np - x_np.min()) / (x_np.max() - x_np.min() + 1e-7)  # 1e-7은 0으로 나누는 오류 예방차원


# 정규화된 값을 원래의 값으로 되돌린다
# 정규화하기 이전의 org_x값과 되돌리고 싶은 x를 입력하면 역정규화된 값을 리턴한다
def reverse_min_max_scaling(org_x, x):
    org_x_np = np.asarray(org_x)
    x_np = np.asarray(x)
    return (x_np * (org_x_np.max() - org_x_np.min() + 1e-7)) + org_x_np.min()


