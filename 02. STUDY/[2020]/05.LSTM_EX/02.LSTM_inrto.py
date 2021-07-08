import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
df = pd.read_csv("E:/Dacon/study/LSTM/samsung_stock.csv", index_col=False)
df.columns = ["eod_date","price_open","price_max","price_min","price_close","trade_cnt"]

df.eod_date = pd.to_datetime(df.eod_date, format = '%Y%m%d')
df = df.sort_values("eod_date")
df = df.set_index("eod_date")

test_size = 200
train = df[:-test_size]
test = df[-test_size:]

def make_dataset(data, label, window_size=20):
    feature_list = []
    label_list = []
    for i in range(len(data) - window_size):
        feature_list.append(np.array(data.iloc[i:i+window_size]))
        label_list.append(np.array(label.iloc[i+window_size]))
    return np.array(feature_list), np.array(label_list)

# feature  와 label 정의
feature_cols = ["price_open","price_max","price_min","trade_cnt"]
label_cols = ["price_close"]

train_feature = train[feature_cols]
train_label = train[label_cols]

# train dataset
train_feature, train_label = make_dataset(train_feature, train_label, 20)

# train, validation set 생성
from sklearn.model_selection import train_test_split
x_train, x_valid, y_train, y_valid = train_test_split(train_feature, train_label, test_size=0.2)

print(x_train.shape, x_valid.shape)


test_feature = test[feature_cols]
test_label = test[label_cols]
# test dataset ( 실제 예측 데이터)
test_feature, test_label = make_dataset(test_feature, test_label, 20)
print(test_feature.shape, test_label.shape)



# LSTM 모델
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import LSTM


model = Sequential()
model.add(LSTM(
    16,
    input_shape = (train_feature.shape[1], train_feature.shape[2]),
    activation = 'relu',
    return_sequences=False
))
model.add(Dense(1))


# model Train
model.compile(loss = "mean_squared_error", optimizer = 'adam')
early_stop = EarlyStopping(monitor='var_loss',patience=5)
model_path = "E:\\Dacon\\study\\LSTM\\"
filename = os.path.join(model_path, 'example_checkpoint.h5')
checkpoint = ModelCheckpoint(filename, monitor='var_loss', verbose=1, save_best_only=True, mode='auto')

model.fit(x_train, y_train,
                 epochs=200,
                 batch_size = 16,
                 validation_data=(x_valid, y_valid),
                 callbacks= [early_stop, checkpoint])

model.load_weights(filename)
pred = model.predict(test_feature)


def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


mape = mean_absolute_percentage_error(test_label, pred)
