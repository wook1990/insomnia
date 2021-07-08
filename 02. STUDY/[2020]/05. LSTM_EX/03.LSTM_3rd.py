import pandas as pd
import numpy as np
import os
import warnings

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import LSTM

warnings.filterwarnings(action='ignore')

# window 기간에 과거 데이터를 통한 예측 데이터 생성 함수
def make_dataset(data, label, window_size):
    feature_list = []
    label_list = []
    for i in range(len(data) - window_size):
        feature_list.append(np.array(data.iloc[i:i+window_size]))
        label_list.append(np.array(label.iloc[i+window_size]))
    return np.array(feature_list), np.array(label_list)

# mape 평가지표 함수
def mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

# 표준화 함수
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
# 결과의 변환을 위한 함수
def reverse_min_max_scaling(org_x, x):
    org_x_np = np.asarray(org_x)
    x_np = np.asarray(x)
    return (x_np * (org_x_np.max() - org_x_np.min() + 1e-7)) + org_x_np.min()



if __name__ == "__main__":

    # data load
    df = pd.read_csv("C:/Users/wook1/Documents/python_work/study/LSTM/equity_output_005490_H.csv", index_col=False, low_memory=False)
    # convert int to datetime
    df.eod_date = pd.to_datetime(df.eod_date, format='%Y%m%d')
    # set_index datetime variable
    df = df.set_index("eod_date")

    # train, test 분리
    test_size = 100
    train = df[:-test_size]
    test = df[-test_size:]

    # feature, label 분리
    feature_cols = df.iloc[:,0:99].columns
    label_cols = df.iloc[:,100:].columns

    # train data
    train_feature = train[feature_cols]
    train_label = train[label_cols]

    # 예측 window에 따른 timestep 생성(20일 한달 주기로 다음 타겟 예측)
    train_feature, train_label = make_dataset(train_feature, train_label,10)
    print(train_feature.shape, train_label.shape)
    # train, validation dataset split
    from sklearn.model_selection import train_test_split
    x_train, x_valid, y_train, y_valid = train_test_split(train_feature, train_label, test_size=0.2)
    print(x_train.shape, x_valid.shape)
    print(y_train.shape, y_valid.shape)
    # test data(예측용)
    test_feature = test[feature_cols]
    test_label = test[label_cols]

    test_feature, test_label = make_dataset(test_feature, test_label, 10)
    print(test_feature.shape, test_label.shape)


    # LSTM 모형 생성


    # 모델 생성

    #model.reset_states()
    model = Sequential()
    # LSTM 레이어 추가
    model.add(LSTM(
        3,
        input_shape=(train_feature.shape[1], train_feature.shape[2]),
        activation='relu',
        return_sequences=False
    ))
    # Dense 레이어 추가
    model.add(Dense(1))
    model.summary()
    # 모델 컴파일 활성함수 loss_function 지정 , 최적화함수 지정
    model.compile(loss='mean_squared_error', optimizer='adam')
    # 얼리스탑핑 ( val_loss의 변화를 비교, 같은 값 5번 일시 중지)
    early_stop = EarlyStopping(monitor='loss', patience=2, verbose=1)
    model_path = "C:/Users/wook1/Documents/python_work/study/LSTM/"
    # 모형 가중치 저장 파일 경로 선언
    # 모형 가중치 판다 및 모델 저장
    filename = os.path.join(model_path, 'stock_checkpoint.h5')
    checkpoint = ModelCheckpoint(filename, monitor='loss', verbose=1, save_best_only=True, mode='auto')
    model.fit(x_train, y_train,
             epochs= 200,
             batch_size= 1,
             validation_data=(x_valid, y_valid),
             callbacks=[early_stop, checkpoint])
    try:
        model.load_weights(filename)
        y_pred = model.predict(test_feature)
        y_df = pd.DataFrame({"y_target": test_label, "y_pred": y_pred})
        y_df.to_csv(filename + "/prediction.csv", index=False)
        mape = mape(test_label, y_pred)
        print(mape)
    except:
        y_pred = model.predict(test_feature)
        pd.DataFrame({"y_target":test_label, "y_pred": y_pred}).to_csv(filename + "/prediction.csv", index=False)
        mape = mape(test_label, y_pred)
        print(mape)



