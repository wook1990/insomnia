import pandas as pd
import numpy as np
import os
import warnings
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
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


def reverse_min_max_scaling(org_x, x):
    org_x_np = np.asarray(org_x)
    x_np = np.asarray(x)
    return (x_np * (org_x_np.max() - org_x_np.min() + 1e-7)) + org_x_np.min()

# data scaling
def Min_Max_Scaler(dataframe):
    scaler = MinMaxScaler()
    scaler.fit(dataframe)
    scale_df = pd.DataFrame(scaler.transform(dataframe), columns  = dataframe.columns)
    return scale_df

# data load
def data_load(path,file_name, Date_col):

    df = pd.read_csv(path + file_name, index_col = False, low_memory = False)
    df[Date_col] = pd.to_datetime(df[Date_col], format='%Y%m%d')
    df = df.set_index(Date_col)

    return df

def split_data(DataFrame, test_size,target_col, window_size):

    # train_test 분리

    train = DataFrame[:-test_size]
    test = DataFrame[-test_size:]

    # feature, target 분리
    feature_cols = list(DataFrame.columns)
    feature_cols.remove(target_col)

    # train_data
    train_feature = train[feature_cols]
    train_label = pd.DataFrame(train[target_col])

    # test_data
    test_feature = test[feature_cols]
    test_label = pd.DataFrame(test[target_col])

    # 예측 주기에 따른 timestep 생성
    train_feature, train_label = make_dataset(train_feature, train_label, window_size)
    test_feature, test_label = make_dataset(test_feature, test_label, window_size)

    print(train_feature.shape, train_label.shape)
    print(test_feature.shape, test_label.shape)

    return train_feature, train_label, test_feature, test_label


def LSTM_MODEL(train_feature, train_label, test_size, units=4, loss="mean_squared_error",
               optimizer='adam', monitor="var_loss", epochs=200, batch_size=16):
    # train_valid 분할
    x_train, x_valid, y_train, y_valid = train_test_split(train_feature, train_label, test_size=test_size)

    print(x_train.shape, x_valid.shape)
    print(y_train.shape, y_valid.shape)

    # Sequential 모델 레이어 생성
    model = Sequential()
    # LSTM 레이어 추가
    model.add(LSTM(
        units,
        input_shape=(train_feature.shape[1], train_feature.shape[2]),
        activation='relu',
        return_sequences=False
    ))
    # Dense 레이어 추가
    model.add(Dense(1))
    print(model.summary())
    # 모델 컴파일 활성함수 loss_function 지정 , 최적화함수 지정
    model.compile(loss=loss, optimizer=optimizer)
    # 얼리스탑핑 ( val_loss의 변화를 비교, 같은 값 5번 일시 중지)
    early_stop = EarlyStopping(monitor=monitor, patience=3)
    # 모형 가중치 저장 파일 경로 선언
    # 모형 가중치 판다 및 모델 저장
    filename = os.path.join(file_path, 'stock_checkpoint.h5')
    checkpoint = ModelCheckpoint(filename, monitor=monitor, verbose=1, save_best_only=True, mode='auto')

    # 모형 학습
    model.fit(x_train, y_train,
              epochs=epochs,
              batch_size=batch_size,
              validation_data=(x_valid, y_valid),
              callbacks=[early_stop, checkpoint])

    return model, filename


if __name__ == "__main__":

    # data load
    #df = pd.read_csv("E:\Dacon\study\LSTM\equity_output_005490_H.csv", index_col=False, low_memory=False)

    work_dir = "C:/Users/wook1/Documents/python_work/study/LSTM/"
    file_dir = "data/equity_output_005490_H.csv"
    df = data_load(work_dir, file_dir, "eod_date")

    # Step1
    # Non Scaling Modeling
    '''    train_feature, train_label, test_feature, test_label = split_data(df, 100, "target_H", 5)

    model, filename = LSTM_MODEL(train_feature, train_label, 0.2, units=4, loss="mean_squared_error",
                                 optimizer='adam', monitor="var_loss", epochs=200, batch_size=16)

    try:
        model.load_weights(filename)
        y_pred = model.predict(test_feature)
        y_df = pd.DataFrame({"y_real": test_label.reshape(1, -1)[0].astype(np.float32),
                             "y_pred": y_pred.reshape(1, -1)[0].astype(np.float32)})
        y_df.to_csv(work_dir + "/non_scaling_prediction.csv", index=False)
        mape = mape(test_label, y_pred)
        print(mape)
    except:
        y_pred = model.predict(test_feature)
        y_df = pd.DataFrame({"y_real": test_label.reshape(1, -1)[0].astype(np.float32),
                             "y_pred": y_pred.reshape(1, -1)[0].astype(np.float32)})
        y_df.to_csv(work_dir + "/prediction.csv", index=False)
        mape = mape(test_label, y_pred)
        print(mape)
    '''

    # step2 Scaling
    scale_df = Min_Max_Scaler(df)

    train_feature, train_label, test_feature, test_label = split_data(scale_df, 100, "target_H", 5)

    model, filename = LSTM_MODEL(train_feature, train_label, 0.2, units=4, loss="mean_squared_error",
                   optimizer='adam', monitor="var_loss", epochs=200, batch_size=16)

    try:
        model.load_weights(filename)
        y_pred = model.predict(test_feature)
        y_df = pd.DataFrame({"y_real": test_label.reshape(1,-1)[0].astype(np.float32),
                             "y_pred": y_pred.reshape(1,-1)[0].astype(np.float32)})
        y_df.to_csv(work_dir + "result/scale_prediction.csv", index=False)
        mape2 = mape(test_label, y_pred)
        print(mape2)
    except:
        y_pred = model.predict(test_feature)
        y_df= pd.DataFrame({"y_real": test_label.reshape(1,-1)[0].astype(np.float64),
                             "y_pred": y_pred.reshape(1,-1)[0].astype(np.float64)})
        y_df.to_csv(work_dir + "result/scale_prediction.csv", index=False)
        mape2 = mape(test_label, y_pred)
        print(mape2)

