import numpy
import pandas
from sklearn import ensemble
import xgboost
import glob

DATA_PATH= "/data/2.TimeSeriesDecompose\\"


def _predict(filename : str, index : str, target : str, model : object):
    '''
    :param filename: 시계열 분해된 데이터셋
    :param index: 시계열 인덱스 컬럼명
    :param target: 예측 타겟 컬럼명
    :param model : 예측할 모델 객체
    :save_file : 에측결과 파일 생성
    '''
    window_size = 550
    period_day = filename.split("_")[1].replace(".csv","")
    model_nm = model.__str__().split("(")[0]
    df_train = pandas.read_csv(filename)
    df_train = df_train.set_index([index])
    list_predict = list()

    for interval in range(30, -1, -1):
        df_train_X = df_train.tail(window_size + interval + 1).head(window_size + 1)
        #print("start date : ", df_train_X.index[1], ", end date : ", df_train_X.index[-1], ", length : ",len(df_train_X))

        df_tests_X = df_train_X.tail(1)
        df_train_X.drop(df_train_X.tail(1).index, inplace=True)

        df_train_Y = df_train_X[target]
        df_tests_Y = df_tests_X[target]

        model.fit(df_train_X, df_train_Y)

        real_value = df_tests_Y.values.tolist()[0]
        predict_value = model.predict(df_tests_X)
        list_predict.append({'train_date' : df_tests_X.index.tolist()[0],
                             target + '_real_value' : real_value,
                             target + '_predict_value' : predict_value[0],
                             target + '_error' : numpy.abs((predict_value - real_value) / real_value * 100)[0]})

        print('Train Date:[', df_tests_X.index.tolist()[0], '], Real:[', real_value, '], Predict:[', predict_value,
              '], ERROR:[', numpy.abs((predict_value - real_value) / real_value * 100), ']')


    model_res_df = pandas.DataFrame(list_predict)
    # model_res_df.to_csv(DATA_PATH + model_nm+ "_" + target + "_" + period_day + "_predict_result.csv", index=False)
    return model_res_df
if __name__ == '__main__':

    #-----------------------------------------------------------
    # 모델 선언부
    '''
    model = ensemble.RandomForestRegressor(criterion='mse',
                                           random_state=0,
                                           n_estimators=100,
                                           max_features='auto',
                                           min_samples_split=2,
                                           min_samples_leaf=1,
                                           max_depth=50,
                                           )
    '''

    # XGBOOST - Model Define
    model = xgboost.XGBRegressor(
                                 object="reg:linear",
                                 booster='gblinear',
                                 learning_rate=0.01,
                                 n_estimators=300,
                                 max_depth=6,
                                 gamma=0.2,
                                 rag_lambda=0.1,
                                 reg_alpha=0.01,
                                 verbosity=0,
                                 eta=0.05
                                )

    #-----------------------------------------------------------

    file_list = glob.glob(DATA_PATH + "*days.csv")
    total_df = pandas.DataFrame()
    for filename in iter(file_list):
        result_df = pandas.DataFrame()
        for col in iter(['target','trend','season','resid']):
            df = _predict(filename, "eod_date", col, model)
            df = df.set_index(["train_date"])
            result_df = pandas.concat([result_df, df], axis=1)
            print("-------------------------------------------------------------------------------")
        period_day = filename.split("_")[1].replace(".csv", "")
        result_df["period"] = period_day
        result_df.to_csv(DATA_PATH + model.__str__().split("(")[0] + "_" + period_day + "_result.csv", index=True)

        print("====================================================================")
        total_df = pandas.concat([total_df, result_df],axis=0)
    total_df["addictive_pred"] = total_df["trend_predict_value"] + total_df["season_predict_value"] + total_df["resid_predict_value"]
    total_df["real_addictive_mape"] = numpy.abs((total_df["target_real_value"] - total_df["addictive_pred"])/total_df["target_real_value"] * 100)
    total_df.to_csv(DATA_PATH + model.__str__().split("(")[0] + "_total_result.csv", index=True)

    # trend 하고  seasonality에서 한번씩 값이 튀는데
    # 주기가 커질수록 데이터의 특성을 잡아내지 못하는 걸까?
