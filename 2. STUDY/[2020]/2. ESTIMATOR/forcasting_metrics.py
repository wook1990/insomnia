import numpy as np
import pandas as pd
from sklearn import linear_model
# 실제값이 0인경우 계산 오류를 방지하기위한 상수
EPSILON = 1e-10

def residual(actual:np.ndarray, predicted : np.ndarray):
    """ Simple residual"""
    return actual - predicted

def sse(actual: np.ndarray , predicted : np.ndarray):
    """Sum of Squared Error"""
    return (residual(actual, predicted)**2).sum()

def ssr(actual : np.ndarray, predicted : np.ndarray):
    """Sum of Squares due to regression """
    return ((predicted - actual.mean())**2).sum()

def sst(actual : np.ndarray, predicted: np.ndarray):
    """ Total Sum of Squared """
    # sst = ((actual + actual.mean())**2).sum()
    return sse(actual, predicted) + ssr(actual, predicted)

def abs_residual(actual : np.ndarray, predicted : np.ndarray):
    """" Absolute residual """
    return np.abs(residual(actual, predicted))

def percentage_error(actual : np.ndarray, predicted : np.ndarray):
    """
    Percentage error

    Not mulipled by 100
    """
    return residual(actual, predicted)/(actual + EPSILON)
def r_squared(actual: np.ndarray, predicted: np.ndarray):
    """  coefficient of determination """
    return 1 - (sse(actual, predicted)/sst(actual, predicted))


def mse(actual : np.ndarray, predicted : np.ndarray):
    """ Mean Squared Error"""
    return np.mean(np.square(residual(actual, predicted)))

def rmse(actual : np.ndarray, predicted : np.ndarray):
    """ Root Mean Squared Error """
    return np.sqrt(mse(actual, predicted))

def mae(actual : np.ndarray , predicted : np.ndarray):
    """ Mean Absolute Error """
    return np.mean(np.abs(residual(actual, predicted)))

def mpe(acutal: np.ndarray , predicted : np.ndarray):
    """
    Mean Percentage Error
    Not multiplied by 100
    """
    return np.mean(percentage_error(acutal, predicted))

def mape(actual: np.ndarray, predicted : np.ndarray):
    """
    Mean Asolute Percentage Error
    Properties:
       + Easy to interpret
       + Scale independent
       - Biased, no symetric
       - Undefined when actual[t]=0
    Not Multiplied by 100
    """
    return np.mean(np.abs(percentage_error(actual, predicted)))


METRICS = {
    "sse" : sse,
    "ssr" : ssr,
    "sst" : sst,
    "r-squared" : r_squared,
    "mse" : mse,
    "rmse" : rmse,
    "mae" : mae,
    "mpe" : mpe,
    "mape" : mape
}

def evaluate(actual: np.ndarray, predicted: np.ndarray, metrics=('mae', 'mse', 'smape', 'umbrae')):
    results = {}
    for name in metrics:
        try:
            results[name] = METRICS[name](actual, predicted)
        except Exception as err:
            results[name] = np.nan
            print('Unable to compute metric {0}: {1}'.format(name, err))
    return results

def evaluate_all(actual: np.ndarray, predicted: np.ndarray):
    return evaluate(actual, predicted, metrics=set(METRICS.keys()))



if __name__ == "__main__":
    # data load
    # 삼성증권 주가 데이터
    def data_load(path, file_name, date_col, drop_col):
        df = pd.read_csv(path + file_name, index_col=False, low_memory=False)
        df[date_col] = pd.to_datetime(df[date_col], format='%Y-%m-%d')
        df = df.set_index(date_col)
        df.drop(drop_col, axis=1, inplace=True)
        df.sort_index(ascending=True, inplace=True)
        return df


    work_dir = "D:/Aaltair/KnowledgeHub/Filesystem/data/samsung_elt/"
    file_path = "price2.csv"
    df = data_load(work_dir, file_path, "eod_date", "item_code")

    # 종가 예측(price_close) 모델 생성
    linear_regresion = linear_model.LinearRegression()
    linear_regresion.fit(X=pd.DataFrame(df.iloc[:, :4]), y=df.iloc[:, 4])
    prediction = linear_regresion.predict(X=pd.DataFrame(df.iloc[:, :4]))
    actual = np.array(df.iloc[:, 4])
    print("a value = ", linear_regresion.intercept_)
    print("a value = ", linear_regresion.coef_)

    # 평가지표 출력
    print(evaluate_all(actual, prediction))
    asset_df = pd.DataFrame.from_dict([evaluate_all(actual, prediction)])
    print(asset_df)
