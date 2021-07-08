# 1. 실제값, 예측값 분리
# 2. 잔차, 잔차제곱합, 회귀 제곱합, 총제곱합
# 3. 결정계수, 평균제곱오차, 평균제곱근오차, 평균백분율 오차, 평균 절대 백분율 오차

# 패키지 모듈
from sklearn import linear_model
import numpy as np
import pandas as pd


# data load
# 삼성증권 주가 데이터
def data_load(path,file_name, date_col, drop_col):

    df = pd.read_csv(path + file_name, index_col = False, low_memory = False)
    df[date_col] = pd.to_datetime(df[date_col], format='%Y-%m-%d')
    df = df.set_index(date_col)
    df.drop(drop_col, axis=1, inplace = True)
    df.sort_index(ascending=True,inplace = True)
    return df

work_dir = "D:/Aaltair/KnowledgeHub/Filesystem/data/samsung_elt/"
file_path = "price2.csv"
df = data_load(work_dir,file_path,"eod_date","item_code")



# 실제값이 0이되어 MAPE, MPE의 계산의 오류가 생기는 것을 방지 하기위한
# 실수값 지정
EPSILON = 1e-10
# 모델 생성
# 종가 예측(price_close) 모델 생성
linear_regresion = linear_model.LinearRegression()
linear_regresion.fit(X=pd.DataFrame(df.iloc[:,:4]),y= df.iloc[:,4])
prediction = linear_regresion.predict(X=pd.DataFrame(df.iloc[:,:4]))
actual = np.array(df.iloc[:,4])
print("a value = ",linear_regresion.intercept_)
print("a value = ",linear_regresion.coef_)


# 회귀식의 검정지표
# 1. 잔차
# actual = np.ndarray
# prediction = np.ndarray
residual = actual - prediction

# 2. SSE
sse = (residual**2).sum()

# 3. SSR
ssr = ((prediction-actual.mean())**2).sum()

# 4. SST(소수점의 차이가 존재하나 무시해도 상관없음)
sst = (((actual - actual.mean()))**2).sum()
sst = sse + ssr

# PE(percentage error)
pe = ((prediction-actual)/(actual+EPSILON))


# 회귀 모형 평가지표
# 모든 평가지표는 평균 값으로 확인 전체 에러텀에 대한 평균의 기본 개념
# 1. R-squared
r_squared = 1-(sse/sst)


# 1. mse
# 수식적 표현
mse = ((actual-prediction)**2).mean()
# 패키지 모듈 사용
from sklearn.metrics import mean_squared_error # mse
mse = mean_squared_error(actual, prediction)

# 2. mae
mse = (np.abs(actual-prediction)).mean()
# 패키지 모듈 사용
from sklearn.metrics import mean_absolute_error #mae
mae = mean_absolute_error(actual,prediction)


# 3. rmse
rmse = np.sqrt(mse)

# 5.mpe
# 평균백분율오류 오차와 실제값의 차이의 비율을 알수 있음
mpe =  (((prediction - actual)/(actual + EPSILON)).mean()) * 100
mpe = (pe.mean())*100

# 6.mape
mape = (np.abs((prediction - actual)/(actual + EPSILON)).mean()) * 100
mape = (np.abs(pe.mean())) * 100


# 각 행별 비교 값으로 결과 표현( 표현된 컬럼 결과의 평균이 각 성능평가지표)
# 성능평가지표의 평균,최대,최소, median등으로  성능의 평가지표 표현 가능 ?






