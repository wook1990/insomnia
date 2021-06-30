import pandas as pd
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from bs4 import BeautifulSoup
from sklearn.linear_model import LinearRegression

boston = load_boston(())
# html 출력
soup = BeautifulSoup(boston.DESCR, 'html.parser')

# 데이터 프레임으로 변환(그냥해봤어)
data = pd.DataFrame(data=boston.data, columns=boston["feature_names"])
target = pd.DataFrame(data=boston.target, columns=["target"])
boston_df = pd.merge(data, target, how="left", left_index=True, right_index=True)
boston_df.to_csv("boston_house_price.csv", index=False)
# 데이터 분할
train_df, test_df = train_test_split(boston_df, train_size=0.7, test_size=0.3)
len(train_df)
len(test_df)

# 기존 Boston Data array 를 분할하여 Train, test 분할
x_train, x_test, y_train, y_test = train_test_split(boston.data, boston.target, random_state=42)
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

# 선형 회귀 모델 선언
boston_lr = LinearRegression()
# 모델 훈련
boston_lr.fit(x_train, y_train)
# test 데이터로 결정계수 계산
boston_lr.score(x_test, y_test)
# 결정계수 :  0.684426
# 예측값 계산
y_hat = boston_lr.predict(x_test)
# 독립변수앞 coeffidence 값
var = boston_lr.coef_

boston_lr.intercept_
# 선형회귀 모형의 평가 매트릭
from sklearn import metrics
import sys

sys.path.append('D:/Python/Haram/Estimator/')
mse = metrics.mean_squared_error(y_hat, y_test)
mae = metrics.mean_absolute_error(y_hat, y_test)
'''
import forcasting_metrics as fm
fm.mae(y_test, y_hat)
fm.mape(y_test, y_hat)
fm.mpe(y_test,y_hat)
'''
# 파라미터 설정에따른 값변화
# LinearRegression(fit_intercept= bool, default = True, normalize=bool, default=False, copy_x=bool, default=True, n_jobs=int, default=None)
# fit_intercept : 절편의 상수항 존재 여부에 따른 모델 학습, 기본값은 True이며 False로 변경시
# 데이터에셋이 원점에 맞추어져 있는경우 사용할 수 있음
# normalize : fit_intercept option이 False 이면 무시되며,
#             True인경우 평균을 빼고 l2-norm으로 나누어 모델 학습전에 정규화 작업 수행
# copy_x : 원본데이터를 변경하지 않으려면 훈련 데이터를 복사해야하는데, copy_X 파라미터로 조절 할 수 있다.
#          기본값은 True로 훈련 데이터를 복사하여 사용
# n_job : 복수개 타겟을 가진 데이터 셋을 훈련할 때, n_job 매개변수를 활용하여
#         병렬훈련을 활용할 수 있게 해주는 파라미터

boston_lr_2 = LinearRegression(fit_intercept=False)
boston_lr_2.fit(x_train, y_train)
boston_lr_2.score(x_test, y_test)
y_hat_2 = boston_lr_2.predict(x_test)

boston_lr_3 = LinearRegression(fit_intercept=True, normalize=True)
boston_lr_3.fit(x_train, y_train)
boston_lr_3.score(x_test, y_test)
y_hat_3 = boston_lr_3.predict(x_test)

boston_lr_4 = LinearRegression(copy_X=False)
boston_lr_4.fit(x_train, y_train)
boston_lr_4.score(x_test, y_test)
y_hat_4 = boston_lr_4.predict(x_test)

boston_lr_5 = LinearRegression(copy_X=True)
boston_lr_5.fit(x_train, y_train)
boston_lr_5.score(x_test, y_test)
y_hat_5 = boston_lr_4.predict(x_test)

from sklearn.linear_model import Ridge
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split

boston = load_boston(())
x_train, x_test, y_train, y_test = train_test_split(boston.data, boston.target, random_state=42)
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

ridge = Ridge()
ridge.fit(x_train, y_train)
ridge.score(x_test, y_test)
y_hat = ridge.predict(x_test)

# alpa 변화에 따른 결정계수 변화
ridge_2 = Ridge(alpha=5)
ridge_2.fit(x_train, y_train)
ridge_2.score(x_test, y_test)

ridge_3 = Ridge(alpha=10)
ridge_3.fit(x_train, y_train)
ridge_3.score(x_test, y_test)

ridge_4 = Ridge(alpha=0.1)
ridge_4.fit(x_train, y_train)
ridge_4.score(x_test, y_test)

print("Basic Ridge : ", round(ridge.score(x_test, y_test), 3),
      "\nalpha 5 Ridge : ", round(ridge_2.score(x_test, y_test), 3),
      "\nalpha 10 Ridge : ", round(ridge_3.score(x_test, y_test), 3),
      "\nalpha 0.1 Ridgd : ", round(ridge_4.score(x_test, y_test), 3))
# 평가
from sklearn import metrics
import sys

sys.path.append('D:/Python/Haram/Estimator/')
mse = metrics.mean_squared_error(y_hat, y_test)
mae = metrics.mean_absolute_error(y_hat, y_test)

from sklearn.linear_model import RidgeCV

# 기본적인 RidgeCV모델
ridge_cv = RidgeCV(alphas=[0.01, 0.1, 1.0, 5.0, 10.0])
ridge_cv.fit(x_train, y_train)
ridge_cv.score(x_test, y_test)
print("Ridge r2 score : ", round(ridge_cv.score(x_test, y_test), 3))
print("alpha : ", ridge_cv.alpha_)

# 범위값에 따른 ridge_model
import numpy as np

ridge_cv_rg = RidgeCV(np.arange(0.01, 10, 0.01), scoring='r2', store_cv_values=True)
ridge_cv_rg.fit(x_train, y_train)
print("Ridge cv Value : ", ridge_cv_rg.cv_values_)
print("RidgeCV Score : ", round(ridge_cv_rg.score(x_test, y_test), 3))
print("RidgeCV best Alpha : ", ridge_cv_rg.alpha_)

from sklearn.linear_model import Lasso

lasso = Lasso(alpha=1.0, fit_intercept=True, normalize=False, precompute=False,
              copy_X=True, max_iter=1000, tol=0.0001, warm_start=False, positive=False,
              random_state=None, selection='cyclic')
lasso.fit(x_train, y_train)
lasso.score(x_test, y_test)
print("lasso r2 : ", round(lasso.score(x_test, y_test), 3))
lasso.predict(x_test)

from sklearn.linear_model import ElasticNet

elasticNet = ElasticNet(alpha=1.0, l1_ratio=0.5,
                        fit_intercept=True, normalize=False,
                        precompute=False, max_iter=1000,
                        copy_X=True, tol=0.0001,
                        warm_start=False, positive=False,
                        random_state=None, selection='cyclic')

elasticNet.fit(x_train, y_train)
elasticNet.score(x_test, y_test)
print("ElasticNet r2 : ", round(elasticNet.score(x_test, y_test), 3))
elasticNet.predict(x_test)

from sklearn.linear_model import SGDRegressor

sgd_rg = SGDRegressor(loss='squared_loss', penalty='l2', alpha=0.0001,
                      l1_ratio=0.15, fit_intercept=True, max_iter=1000,
                      tol=0.001, shuffle=True, verbose=0, epsilon=0.1,
                      random_state=None, learning_rate='invscaling', eta0=0.01,
                      power_t=0.25, early_stopping=False, validation_fraction=0.1,
                      n_iter_no_change=5, warm_start=False, average=False)

sgd_rg.fit(x_train, y_train)
sgd_rg.score(x_test, y_test)
sgd_rg.coef_

y_hat = sgd_rg.predict(x_test)

from sklearn.linear_model import SGDRegressor
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split

boston = load_boston(())
x_train, x_test, y_train, y_test = train_test_split(boston.data, boston.target, random_state=42)
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)
sgd_rg = SGDRegressor(loss='squared_loss', penalty='l2', alpha=0.0001,
                      l1_ratio=0.15, fit_intercept=True, max_iter=1000,
                      tol=0.001, shuffle=True, verbose=0, epsilon=0.1,
                      random_state=None, learning_rate='invscaling', eta0=0.01,
                      power_t=0.25, early_stopping=False, validation_fraction=0.1,
                      n_iter_no_change=5, warm_start=False, average=False)

sgd_rg.fit(x_train, y_train)
sgd_rg.score(x_test, y_test)
print("SGDRegressor R2 : ", round(sgd_rg.score(x_test, y_test), 3))



from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.linear_model import HuberRegressor

boston = load_boston(())
x_train, x_test, y_train, y_test = train_test_split(boston.data, boston.target, random_state=42)
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

huber = HuberRegressor(epsilon=1.35, max_iter=100, alpha=0.0001, warm_start=False, fit_intercept=True, tol=1e-05)
huber.fit(x_train, y_train)
huber.score(x_train, y_train)
print("Train R2 Score : ", round(huber.score(x_train, y_train),3))
# Train R2 Score :  0.707
huber.score(x_test, y_test)
print("Test R2 Score : ", round(huber.score(x_test, y_test),3))
# Test R2 Score :  0.619
y_hat = huber.predict(x_test)

'''
array([26.45477202, 31.90441301, 12.64628524, 24.43920892, 19.32759573,
       21.37040786, 18.3930275 , 16.01980675, 20.26953432, 20.1629845 ,
       21.52397407, 19.31915431, -8.02057075, 21.03551137, 19.15406102,
       25.79232343, 21.0572328 ,  4.76930124, 39.01506258, 17.70460392,
       27.0359274 , 28.45074374, 13.57739774, 25.53060099, 17.26221706,
       12.75735116, 20.8115369 , 16.21688368, 19.84450675, 18.81083089,
       19.33321067, 25.77575706, 25.04546552, 16.38808353, 15.4285586 ,
       17.87757888, 31.75537273, 22.03596275, 21.81186266, 24.68740527,
       13.92692259, 30.96181673, 40.95162112, 17.02559207, 25.81605201,
       15.79691995, 15.20221475, 26.0939191 , 18.88865359, 28.14400638,
       21.09114144, 33.19692599, 18.12923359, 26.08843364, 37.70064826,
       21.9515668 , 19.15140346, 31.4812219 , 25.23500146, 14.53135024,
       26.14472199, 32.45058916, 30.23173536, 16.65842686, 22.34527604,
       13.92078712, 20.13185079, 25.46877424, 29.37556523, 12.49586374,
       20.75312642, 25.13488472, 11.65019151, 18.95367891, 21.93211302,
        4.69255032, 20.54269041, 39.52881794, 18.21310414,  6.69805198,
       20.36535796, 11.22681316, 20.40396305,  8.1114256 , 21.93646296,
       28.82804198, 20.51431904, 26.23740137, 27.16387491, 20.83624657,
       24.004625  ,  4.21664391, 20.53637405, 16.84963287,  6.60689559,
       21.14038638, 21.12566165, -2.7604581 , 14.60032448, 14.90868708,
       22.11873973, 23.00607357,  8.66323071, 20.43231473, 22.43043434,
       11.66630785, 18.84933664, 26.87110691, 23.50349522, 24.17846626,
        9.38224732, 18.22560617, 25.12863385, 25.94892219, 31.64038765,
       14.82025449, 35.7931858 , 14.46144103, 19.72007173, 27.59390602,
       16.18364133, 25.9856052 ,  0.92950092, 21.17817088, 26.07446911,
       23.1982101 , 25.01895148])
'''
print("Coeffidence : \n", huber.coef_)
'''
Coeffidence : 
 [-0.16155126  0.01178409  0.03785247  0.49553497  0.31155027  6.0371921
 -0.0236153  -0.59083438  0.25440629 -0.0175301  -0.43071119  0.0115446
 -0.34697967]
'''
print("Intercept : " , huber.intercept_)
# Intercept :  0.6098432714138542
print("Train Data Outliers : \n", huber.outliers_)
'''
Train Data Outliers : 
 [ True False  True False False False False  True False False False False
 False False False False  True False False  True False False  True False
  True False False False False False False False False  True False False
 False False False  True False False  True  True False False  True False
 False False False  True False False  True False False False False False
  True  True False False False False False  True False False False False
 False False False False False False False False  True  True False False
 False False  True False False False False False False  True  True False
  True False False False False  True False False False False False False
 False False  True  True False False False False  True  True False  True
  True False False False False False False False False False  True False
  True False False False False  True False False False False False False
 False False  True False False  True False  True  True False False False
 False  True False False False False False False False False False False
  True False False False False  True False False False  True False False
 False False False False False False False  True  True False False False
 False False  True  True False  True False  True False  True False False
  True False False False False False False False  True False False False
 False  True False False False False False  True False  True False  True
  True  True  True False False False False False False  True False False
 False False False  True False False  True False  True False False False
  True False False False False False False False False False  True False
 False False False  True False  True False False False  True False False
 False False False False False False False False False False False  True
 False False  True  True False False False False False False False  True
 False  True False False False False False False  True False False False
 False False False False False False  True False False  True False False
 False False False False False False False False False False False False
 False False False False False False False False False False False False
 False False False False False False  True False  True  True False False
 False False False  True False False False  True  True False  True False
 False False False False False False False]
'''

import numpy as np

from sklearn.linear_model import RANSACRegressor
ranscan = RANSACRegressor(base_estimator=None, min_samples=None, residual_threshold=None, is_data_valid=None,
                          is_model_valid=None, max_trials=100, max_skips=np.inf, stop_n_inliers=np.inf, stop_score=np.inf,
                          stop_probability=0.99, loss='absolute_loss', random_state=None)

ranscan.fit(x_train,y_train)
ranscan.score(x_train, y_train)
ranscan.score(x_test, y_test)
ranscan.predict(x_test)

# Attribute
ranscan.n_skips_invalid_model_
ranscan.n_skips_invalid_data_
ranscan.n_skips_invalid_model_
ranscan.n_trials_
ranscan.estimator_
ranscan.inlier_mask_


from sklearn.linear_model import TheilSenRegressor
theilsen = TheilSenRegressor(fit_intercept=True, copy_X=True, max_subpopulation=10000.0,
                             n_subsamples=None, max_iter=300, tol=0.001, random_state=None,
                             n_jobs=None, verbose=False)
theilsen.fit(x_train, y_train)
theilsen.score(x_train, y_train)
theilsen.score(x_test, y_test)
theilsen.predict(x_test)


# MLPRegressor

from sklearn.neural_network import MLPRegressor

mlp_reg = MLPRegressor(hidden_layer_sizes=(100, ), activation='relu', solver='adam', alpha=0.0001, batch_size='auto',
                       learning_rate='constant', learning_rate_init=0.001, power_t=0.5, max_iter=200, shuffle=True,
                       random_state=None, tol=0.0001, verbose=False, warm_start=False, momentum=0.9,
                       nesterovs_momentum=True, early_stopping=False, validation_fraction=0.1, beta_1=0.9,
                       beta_2=0.999, epsilon=1e-08, n_iter_no_change=10, max_fun=15000)

# 모델 학습
mlp_reg.fit(x_train, y_train)

# 모델 적합성 검증(Train Score)
mlp_reg.score(x_train,y_train)

# 모델 적합성 검증(Test Score)
mlp_reg.score(x_test,y_test)

# predict 함수는 예측된 클래스의 결과를 반환
y_hat = mlp_reg.predict(x_test)


from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor

boston = load_boston(())
x_train, x_test, y_train, y_test = train_test_split(boston.data, boston.target, random_state=42)
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

mlp_reg = MLPRegressor(hidden_layer_sizes=(100, ), activation='relu', solver='adam', alpha=0.0001, batch_size='auto',
                       learning_rate='constant', learning_rate_init=0.001, power_t=0.5, max_iter=200, shuffle=True,
                       random_state=None, tol=0.0001, verbose=False, warm_start=False, momentum=0.9,
                       nesterovs_momentum=True, early_stopping=False, validation_fraction=0.1, beta_1=0.9,
                       beta_2=0.999, epsilon=1e-08, n_iter_no_change=10, max_fun=15000)

# 모델 학습
mlp_reg.fit(x_train, y_train)

# 모델 적합성 검증(Train Score)
print("Train R2 score : ", round(mlp_reg.score(x_train,y_train),3))
# Train R2 score :  0.672

# 모델 적합성 검증(Test Score)
print("Test R2 score : ", round(mlp_reg.score(x_test,y_test),3))
# Test R2 score :  0.672

# predict 함수는 예측된 클래스의 결과를 반환
y_hat = mlp_reg.predict(x_test)

# 모델 결과 검증
from sklearn.metrics import mean_squared_error, mean_absolute_error
print("MSE : " , round(mean_squared_error(y_test, y_hat),3))
# MSE :  22.97
print("MAE : " , round(mean_absolute_error(y_test, y_hat),3))
# MAE :  3.571



#-----------------------------------------------------------------#
import xgboost

# 모델선언
xgb_reg = xgboost.XGBRegressor(base_score=-.5, booster='gbtree', colsample_bylevel=1, colsample_bytree=1, gamma=0, learning_rate=0.08,
                     max_delta_step=0, max_depth=6, min_child_weight=1, missing=None, n_estimators=100, n_jobs=1,
                     nthread=None, objective='reg:linear', random_state=0, reg_alpha=0, reg_lambda=1, scale_pos_weight=1,
                     seed=None, silent=True, subsample=0.75)
# 모델 학습
xgb_reg.fit(x_train,y_train)

# 모델 적합성 검증(Train Score)
xgb_reg.score(x_train,y_train)

# 모델 적합성 검증(Test Score)
xgb_reg.score(x_test,y_test)

# predict 함수는 예측 값을 반환
y_hat = xgb_reg.predict(x_test)


from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
import xgboost

boston = load_boston(())
x_train, x_test, y_train, y_test = train_test_split(boston.data, boston.target, random_state=42)
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

xgb_reg = xgboost.XGBRegressor(base_score=-.5, booster='gbtree', colsample_bylevel=1, colsample_bytree=1, gamma=0, learning_rate=0.08,
                     max_delta_step=0, max_depth=6, min_child_weight=1, missing=None, n_estimators=100, n_jobs=1,
                     nthread=None, objective='reg:linear', random_state=0, reg_alpha=0, reg_lambda=1, scale_pos_weight=1,
                     seed=None, silent=True, subsample=0.75)

# 모델 학습
xgb_reg.fit(x_train, y_train)

# 모델 적합성 검증(Train Score)
print("Train R2 score : ", round(xgb_reg.score(x_train,y_train),3))
# Train R2 score :  0.995

# 모델 적합성 검증(Test Score)
print("Test R2 score : ", round(xgb_reg.score(x_test,y_test),3))
# Test R2 score :  0.859

# predict 함수는 예측된 클래스의 결과를 반환
y_hat = xgb_reg.predict(x_test)

# 모델 결과 검증
from sklearn.metrics import mean_squared_error, mean_absolute_error
print("MSE : " , round(mean_squared_error(y_test, y_hat),3))
# MSE :  9.872
print("MAE : " , round(mean_absolute_error(y_test, y_hat),3))
# MAE :  2.01

