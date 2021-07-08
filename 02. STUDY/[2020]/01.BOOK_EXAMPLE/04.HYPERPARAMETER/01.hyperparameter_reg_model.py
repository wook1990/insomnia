from sklearn import linear_model
from sklearn import tree
from sklearn import ensemble
from sklearn import neural_network
import xgboost
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV,cross_val_score
import numpy as np
from sklearn.datasets import make_regression
from typing import Any, List, Optional, Dict, Tuple, Union

def np_list_arange(start: float, stop: float, step: float, inclusive: bool = False) -> List[float]:
    """
    Numpy arange returned as list with floating point conversion
    failsafes.
    """
    convert_to_float = (
        isinstance(start, float) or isinstance(stop, float) or isinstance(step, float)
    )
    if convert_to_float:
        stop = float(stop)
        start = float(start)
        step = float(step)
    stop = stop + (step if inclusive else 0)
    range = list(np.arange(start, stop, step))
    range = [
        start
        if x < start
        else stop
        if x > stop
        else float(round(x, 15))
        if isinstance(x, float)
        else x
        for x in range
    ]
    range[0] = start
    range[-1] = stop - step
    return range


# regression 모델을 위한 임의의 데이터 셋 생성
X, y, w = make_regression(n_samples=1000, n_features=50, coef=True, random_state=1, bias=3.5)

# 학습 및 검증 데이터 셋 분할
x_train, x_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=0)

# 1. Regressor

# 1) Ridge
# 하이퍼파라미터 그리드 선언

param_grid = {
              "alpha": np_list_arange(0.01,10,0.01,inclusive=True),
              "fit_intercept": [True, False],
              "normalize": [True, False]
              }

# 모델 선언
ridge_md = linear_model.Ridge()

# 활용 모델 Ridge, 하이퍼파라미터 그리드, Cross-Validation의 K-fold의 K값은 5
# 튜닝성능을 비교하는 score 지료는 r2-score를 사용
# 더 높은 score의 모형이 만들어 질때 최적의 모델로 업데이트하기 위하여 refit=True 선언
ridge_grid_search = GridSearchCV(ridge_md, param_grid, cv=5, scoring='r2', refit=True)

# GridSearchCV를 통한 최적의 모델 fitting 수행하여 모델 생성
ridge_grid_search.fit(x_train,y_train)

# model evaluation
# 교차 검증과 그리드 서치의 결과로 산정한 최적의 매개변수를 적용하여 전체 훈련 데이터 셋에 대해
# 훈련한 최종 모델에 테스트 데이터로 적용했을때 결과
print('test set score : {}'.format(ridge_grid_search.score(x_test,y_test)))
# test set score : 0.9999999996937393


# best parameter and best score
# 훈련 세트에서 수행한 교차검증에 대한 최적의 파라미터
print("best parameter : {}".format(ridge_grid_search.best_params_))
# best parameter : {'alpha': 0.01, 'fit_intercept': True, 'normalize': False}

# 훈련 세트에서 수행한 교차검증의 평균 정확도의 score
print("best score : {}".format(ridge_grid_search.best_score_))
# best score : 0.9999999994445513



# 2) Lasso
param_grid = {
            "fit_intercept": [True, False],
            "normalize": [True, False],
            "alpha": [0.0000001,0.000001,0.0001,0.001,0.01,0.0005,0.005,0.05,0.1,0.15,0.2,0.3,0.4,0.5,0.7,0.9,],
            "eps": [0.00001,0.0001,0.001,0.01,0.05,0.0005,0.005,0.00005,0.02,0.007,0.1,]
             }

# 모델 선언
lasso_md = linear_model.Lasso()

# GridSearchCV 선언
# 활용 모델 Lasso, 하이퍼파라미터 그리드, Cross-Validation의 K-fold의 K값은 5
# 튜닝성능을 비교하는 score 지료는 r2-score를 사용
# 더 높은 score의 모형이 만들어 질때 최적의 모델로 업데이트하기 위하여 refit=True 선언
lasso_grid_search = GridSearchCV(lasso_md, param_grid, cv=5, scoring='r2', refit=True)

# GridSearchCV를 통한 최적의 모델 fitting 수행하여 모델 생성
lasso_grid_search.fit(x_train,y_train)

# model evaluation
# 교차 검증과 그리드 서치의 결과로 산정한 최적의 매개변수를 적용하여 전체 훈련 데이터 셋에 대해
# 훈련한 최종 모델에 테스트 데이터로 적용했을때 결과
print('test set score : {}'.format(lasso_grid_search.score(x_test,y_test)))
# test set score : 0.9999999721836675


# best parameter and best score
# 훈련 세트에서 수행한 교차검증에 대한 최적의 파라미터
print("best parameter : {}".format(lasso_grid_search.best_params_))
# best parameter : {'alpha': 0.01, 'fit_intercept': True, 'normalize': False}

# 훈련 세트에서 수행한 교차검증의 평균 정확도의 score
print("best score : {}".format(lasso_grid_search.best_score_))
# best score : 0.9999999735203605


# 3) ElastiNet
# 파라미터 그리드 선언
param_grid = {
              "alpha": np_list_arange(0.01, 10, 0.01, inclusive=True),
              "l1_ratio": np_list_arange(0.01, 1, 0.01, inclusive=False),
              "fit_intercept": [True, False],
              "normalize": [True, False]
              }

# 모델 선언
Elastic_md = linear_model.ElasticNet()

# GridSearchCV 선언
# 활용 모델 ElastiNet, 하이퍼파라미터 그리드, Cross-Validation의 K-fold의 K값은 5
# 튜닝성능을 비교하는 score 지료는 r2-score를 사용
# 더 높은 score의 모형이 만들어 질때 최적의 모델로 업데이트하기 위하여 refit=True 선언
elastic_grid_search = GridSearchCV(Elastic_md, param_grid, cv=5, scoring='r2', refit=True)

# GridSearchCV를 통한 최적의 모델 fitting 수행하여 모델 생성
elastic_grid_search.fit(x_train,y_train)

# model evaluation
# 교차 검증과 그리드 서치의 결과로 산정한 최적의 매개변수를 적용하여 전체 훈련 데이터 셋에 대해
# 훈련한 최종 모델에 테스트 데이터로 적용했을때 결과
print('test set score : {}'.format(elastic_grid_search.score(x_test,y_test)))
# test set score : 0.9999999133082321


# best parameter and best score
# 훈련 세트에서 수행한 교차검증에 대한 최적의 파라미터
print("best parameter : {}".format(elastic_grid_search.best_params_))
# best parameter : {'alpha': 0.01, 'fit_intercept': True, 'l1_ratio': 0.99, 'normalize': False}

# 훈련 세트에서 수행한 교차검증의 평균 정확도의 score
print("best score : {}".format(elastic_grid_search.best_score_))
# best score : 0.9999999128296342

# 4) PassiveAggressive
param_grid = {
               "C": np_list_arange(0, 10, 0.001, inclusive=True),
               "fit_intercept": [True, False],
               "loss": ["epsilon_insensitive", "squared_epsilon_insensitive"],
               "epsilon": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
               "shuffle": [True, False]
              }

# 모델 선언
par_md = linear_model.PassiveAggressiveRegressor()

# GridSearchCV 선언
# 활용 모델 PassiveAggressive, 하이퍼파라미터 그리드, Cross-Validation의 K-fold의 K값은 5
# 튜닝성능을 비교하는 score 지료는 r2-score를 사용
# 더 높은 score의 모형이 만들어 질때 최적의 모델로 업데이트하기 위하여 refit=True 선언
par_grid_search = GridSearchCV(par_md, param_grid, cv=5, scoring='r2', refit=True)

# GridSearchCV를 통한 최적의 모델 fitting 수행하여 모델 생성
par_grid_search.fit(x_train,y_train)

# model evaluation
# 교차 검증과 그리드 서치의 결과로 산정한 최적의 매개변수를 적용하여 전체 훈련 데이터 셋에 대해
# 훈련한 최종 모델에 테스트 데이터로 적용했을때 결과
print('test set score : {}'.format(par_grid_search.score(x_test,y_test)))
# test set score : 0.9999999240911739


# best parameter and best score
# 훈련 세트에서 수행한 교차검증에 대한 최적의 파라미터
print("best parameter : {}".format(par_grid_search.best_params_))
# best parameter : {'C': 7.251, 'epsilon': 0.1, 'fit_intercept': True, 'loss': 'epsilon_insensitive', 'shuffle': True}

# 훈련 세트에서 수행한 교차검증의 평균 정확도의 score
print("best score : {}".format(par_grid_search.best_score_))
# best score : 0.9999999399053581


# 5) HuberRegressor
param_grid = {
             "epsilon": [1, 1.1, 1.2, 1.3, 1.35, 1.4, 1.5, 1.55, 1.6, 1.7, 1.8, 1.9],
             "alpha": [0.0000001,0.000001,0.0001,0.001,0.01,0.0005,0.005,0.05,0.1,0.15,0.2,0.3,0.4,0.5,0.7,0.9],
            }

# 모델 선언
huber_md = linear_model.HuberRegressor()

# GridSearchCV 선언
# 활용 모델 HuberRegressor, 하이퍼파라미터 그리드, Cross-Validation의 K-fold의 K값은 5
# 튜닝성능을 비교하는 score 지료는 r2-score를 사용
# 더 높은 score의 모형이 만들어 질때 최적의 모델로 업데이트하기 위하여 refit=True 선언
huber_grid_search = GridSearchCV(huber_md, param_grid, cv=5, scoring='r2', refit=True)

# GridSearchCV를 통한 최적의 모델 fitting 수행하여 모델 생성
huber_grid_search.fit(x_train,y_train)

# model evaluation
# 교차 검증과 그리드 서치의 결과로 산정한 최적의 매개변수를 적용하여 전체 훈련 데이터 셋에 대해
# 훈련한 최종 모델에 테스트 데이터로 적용했을때 결과
print('test set score : {}'.format(huber_grid_search.score(x_test,y_test)))



# best parameter and best score
# 훈련 세트에서 수행한 교차검증에 대한 최적의 파라미터
print("best parameter : {}".format(huber_grid_search.best_params_))


# 훈련 세트에서 수행한 교차검증의 평균 정확도의 score
print("best score : {}".format(huber_grid_search.best_score_))



# 6) RANSACRegressor
param_grid ={
            "min_samples": np_list_arange(0, 1, 0.05, inclusive=True),
            "max_trials": np_list_arange(1, 20, 1, inclusive=True),
            "max_skips": np_list_arange(1, 20, 1, inclusive=True),
            "stop_n_inliers": np_list_arange(1, 25, 1, inclusive=True),
            "stop_probability": np_list_arange(0, 1, 0.01, inclusive=True),
            "loss": ["absolute_loss", "squared_loss"]
             }

# 모델 선언
ransacr_md = linear_model.RANSACRegressor()

# GridSearchCV 선언
# 활용 모델 RANSACR, 하이퍼파라미터 그리드, Cross-Validation의 K-fold의 K값은 5
# 튜닝성능을 비교하는 score 지료는 r2-score를 사용
# 더 높은 score의 모형이 만들어 질때 최적의 모델로 업데이트하기 위하여 refit=True 선언
rannsacr_grid_search = GridSearchCV(ransacr_md, param_grid, cv=5, scoring='r2', refit=True)

# GridSearchCV를 통한 최적의 모델 fitting 수행하여 모델 생성
rannsacr_grid_search.fit(x_train,y_train)

# model evaluation
# 교차 검증과 그리드 서치의 결과로 산정한 최적의 매개변수를 적용하여 전체 훈련 데이터 셋에 대해
# 훈련한 최종 모델에 테스트 데이터로 적용했을때 결과
print('test set score : {}'.format(rannsacr_grid_search.score(x_test,y_test)))



# best parameter and best score
# 훈련 세트에서 수행한 교차검증에 대한 최적의 파라미터
print("best parameter : {}".format(rannsacr_grid_search.best_params_))


# 훈련 세트에서 수행한 교차검증의 평균 정확도의 score
print("best score : {}".format(rannsacr_grid_search.best_score_))



# 7) DecisionTree Regressor
param_grid = {
              "max_depth": np_list_arange(1, 16, 1, inclusive=True),
              "max_features": [1.0, "sqrt", "log2"],
              "min_samples_leaf": [2, 3, 4, 5, 6],
              "min_samples_split": [2, 5, 7, 9, 10],
              "min_impurity_decrease": [0,0.0001,0.001,0.01,0.0002,0.002,0.02,0.0005,0.005,0.05,0.1,0.2,0.3,0.4,0.5],
              "criterion": ["mse", "mae", "friedman_mse"],
              }
# 모델 선언
dt_reg_md = tree.DecisionTreeRegressor()

# GridSearchCV 선언
# 활용 모델 RANSACR, 하이퍼파라미터 그리드, Cross-Validation의 K-fold의 K값은 5
# 튜닝성능을 비교하는 score 지료는 r2-score를 사용
# 더 높은 score의 모형이 만들어 질때 최적의 모델로 업데이트하기 위하여 refit=True 선언
dt_reg_grid_search = GridSearchCV(dt_reg_md, param_grid, cv=5, scoring='r2', refit=True)

# GridSearchCV를 통한 최적의 모델 fitting 수행하여 모델 생성
dt_reg_grid_search.fit(x_train,y_train)

# model evaluation
# 교차 검증과 그리드 서치의 결과로 산정한 최적의 매개변수를 적용하여 전체 훈련 데이터 셋에 대해
# 훈련한 최종 모델에 테스트 데이터로 적용했을때 결과
print('test set score : {}'.format(dt_reg_grid_search.score(x_test,y_test)))

# best parameter and best score
# 훈련 세트에서 수행한 교차검증에 대한 최적의 파라미터
print("best parameter : {}".format(dt_reg_grid_search.best_params_))


# 훈련 세트에서 수행한 교차검증의 평균 정확도의 score
print("best score : {}".format(dt_reg_grid_search.best_score_))

# 8) RandomForestRegressor
param_grid = {
               "n_estimators": np_list_arange(10, 300, 10, inclusive=True),
               "max_depth": np_list_arange(1, 11, 1, inclusive=True),
               "min_impurity_decrease": [0,0.0001,0.001,0.01,0.0002,0.002,0.02,0.0005,0.005,0.05,0.1,0.2,0.3,0.4,0.5],
               "max_features": [1.0, "sqrt", "log2"],
               "bootstrap": [True, False],
              }
# 모델 선언
rf_reg_md = ensemble.RandomForestRegressor()

# GridSearchCV 선언
# 활용 모델 RandomForestRegressor, 하이퍼파라미터 그리드, Cross-Validation의 K-fold의 K값은 5
# 튜닝성능을 비교하는 score 지료는 r2-score를 사용
# 더 높은 score의 모형이 만들어 질때 최적의 모델로 업데이트하기 위하여 refit=True 선언
rf_reg_grid_search = GridSearchCV(rf_reg_md, param_grid, cv=5, scoring='r2', refit=True)

# GridSearchCV를 통한 최적의 모델 fitting 수행하여 모델 생성
rf_reg_grid_search.fit(x_train,y_train)

# model evaluation
# 교차 검증과 그리드 서치의 결과로 산정한 최적의 매개변수를 적용하여 전체 훈련 데이터 셋에 대해
# 훈련한 최종 모델에 테스트 데이터로 적용했을때 결과
print('test set score : {}'.format(rf_reg_grid_search.score(x_test,y_test)))

# best parameter and best score
# 훈련 세트에서 수행한 교차검증에 대한 최적의 파라미터
print("best parameter : {}".format(rf_reg_grid_search.best_params_))


# 훈련 세트에서 수행한 교차검증의 평균 정확도의 score
print("best score : {}".format(dt_reg_grid_search.best_score_))

# 9) ExtraTreesRegressor
param_grid = {
            "n_estimators": np_list_arange(10, 300, 10, inclusive=True),
            "criterion": ["mse", "mae"],
            "max_depth": np_list_arange(1, 11, 1, inclusive=True),
            "min_impurity_decrease": [0,0.0001,0.001,0.01,0.0002,0.002,0.02,0.0005,0.005,0.05,0.1,0.2,0.3,0.4,0.5],
            "max_features": [1.0, "sqrt", "log2"],
            "bootstrap": [True, False],
            "min_samples_split": [2, 5, 7, 9, 10],
            "min_samples_leaf": [2, 3, 4, 5, 6],
            }

# 모델 선언
extr_reg_md = ensemble.ExtraTreesRegressor()

# GridSearchCV 선언
# 활용 모델 ExtraTreesRegressor, 하이퍼파라미터 그리드, Cross-Validation의 K-fold의 K값은 5
# 튜닝성능을 비교하는 score 지료는 r2-score를 사용
# 더 높은 score의 모형이 만들어 질때 최적의 모델로 업데이트하기 위하여 refit=True 선언
ex_tr_reg_grid_search = GridSearchCV(extr_reg_md, param_grid, cv=5, scoring='r2', refit=True)

# GridSearchCV를 통한 최적의 모델 fitting 수행하여 모델 생성
ex_tr_reg_grid_search.fit(x_train,y_train)

# model evaluation
# 교차 검증과 그리드 서치의 결과로 산정한 최적의 매개변수를 적용하여 전체 훈련 데이터 셋에 대해
# 훈련한 최종 모델에 테스트 데이터로 적용했을때 결과
print('test set score : {}'.format(ex_tr_reg_grid_search.score(x_test,y_test)))

# best parameter and best score
# 훈련 세트에서 수행한 교차검증에 대한 최적의 파라미터
print("best parameter : {}".format(ex_tr_reg_grid_search.best_params_))


# 훈련 세트에서 수행한 교차검증의 평균 정확도의 score
print("best score : {}".format(ex_tr_reg_grid_search.best_score_))



# 10) GradientBoostingRegressor
param_grid = {
              "n_estimators": np_list_arange(10, 300, 10, inclusive=True),
              "learning_rate": np_list_arange(0.001, 0.5, 0.001, inclusive=True),
              "subsample": np_list_arange(0.2, 1, 0.05, inclusive=True),
              "min_samples_split": [2, 4, 5, 7, 9, 10],
              "min_samples_leaf": [1, 2, 3, 4, 5],
              "max_depth": np_list_arange(1, 11, 1, inclusive=True),
              "min_impurity_decrease": [0,0.0001,0.001,0.01,0.0002,0.002,0.02,0.0005,0.005,0.05,0.1,0.2,0.3,0.4,0.5],
              "max_features": [1.0, "sqrt", "log2"],
               }

# 모델 선언
gbr_reg_md = ensemble.GradientBoostingRegressor()

# GridSearchCV 선언
# 활용 모델 GradientBoostingRegressor, 하이퍼파라미터 그리드, Cross-Validation의 K-fold의 K값은 5
# 튜닝성능을 비교하는 score 지료는 r2-score를 사용
# 더 높은 score의 모형이 만들어 질때 최적의 모델로 업데이트하기 위하여 refit=True 선언
gbr_reg_grid_search = GridSearchCV(gbr_reg_md, param_grid, cv=5, scoring='r2', refit=True)

# GridSearchCV를 통한 최적의 모델 fitting 수행하여 모델 생성
gbr_reg_grid_search.fit(x_train,y_train)

# model evaluation
# 교차 검증과 그리드 서치의 결과로 산정한 최적의 매개변수를 적용하여 전체 훈련 데이터 셋에 대해
# 훈련한 최종 모델에 테스트 데이터로 적용했을때 결과
print('test set score : {}'.format(gbr_reg_grid_search.score(x_test,y_test)))

# best parameter and best score
# 훈련 세트에서 수행한 교차검증에 대한 최적의 파라미터
print("best parameter : {}".format(gbr_reg_grid_search.best_params_))


# 훈련 세트에서 수행한 교차검증의 평균 정확도의 score
print("best score : {}".format(gbr_reg_grid_search.best_score_))


# 11) MLPRegressor
param_grid = {
            "learning_rate": ["constant", "invscaling", "adaptive"],
            "alpha": [0.0000001,0.000001,0.0001,0.001,0.01,0.0005,0.005,0.05,0.1,0.15,0.2,0.3,0.4,0.5,0.7,0.9],
            "hidden_layer_size_0": [50, 100],
            "hidden_layer_size_1": [0, 50, 100],
            "hidden_layer_size_2": [0, 50, 100],
            "activation": ["tanh", "identity", "logistic", "relu"],
            }

# 모델 선언
mlp_reg_md = neural_network.MLPRegressor()

# GridSearchCV 선언
# 활용 모델 MLPRegressor, 하이퍼파라미터 그리드, Cross-Validation의 K-fold의 K값은 5
# 튜닝성능을 비교하는 score 지료는 r2-score를 사용
# 더 높은 score의 모형이 만들어 질때 최적의 모델로 업데이트하기 위하여 refit=True 선언
mlp_reg_grid_search = GridSearchCV(mlp_reg_md, param_grid, cv=5, scoring='r2', refit=True)

# GridSearchCV를 통한 최적의 모델 fitting 수행하여 모델 생성
mlp_reg_grid_search.fit(x_train,y_train)

# model evaluation
# 교차 검증과 그리드 서치의 결과로 산정한 최적의 매개변수를 적용하여 전체 훈련 데이터 셋에 대해
# 훈련한 최종 모델에 테스트 데이터로 적용했을때 결과
print('test set score : {}'.format(mlp_reg_grid_search.score(x_test,y_test)))

# best parameter and best score
# 훈련 세트에서 수행한 교차검증에 대한 최적의 파라미터
print("best parameter : {}".format(mlp_reg_grid_search.best_params_))


# 훈련 세트에서 수행한 교차검증의 평균 정확도의 score
print("best score : {}".format(mlp_reg_grid_search.best_score_))


# 12) XGBRegressor

param_grid = {
              "learning_rate": np_list_arange(0.001, 0.5, 0.001, inclusive=True),
              "n_estimators": np_list_arange(10, 300, 10, inclusive=True),
              "subsample": [0.2, 0.3, 0.5, 0.7, 0.9, 1],
              "max_depth": np_list_arange(1, 11, 1, inclusive=True),
              "colsample_bytree": [0.5, 0.7, 0.9, 1],
              "min_child_weight": [1, 2, 3, 4],
              "reg_alpha": [0.0000001,0.000001,0.0001,0.001,0.01,0.0005,0.005,0.05,0.1,0.15,0.2,0.3,0.4,0.5,0.7,1,2,3,4,5,10],
              "reg_lambda": [0.0000001,0.000001,0.0001,0.001,0.01,0.0005,0.005,0.05,0.1,0.15,0.2,0.3,0.4,0.5,0.7,1,2,3,4,5,10],
              "scale_pos_weight": np_list_arange(0, 50, 0.1, inclusive=True),
             }


# 모델 선언
xgb_reg_md = xgboost.XGBRegressor()

# GridSearchCV 선언
# 활용 모델 XGBRegressor, 하이퍼파라미터 그리드, Cross-Validation의 K-fold의 K값은 5
# 튜닝성능을 비교하는 score 지료는 r2-score를 사용
# 더 높은 score의 모형이 만들어 질때 최적의 모델로 업데이트하기 위하여 refit=True 선언
xgb_reg_grid_search = GridSearchCV(xgb_reg_md, param_grid, cv=5, scoring='r2', refit=True)

# GridSearchCV를 통한 최적의 모델 fitting 수행하여 모델 생성
xgb_reg_grid_search.fit(x_train,y_train)

# model evaluation
# 교차 검증과 그리드 서치의 결과로 산정한 최적의 매개변수를 적용하여 전체 훈련 데이터 셋에 대해
# 훈련한 최종 모델에 테스트 데이터로 적용했을때 결과
print('test set score : {}'.format(xgb_reg_grid_search.score(x_test,y_test)))

# best parameter and best score
# 훈련 세트에서 수행한 교차검증에 대한 최적의 파라미터
print("best parameter : {}".format(xgb_reg_grid_search.best_params_))


# 훈련 세트에서 수행한 교차검증의 평균 정확도의 score
print("best score : {}".format(xgb_reg_grid_search.best_score_))

# 13) AdaBoostRegressor
param_grid={
            "n_estimators": np_list_arange(10, 300, 10, inclusive=True),
            "learning_rate": np_list_arange(0.001, 0.5, 0.001, inclusive=True),
            "loss": ["linear", "square", "exponential"],
            }

# 모델 선언
ada_reg_md = ensemble.AdaBoostRegressor()

# GridSearchCV 선언
# 활용 모델 AdaBoostRegressor, 하이퍼파라미터 그리드, Cross-Validation의 K-fold의 K값은 5
# 튜닝성능을 비교하는 score 지료는 r2-score를 사용
# 더 높은 score의 모형이 만들어 질때 최적의 모델로 업데이트하기 위하여 refit=True 선언
ada_reg_grid_search = GridSearchCV(ada_reg_md, param_grid, cv=5, scoring='r2', refit=True)

# GridSearchCV를 통한 최적의 모델 fitting 수행하여 모델 생성
ada_reg_grid_search.fit(x_train,y_train)

# model evaluation
# 교차 검증과 그리드 서치의 결과로 산정한 최적의 매개변수를 적용하여 전체 훈련 데이터 셋에 대해
# 훈련한 최종 모델에 테스트 데이터로 적용했을때 결과
print('test set score : {}'.format(ada_reg_grid_search.score(x_test,y_test)))

# best parameter and best score
# 훈련 세트에서 수행한 교차검증에 대한 최적의 파라미터
print("best parameter : {}".format(ada_reg_grid_search.best_params_))


# 훈련 세트에서 수행한 교차검증의 평균 정확도의 score
print("best score : {}".format(ada_reg_grid_search.best_score_))


# 14) BaggingRegressor

param_grid= {
             "bootstrap": [True, False],
             "bootstrap_features": [True, False],
             "max_features": np_list_arange(0.4, 1, 0.1, inclusive=True),
             "max_samples": np_list_arange(0.4, 1, 0.1, inclusive=True),
            }

# 모델 선언
bag_reg_md = ensemble.BaggingRegressor()

# GridSearchCV 선언
# 활용 모델 BaggingRegressor, 하이퍼파라미터 그리드, Cross-Validation의 K-fold의 K값은 5
# 튜닝성능을 비교하는 score 지료는 r2-score를 사용
# 더 높은 score의 모형이 만들어 질때 최적의 모델로 업데이트하기 위하여 refit=True 선언
bag_reg_grid_search = GridSearchCV(bag_reg_md, param_grid, cv=5, scoring='r2', refit=True)

# GridSearchCV를 통한 최적의 모델 fitting 수행하여 모델 생성
bag_reg_grid_search.fit(x_train,y_train)

# model evaluation
# 교차 검증과 그리드 서치의 결과로 산정한 최적의 매개변수를 적용하여 전체 훈련 데이터 셋에 대해
# 훈련한 최종 모델에 테스트 데이터로 적용했을때 결과
print('test set score : {}'.format(bag_reg_grid_search.score(x_test,y_test)))

# best parameter and best score
# 훈련 세트에서 수행한 교차검증에 대한 최적의 파라미터
print("best parameter : {}".format(bag_reg_grid_search.best_params_))

# 훈련 세트에서 수행한 교차검증의 평균 정확도의 score
print("best score : {}".format(ada_reg_grid_search.best_score_))





