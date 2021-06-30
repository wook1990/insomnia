from sklearn import linear_model
from sklearn import linear_model
from sklearn import tree
from sklearn import ensemble
from sklearn import neural_network
import xgboost
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
import numpy as np
from sklearn.datasets import make_classification
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

X,y = make_classification(n_classes=2, n_features=100, n_samples=1000, random_state=1, shuffle=True)
# 학습 및 검증 데이터 셋 분할
x_train, x_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=0)

# Classification
# 1) LogisticRegressorClassifier
param_grid ={
            "C": np_list_arange(0,10,0.001, inclusive=True),
            "penalty":["l2","none"],
}

# 모델 선언
log_md = linear_model.LogisticRegression()

# 활용 모델 LogisticRegressorClassifier, 하이퍼파라미터 그리드, Cross-Validation의 K-fold의 K값은 5
# 튜닝성능을 비교하는 score 지료는 r2-score를 사용
# 더 높은 score의 모형이 만들어 질때 최적의 모델로 업데이트하기 위하여 refit=True 선언
log_grid_search = GridSearchCV(log_md, param_grid, cv=5, scoring='r2', refit=True)

# GridSearchCV를 통한 최적의 모델 fitting 수행하여 모델 생성
log_grid_search.fit(x_train,y_train)

# model evaluation
# 교차 검증과 그리드 서치의 결과로 산정한 최적의 매개변수를 적용하여 전체 훈련 데이터 셋에 대해
# 훈련한 최종 모델에 테스트 데이터로 적용했을때 결과
print('test set score : {}'.format(log_grid_search.score(x_test,y_test)))


# best parameter and best score
# 훈련 세트에서 수행한 교차검증에 대한 최적의 파라미터
print("best parameter : {}".format(log_grid_search.best_params_))

# 훈련 세트에서 수행한 교차검증의 평균 정확도의 score
print("best score : {}".format(log_grid_search.best_score_))

# 2) DecisionTreeClassifier
param_grid = {
             "max_depth": np_list_arange(1, 16, 1, inclusive=True),
             "max_features": [1.0, "sqrt", "log2"],
             "min_samples_leaf": [2, 3, 4, 5, 6],
             "min_samples_split": [2, 5, 7, 9, 10],
             "criterion": ["gini", "entropy"],
             "min_impurity_decrease": [0,0.0001,0.001,0.01,0.0002,0.002,0.02,0.0005,0.005,0.05,0.1,0.2,0.3,0.4,0.5]
             }

# 모델 선언
dt_clf_md = tree.DecisionTreeClassifier()

# 활용 모델 DecisionTreeClassifier, 하이퍼파라미터 그리드, Cross-Validation의 K-fold의 K값은 5
# 튜닝성능을 비교하는 score 지료는 r2-score를 사용
# 더 높은 score의 모형이 만들어 질때 최적의 모델로 업데이트하기 위하여 refit=True 선언
dt_clf_grid_search = GridSearchCV(dt_clf_md, param_grid, cv=5, scoring='r2', refit=True)

# GridSearchCV를 통한 최적의 모델 fitting 수행하여 모델 생성
dt_clf_grid_search.fit(x_train,y_train)

# model evaluation
# 교차 검증과 그리드 서치의 결과로 산정한 최적의 매개변수를 적용하여 전체 훈련 데이터 셋에 대해
# 훈련한 최종 모델에 테스트 데이터로 적용했을때 결과
print('test set score : {}'.format(dt_clf_grid_search.score(x_test,y_test)))


# best parameter and best score
# 훈련 세트에서 수행한 교차검증에 대한 최적의 파라미터
print("best parameter : {}".format(dt_clf_grid_search.best_params_))

# 훈련 세트에서 수행한 교차검증의 평균 정확도의 score
print("best score : {}".format(dt_clf_grid_search.best_score_))

# 3) SGDClassifier
param_grid= {
             "penalty": ["elasticnet", "l2", "l1"],
             "l1_ratio": np_list_arange(0.0000000001, 1, 0.01, inclusive=False),
             "alpha": [0.0000001,0.000001,0.0001,0.001,0.01,0.0002,0.002,0.02,0.0005,0.005,0.05,0.1,0.15,0.2,0.3,0.4,0.5],
             "fit_intercept": [True, False],
             "learning_rate": ["constant", "invscaling", "adaptive", "optimal"],
             "eta0": [0.001, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5],
            }

# 모델 선언
sgd_clf_md = linear_model.SGDClassifier()
# 활용 모델 DecisionTreeClassifier, 하이퍼파라미터 그리드, Cross-Validation의 K-fold의 K값은 5
# 튜닝성능을 비교하는 score 지료는 r2-score를 사용
# 더 높은 score의 모형이 만들어 질때 최적의 모델로 업데이트하기 위하여 refit=True 선언
sgd_clf_grid_search = GridSearchCV(sgd_clf_md, param_grid, cv=5, scoring='r2', refit=True)

# GridSearchCV를 통한 최적의 모델 fitting 수행하여 모델 생성
sgd_clf_grid_search.fit(x_train,y_train)

# model evaluation
# 교차 검증과 그리드 서치의 결과로 산정한 최적의 매개변수를 적용하여 전체 훈련 데이터 셋에 대해
# 훈련한 최종 모델에 테스트 데이터로 적용했을때 결과
print('test set score : {}'.format(sgd_clf_grid_search.score(x_test,y_test)))


# best parameter and best score
# 훈련 세트에서 수행한 교차검증에 대한 최적의 파라미터
print("best parameter : {}".format(sgd_clf_grid_search.best_params_))

# 훈련 세트에서 수행한 교차검증의 평균 정확도의 score
print("best score : {}".format(sgd_clf_grid_search.best_score_))


# 4) MLPClassifier
param_grid = {
            "learning_rate": ["constant", "invscaling", "adaptive"],
            "alpha": [0.0000001,0.000001,0.0001,0.001,0.01,0.0005,0.005,0.05,0.1,0.15,0.2,0.3,0.4,0.5,0.7,0.9],
            "hidden_layer_size_0": [50, 100],
            "hidden_layer_size_1": [0, 50, 100],
            "hidden_layer_size_2": [0, 50, 100],
            "activation": ["tanh", "identity", "logistic", "relu"],
            }

# 모델 선언
mlp_clf_md = neural_network.MLPClassifier()

# 활용 모델 DecisionTreeClassifier, 하이퍼파라미터 그리드, Cross-Validation의 K-fold의 K값은 5
# 튜닝성능을 비교하는 score 지료는 r2-score를 사용
# 더 높은 score의 모형이 만들어 질때 최적의 모델로 업데이트하기 위하여 refit=True 선언
mlp_clf_grid_search = GridSearchCV(mlp_clf_md, param_grid, cv=5, scoring='r2', refit=True)

# GridSearchCV를 통한 최적의 모델 fitting 수행하여 모델 생성
mlp_clf_grid_search.fit(x_train,y_train)

# model evaluation
# 교차 검증과 그리드 서치의 결과로 산정한 최적의 매개변수를 적용하여 전체 훈련 데이터 셋에 대해
# 훈련한 최종 모델에 테스트 데이터로 적용했을때 결과
print('test set score : {}'.format(mlp_clf_grid_search.score(x_test,y_test)))

# best parameter and best score
# 훈련 세트에서 수행한 교차검증에 대한 최적의 파라미터
print("best parameter : {}".format(mlp_clf_grid_search.best_params_))

# 훈련 세트에서 수행한 교차검증의 평균 정확도의 score
print("best score : {}".format(mlp_clf_grid_search.best_score_))


# 5) RidgeClassifier
param_grid={
            "alpha":np_list_arange(0.01, 10, 0.01, inclusive=False),
            "fit_intercept":[True,False]
            }

# 모델 선언
rig_clf_md = linear_model.RidgeClassifier()

# 활용 모델 RidgeClassifier, 하이퍼파라미터 그리드, Cross-Validation의 K-fold의 K값은 5
# 튜닝성능을 비교하는 score 지료는 r2-score를 사용
# 더 높은 score의 모형이 만들어 질때 최적의 모델로 업데이트하기 위하여 refit=True 선언
rig_clf_grid_search = GridSearchCV(rig_clf_md, param_grid, cv=5, scoring='r2', refit=True)

# GridSearchCV를 통한 최적의 모델 fitting 수행하여 모델 생성
rig_clf_grid_search.fit(x_train,y_train)

# model evaluation
# 교차 검증과 그리드 서치의 결과로 산정한 최적의 매개변수를 적용하여 전체 훈련 데이터 셋에 대해
# 훈련한 최종 모델에 테스트 데이터로 적용했을때 결과
print('test set score : {}'.format(rig_clf_grid_search.score(x_test,y_test)))

# best parameter and best score
# 훈련 세트에서 수행한 교차검증에 대한 최적의 파라미터
print("best parameter : {}".format(rig_clf_grid_search.best_params_))

# 훈련 세트에서 수행한 교차검증의 평균 정확도의 score
print("best score : {}".format(rig_clf_grid_search.best_score_))

# 6) RandomForestClassifier
param_grid = {
            "n_estimators": np_list_arange(10, 300, 10, inclusive=True),
            "criterion": ["gini", "entropy"],
            "max_depth": np_list_arange(1, 11, 1, inclusive=True),
            "min_samples_split": [2, 5, 7, 9, 10],
            "min_samples_leaf":  [2, 3, 4, 5, 6],
            "min_impurity_decrease": [0,0.0001,0.001,0.01,0.0002,0.002,0.02,0.0005,0.005,0.05,0.1,0.2,0.3,0.4,0.5],
            "max_features": [1.0, "sqrt", "log2"],
            "bootstrap": [True, False]
            }

# 모델 선언
rf_clf_md = ensemble.RandomForestClassifier()

# 활용 모델 RandomForestClassifier, 하이퍼파라미터 그리드, Cross-Validation의 K-fold의 K값은 5
# 튜닝성능을 비교하는 score 지료는 r2-score를 사용
# 더 높은 score의 모형이 만들어 질때 최적의 모델로 업데이트하기 위하여 refit=True 선언
rf_clf_grid_search = GridSearchCV(rf_clf_md, param_grid, cv=5, scoring='r2', refit=True)

# GridSearchCV를 통한 최적의 모델 fitting 수행하여 모델 생성
rf_clf_grid_search.fit(x_train,y_train)

# model evaluation
# 교차 검증과 그리드 서치의 결과로 산정한 최적의 매개변수를 적용하여 전체 훈련 데이터 셋에 대해
# 훈련한 최종 모델에 테스트 데이터로 적용했을때 결과
print('test set score : {}'.format(rf_clf_grid_search.score(x_test,y_test)))

# best parameter and best score
# 훈련 세트에서 수행한 교차검증에 대한 최적의 파라미터
print("best parameter : {}".format(rf_clf_grid_search.best_params_))

# 훈련 세트에서 수행한 교차검증의 평균 정확도의 score
print("best score : {}".format(rf_clf_grid_search.best_score_))


# 7) AdaBoostClassifier
param_grid = {
              "n_estimators": np_list_arange(10, 300, 10, inclusive=True),
              "learning_rate": np_list_arange(0.001, 0.5, 0.001, inclusive=True),
              "algorithm": ["SAMME", "SAMME.R"],
             }


# 모델 선언
ada_clf_md = ensemble.AdaBoostClassifier()

# 활용 모델 AdaBoostClassifier, 하이퍼파라미터 그리드, Cross-Validation의 K-fold의 K값은 5
# 튜닝성능을 비교하는 score 지료는 r2-score를 사용
# 더 높은 score의 모형이 만들어 질때 최적의 모델로 업데이트하기 위하여 refit=True 선언
ada_clf_grid_search = GridSearchCV(ada_clf_md, param_grid, cv=5, scoring='r2', refit=True)

# GridSearchCV를 통한 최적의 모델 fitting 수행하여 모델 생성
ada_clf_grid_search.fit(x_train,y_train)

# model evaluation
# 교차 검증과 그리드 서치의 결과로 산정한 최적의 매개변수를 적용하여 전체 훈련 데이터 셋에 대해
# 훈련한 최종 모델에 테스트 데이터로 적용했을때 결과
print('test set score : {}'.format(ada_clf_grid_search.score(x_test,y_test)))

# best parameter and best score
# 훈련 세트에서 수행한 교차검증에 대한 최적의 파라미터
print("best parameter : {}".format(ada_clf_grid_search.best_params_))

# 훈련 세트에서 수행한 교차검증의 평균 정확도의 score
print("best score : {}".format(ada_clf_grid_search.best_score_))


# 8) GradientBoostingClassifier
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
gb_clf_md = ensemble.GradientBoostingClassifier()

# 활용 모델 GradientBoostingClassifier, 하이퍼파라미터 그리드, Cross-Validation의 K-fold의 K값은 5
# 튜닝성능을 비교하는 score 지료는 r2-score를 사용
# 더 높은 score의 모형이 만들어 질때 최적의 모델로 업데이트하기 위하여 refit=True 선언
gb_clf_grid_search = GridSearchCV(gb_clf_md, param_grid, cv=5, scoring='r2', refit=True)

# GridSearchCV를 통한 최적의 모델 fitting 수행하여 모델 생성
gb_clf_grid_search.fit(x_train,y_train)

# model evaluation
# 교차 검증과 그리드 서치의 결과로 산정한 최적의 매개변수를 적용하여 전체 훈련 데이터 셋에 대해
# 훈련한 최종 모델에 테스트 데이터로 적용했을때 결과
print('test set score : {}'.format(gb_clf_grid_search.score(x_test,y_test)))

# best parameter and best score
# 훈련 세트에서 수행한 교차검증에 대한 최적의 파라미터
print("best parameter : {}".format(gb_clf_grid_search.best_params_))

# 훈련 세트에서 수행한 교차검증의 평균 정확도의 score
print("best score : {}".format(gb_clf_grid_search.best_score_))

# 9) ExtraTreesClassifier
param_grid = {
             "n_estimators": np_list_arange(10, 300, 10, inclusive=True),
             "criterion": ["gini", "entropy"],
             "max_depth": np_list_arange(1, 11, 1, inclusive=True),
             "min_impurity_decrease": [0,0.0001,0.001,0.01,0.0002,0.002,0.02,0.0005,0.005,0.05,0.1,0.2,0.3,0.4,0.5],
             "max_features": [1.0, "sqrt", "log2"],
             "bootstrap": [True, False],
             "min_samples_split": [2, 5, 7, 9, 10],
             "min_samples_leaf": [2, 3, 4, 5, 6]
             }

# 모델 선언
ext_clf_md = ensemble.ExtraTreesClassifier()

# 활용 모델 ExtraTreesClassifier, 하이퍼파라미터 그리드, Cross-Validation의 K-fold의 K값은 5
# 튜닝성능을 비교하는 score 지료는 r2-score를 사용
# 더 높은 score의 모형이 만들어 질때 최적의 모델로 업데이트하기 위하여 refit=True 선언
ext_clf_grid_search = GridSearchCV(ext_clf_md, param_grid, cv=5, scoring='r2', refit=True)

# GridSearchCV를 통한 최적의 모델 fitting 수행하여 모델 생성
ext_clf_grid_search.fit(x_train,y_train)

# model evaluation
# 교차 검증과 그리드 서치의 결과로 산정한 최적의 매개변수를 적용하여 전체 훈련 데이터 셋에 대해
# 훈련한 최종 모델에 테스트 데이터로 적용했을때 결과
print('test set score : {}'.format(ext_clf_grid_search.score(x_test,y_test)))

# best parameter and best score
# 훈련 세트에서 수행한 교차검증에 대한 최적의 파라미터
print("best parameter : {}".format(ext_clf_grid_search.best_params_))

# 훈련 세트에서 수행한 교차검증의 평균 정확도의 score
print("best score : {}".format(ext_clf_grid_search.best_score_))


# 10) XGBClassifier
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
xgb_clf_md = xgboost.XGBClassifier()
# 활용 모델 ExtraTreesClassifier, 하이퍼파라미터 그리드, Cross-Validation의 K-fold의 K값은 5
# 튜닝성능을 비교하는 score 지료는 r2-score를 사용
# 더 높은 score의 모형이 만들어 질때 최적의 모델로 업데이트하기 위하여 refit=True 선언
xgb_clf_grid_search = GridSearchCV(xgb_clf_md, param_grid, cv=5, scoring='r2', refit=True)

# GridSearchCV를 통한 최적의 모델 fitting 수행하여 모델 생성
xgb_clf_grid_search.fit(x_train,y_train)

# model evaluation
# 교차 검증과 그리드 서치의 결과로 산정한 최적의 매개변수를 적용하여 전체 훈련 데이터 셋에 대해
# 훈련한 최종 모델에 테스트 데이터로 적용했을때 결과
print('test set score : {}'.format(xgb_clf_grid_search.score(x_test,y_test)))

# best parameter and best score
# 훈련 세트에서 수행한 교차검증에 대한 최적의 파라미터
print("best parameter : {}".format(xgb_clf_grid_search.best_params_))

# 훈련 세트에서 수행한 교차검증의 평균 정확도의 score
print("best score : {}".format(xgb_clf_grid_search.best_score_))


# 12) BaggingClassifier
param_grid = {
             "bootstrap": [True, False],
             "bootstrap_features": [True, False],
             "max_features": np_list_arange(0.4, 1, 0.1, inclusive=True),
             "max_samples": np_list_arange(0.4, 1, 0.1, inclusive=True),
             }

# 모델 선언
bag_clf_md = ensemble.BaggingClassifier()
# 활용 모델 BaggingClassifier, 하이퍼파라미터 그리드, Cross-Validation의 K-fold의 K값은 5
# 튜닝성능을 비교하는 score 지료는 r2-score를 사용
# 더 높은 score의 모형이 만들어 질때 최적의 모델로 업데이트하기 위하여 refit=True 선언
bag_clf_grid_search = GridSearchCV(bag_clf_md, param_grid, cv=5, scoring='r2', refit=True)

# GridSearchCV를 통한 최적의 모델 fitting 수행하여 모델 생성
bag_clf_grid_search.fit(x_train,y_train)

# model evaluation
# 교차 검증과 그리드 서치의 결과로 산정한 최적의 매개변수를 적용하여 전체 훈련 데이터 셋에 대해
# 훈련한 최종 모델에 테스트 데이터로 적용했을때 결과
print('test set score : {}'.format(bag_clf_grid_search.score(x_test,y_test)))

# best parameter and best score
# 훈련 세트에서 수행한 교차검증에 대한 최적의 파라미터
print("best parameter : {}".format(bag_clf_grid_search.best_params_))

# 훈련 세트에서 수행한 교차검증의 평균 정확도의 score
print("best score : {}".format(bag_clf_grid_search.best_score_))