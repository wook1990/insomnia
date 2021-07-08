# pip insatll baysian-optimization
# 라이브러리 설치

from sklearn import datasets, svm
from sklearn.model_selection import cross_val_score, train_test_split
from bayes_opt import BayesianOptimization
from functools import partial

# Load Data
iris = datasets.load_iris()

# Data Split
x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=.25, random_state=0)

# Model Define
def svm_rfb_cv(gamma, C):

    model = svm.SVC(kernel= 'rbf', gamma=gamma, C = C)
    train_RMSE = cross_val_score(model, x_train, y_train,scoring='accuracy',cv = 5).mean()
    return train_RMSE

# 하이퍼파라미터 최적화를 위한 파라미터 번위 설
param_grid = {'gamma':(0.001, 1000), "C":(0.001,1000)}
# 최적화 수행
bayes_opt = BayesianOptimization(svm_rfb_cv, pbounds=param_grid, verbose = 2, random_state=1)

# init_points : 처음 랜덤 값은 추출하여 스코어를 계산할 횟수
# n_iter : 랜덤 추출하여 생성된 score를 기준으로 최적화 수행 횟수
bayes_opt.maximize(init_points=0, n_iter=10)
'''
|   iter    |  target   |     C     |   gamma   |
-------------------------------------------------
|  1        |  0.3399   |  70.8     |  815.1    |
|  2        |  0.3763   |  767.9    |  286.4    |
|  3        |  0.3399   |  996.5    |  992.5    |
|  4        |  0.9474   |  995.1    |  0.1442   |
|  5        |  0.5791   |  0.001    |  0.001    |
|  6        |  0.9387   |  997.4    |  0.243    |
|  7        |  0.9474   |  492.2    |  0.001    |
|  8        |  0.3399   |  498.0    |  1e+03    |
|  9        |  0.3308   |  0.001    |  390.5    |
|  10       |  0.9474   |  735.4    |  0.001    |
|  11       |  0.3399   |  385.5    |  486.9    |
|  12       |  0.3399   |  1e+03    |  612.7    |
|  13       |  0.9648   |  263.0    |  0.001    |
|  14       |  0.3308   |  0.5189   |  992.0    |
|  15       |  0.5534   |  328.1    |  152.7    |
|  16       |  0.9296   |  888.6    |  4.071    |
|  17       |  0.3399   |  683.7    |  715.0    |
|  18       |  0.3763   |  998.6    |  270.9    |
|  19       |  0.3399   |  243.1    |  997.1    |
|  20       |  0.93     |  613.3    |  1.106    |
|  21       |  0.9296   |  353.9    |  3.234    |
|  22       |  0.9296   |  177.9    |  4.752    |
=================================================
'''

# 최적의 파라미터 추출
bayes_opt.max
"""
{'target': 0.9727272727272727,
 'params': {'C': 147.53206626730363, 'gamma': 0.024215353253278435}}
"""



# 모델의 scoring 지표를 변화하여 확인
def svm_cv(gamma, C, x_data = None, y_data = None, test_size = 0.25, output = 'score'):

    models = []
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size = test_size, random_state= 4213)
    model = svm.SVC(kernel = 'rbf', gamma = gamma, C = C, probability=True)
    model.fit(x_train, y_train)
    models.append(model)
    score = cross_val_score(model, x_train, y_train, scoring="accuracy", cv =5 ).mean()

    if output == 'score':
        return score
    if output == 'model':
        return models


# 필요없는 상수항의 고정
func_fixed = partial(svm_cv, x_data = iris.data, y_data = iris.target, test_size = 0.25, output = 'score')
# 하이퍼파라미터 최적화 범위 선언
param_grid = {'gamma':(0.001,1000), 'C':(0.001,1000)}
# Bayesian최적화 객체 선언
smvOB = BayesianOptimization(func_fixed, param_grid, random_state= 4321)

# 처음 2회 랜덥 값으로 score 계산후 20회 최적화 수행
smvOB.maximize(init_points=2, n_iter=20)
'''
|   iter    |  target   |     C     |   gamma   |
-------------------------------------------------
|  1        |  0.3399   |  70.8     |  815.1    |
|  2        |  0.3763   |  767.9    |  286.4    |
|  3        |  0.3399   |  996.5    |  992.5    |
|  4        |  0.9474   |  995.1    |  0.1442   |
|  5        |  0.5791   |  0.001    |  0.001    |
|  6        |  0.9387   |  997.4    |  0.243    |
|  7        |  0.9474   |  492.2    |  0.001    |
|  8        |  0.3399   |  498.0    |  1e+03    |
|  9        |  0.3308   |  0.001    |  390.5    |
|  10       |  0.9474   |  735.4    |  0.001    |
|  11       |  0.3399   |  385.5    |  486.9    |
|  12       |  0.3399   |  1e+03    |  612.7    |
|  13       |  0.9648   |  263.0    |  0.001    |
|  14       |  0.3308   |  0.5189   |  992.0    |
|  15       |  0.5534   |  328.1    |  152.7    |
|  16       |  0.9296   |  888.6    |  4.071    |
|  17       |  0.3399   |  683.7    |  715.0    |
|  18       |  0.3763   |  998.6    |  270.9    |
|  19       |  0.3399   |  243.1    |  997.1    |
|  20       |  0.93     |  613.3    |  1.106    |
|  21       |  0.9296   |  353.9    |  3.234    |
|  22       |  0.9296   |  177.9    |  4.752    |
=================================================
'''
# 최적의 파라미터 추출
smvOB.max
'''
{'target': 0.9648221343873518,
 'params': {'C': 262.9587608609658, 'gamma': 0.001}}
'''

'''
scoring에 들어 갈수 있는 평가 메트릭
import sklearn
sklearn.metrics.SCORERS.keys()
dict_keys(['explained_variance', 'r2', 'max_error', 'neg_median_absolute_error', 
           'neg_mean_absolute_error', 'neg_mean_squared_error', 'neg_mean_squared_log_error', 
           'neg_root_mean_squared_error', 'neg_mean_poisson_deviance', 'neg_mean_gamma_deviance', 
           'accuracy', 'roc_auc', 'roc_auc_ovr', 'roc_auc_ovo', 'roc_auc_ovr_weighted', 'roc_auc_ovo_weighted',
           'balanced_accuracy', 'average_precision', 'neg_log_loss', 'neg_brier_score', 'adjusted_rand_score', 
           'homogeneity_score', 'completeness_score', 'v_measure_score', 'mutual_info_score', 'adjusted_mutual_info_score', 
           'normalized_mutual_info_score', 'fowlkes_mallows_score', 'precision', 'precision_macro', 'precision_micro', 
           'precision_samples', 'precision_weighted', 'recall', 'recall_macro', 'recall_micro', 'recall_samples',
           'recall_weighted', 'f1', 'f1_macro', 'f1_micro', 'f1_samples', 'f1_weighted', 'jaccard', 'jaccard_macro',
           'jaccard_micro', 'jaccard_samples', 'jaccard_weighted'])
'''
