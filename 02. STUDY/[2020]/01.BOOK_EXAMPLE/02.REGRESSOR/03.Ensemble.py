from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostRegressor

boston = load_boston()
x_train, x_test, y_train, y_test = train_test_split(boston.data, boston.target, random_state=42)
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

adaboost = AdaBoostRegressor(base_estimator=None,n_estimators=50,
                             learning_rate=1.0, loss='linear', random_state=None)
adaboost.fit(x_train,y_train)
adaboost.score(x_train,y_train)
print("Train R2 score : ", round(adaboost.score(x_train,y_train),3))
adaboost.score(x_test,y_test)
print("Test R2 score : ", round(adaboost.score(x_test,y_test),3))
print("Model Param : \n", adaboost.get_params())
print("Base Estimator :\n", adaboost.base_estimator_)
print("Feautre_importances : \n", adaboost.feature_importances_)




y_hat = adaboost.predict(x_test)

from sklearn.metrics import mean_absolute_error, mean_squared_error
mae = mean_absolute_error(y_test,y_hat)
mse = mean_squared_error(y_test,y_hat)

print("MAE : ",round(mae,3), " MSE : ",round(mse,3))

from sklearn.ensemble import BaggingRegressor

bagging = BaggingRegressor(base_estimator=None, n_estimators=10, max_samples=1.0,
                            max_features=1.0, bootstrap=True, bootstrap_features=False,
                           oob_score=False, warm_start=False, n_jobs=None, random_state=None,
                           verbose=0)
bagging.fit(x_train,y_train)
bagging.score(x_train,y_train)
print("Train R2 score : ", round(bagging.score(x_train,y_train),3))
bagging.score(x_test,y_test)
print("Test R2 score : ", round(bagging.score(x_test,y_test),3))
y_hat = bagging.predict(x_test)
# 선언된 모델의 파라미터 출력
print("Model Param : \n", bagging.get_params())
# 기본 학습 모델 객체
print("Base Estimator : \n", bagging.base_estimator_)
# 각 base estimator의 학습 데이터에 사용된 특성
print("Estimator Features : \n", bagging.estimators_features_)


from sklearn.metrics import mean_absolute_error, mean_squared_error
mae = mean_absolute_error(y_test,y_hat)
mse = mean_squared_error(y_test,y_hat)

print("MAE : ",round(mae,3), " MSE : ",round(mse,3))

from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor

boston = load_boston()
x_train, x_test, y_train, y_test = train_test_split(boston.data, boston.target, random_state=42)
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)


gbm = GradientBoostingRegressor(loss='ls', learning_rate=0.1, n_estimators=100, subsample=1.0,
                                criterion='friedman_mse', min_samples_split=2, min_samples_leaf=1,
                                min_weight_fraction_leaf=0.0, max_depth=3, min_impurity_decrease=0.0,
                                min_impurity_split=None, init=None, random_state=None, max_features=None,
                                alpha=0.9, verbose=0, max_leaf_nodes=None, warm_start=False, presort='deprecated',
                                validation_fraction=0.1, n_iter_no_change=None, tol=0.0001, ccp_alpha=0.0)
# 모델 학습
gbm.fit(x_train,y_train)
# 모델 적합성 검
gbm.score(x_train,y_train)
print("Train R2 Score : ", round(gbm.score(x_train,y_train),3))
# Train R2 Score :  1.0
gbm.score(x_test,y_test)
print("Test R2 Score : ", round(gbm.score(x_test, y_test),3))
y_hat = gbm.predict(x_test)
'''
array([23.04052211, 30.80125144, 16.59687893, 24.18382986, 17.51775094,
       22.20424803, 18.37679613, 13.87826852, 20.69567546, 21.05630329,
       20.65800177, 18.17504716,  7.59628901, 21.70412526, 20.42411344,
       25.68150134, 19.65025197,  9.04186004, 45.76829976, 16.24352807,
       24.16750847, 25.58242866, 13.55637357, 21.63216117, 15.24383795,
       16.02780139, 21.97039232, 14.1308893 , 19.7882457 , 21.4943401 ,
       19.96374277, 23.55638271, 23.44639924, 19.94756007, 14.59751632,
       17.07294658, 33.48430381, 19.44918017, 21.13751246, 23.93977426,
       18.32202324, 30.25253963, 45.28348352, 20.90449728, 22.53997442,
       15.13571919, 16.28600727, 23.74085484, 18.01993116, 27.80166399,
       20.29367355, 35.77815626, 16.5197479 , 25.490704  , 47.51880799,
       21.53764501, 15.99636471, 31.79864176, 21.85748794, 18.29080884,
       22.7379009 , 34.01034737, 30.7125856 , 19.8255971 , 24.76787729,
       18.05612794, 14.58785612, 23.67111194, 28.79028621, 15.10698901,
       21.29168096, 25.25460539, 10.38756407, 20.76810568, 22.63374113,
        5.88198787, 20.52105443, 45.32080046, 12.12777804, 12.34677602,
       21.61277948, 11.8664967 , 18.42404088, 10.4715874 , 20.61979459,
       26.10759046, 15.26338321, 24.13265475, 25.09335989, 17.35813115,
       22.13602129,  9.83535921, 19.4083156 , 18.81635683, 23.01973356,
       19.69572675, 39.10462376, 10.272353  , 12.21727119, 11.48848368,
       20.54377468, 22.8120411 , 13.30665347, 20.00338522, 20.49866238,
       11.85450479, 19.55082284, 27.15203261, 20.18983546, 23.63865688,
        8.5618845 , 14.07088631, 21.85576642, 23.88628685, 33.16769561,
       13.59110845, 42.97428212, 15.53291817, 21.52483935, 23.98481851,
       19.21632547, 24.1684173 ,  6.61056156, 21.20873172, 23.66097026,
       22.9994774 , 22.4328139 ])

'''
#  모델 성능 평가
from sklearn.metrics import mean_absolute_error, mean_squared_error
mae = mean_absolute_error(y_test,y_hat)
mse = mean_squared_error(y_test, y_hat)
print("MAE : ", round(mae,3), " MSE : ", round(mse,3))
# MAE :  2.024  MSE :  8.857
# 특성 변수의 중요도
print("Feature Importance : \n" ,gbm.feature_importances_)
'''
Feature Importance : 
 [2.84289150e-02 8.77238347e-05 1.29980139e-03 6.35463760e-04
 1.28706048e-02 3.98147263e-01 1.57017415e-02 8.09367636e-02
 5.30099306e-03 8.27964647e-03 3.44922746e-02 8.75416701e-03
 4.05064642e-01]
'''
# OOB(out ob bag) 측정 값 : subsample의 값이 1.0 보다 작아야만 계산됨
print("OOB : \n" , gbm.oob_improvement_)
# 학습을 진행하면서 training set의 Loss 값 출력
print("Train Score Each Traing process : \n",gbm.train_score_)
'''
Train Score Each Traing process : 
 [74.74807757 63.38920518 53.95791929 46.35132816 39.87303583 34.40014235
 29.7507313  25.87733328 22.62713565 19.93985039 17.67855546 15.77584454
 14.16527006 12.83661911 11.71101806 10.72969891  9.8830033   9.20096719
  8.57462594  8.05718295  7.60329052  7.22602519  6.9135204   6.61644169
  6.32723847  6.0886056   5.88942927  5.69372489  5.49252597  5.29859737
  5.16594049  5.01130594  4.85230051  4.73383718  4.60605646  4.51928091
  4.39107021  4.30610192  4.1669821   4.08457893  3.95760686  3.88828087
  3.81220617  3.75917263  3.68518343  3.63225185  3.60284945  3.5043056
  3.43855082  3.40299224  3.34730598  3.30498064  3.28205788  3.22216048
  3.16288869  3.07922615  3.03505692  2.99774308  2.97277153  2.95527996
  2.92542637  2.88886756  2.86847221  2.81499988  2.78813112  2.77149545
  2.71291365  2.68583668  2.64925369  2.62729248  2.5906882   2.52229206
  2.5047465   2.47565466  2.45771488  2.43500212  2.42113444  2.40737406
  2.38717641  2.36437394  2.34073154  2.31772145  2.27980542  2.24414371
  2.22935576  2.19731188  2.17643512  2.15589409  2.13721415  2.110981
  2.10145542  2.06146282  2.04610161  2.03084984  2.01156734  1.99764752
  1.99106368  1.95363328  1.91651524  1.90203996]

'''

from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesRegressor

boston = load_boston()
x_train, x_test, y_train, y_test = train_test_split(boston.data, boston.target, random_state=42)
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

extra = ExtraTreesRegressor(n_estimators=100, criterion='mse', max_depth=None, min_samples_split=2,
                            min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto',
                            max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None,
                            bootstrap=True, oob_score=True, n_jobs=None, random_state=None,
                            verbose=0, warm_start=False, ccp_alpha=0.0, max_samples=None)
# 모델 학습
extra.fit(x_train,y_train)
# 모델 적합성 검
extra.score(x_train,y_train)
print("Train R2 Score : ", round(extra.score(x_train,y_train),3))
# Train R2 Score :  0.983
extra.score(x_test,y_test)
print("Test R2 Score : ", round(extra.score(x_test, y_test),3))
# Test R2 Score :  0.834
y_hat = extra.predict(x_test)
'''
array([23.47 , 32.348, 15.559, 23.838, 16.431, 21.784, 19.359, 15.612,
       20.965, 19.785, 20.988, 20.147,  8.883, 21.655, 19.681, 23.289,
       19.175,  8.907, 44.634, 14.983, 24.212, 24.471, 14.834, 22.662,
       15.664, 16.142, 21.567, 14.027, 19.811, 20.291, 20.536, 23.339,
       20.711, 20.774, 15.618, 16.718, 34.421, 19.275, 21.661, 23.856,
       18.806, 29.913, 44.323, 19.809, 23.472, 14.087, 15.144, 24.991,
       18.214, 27.185, 21.433, 33.947, 16.289, 25.426, 43.193, 22.073,
       15.023, 32.674, 23.047, 19.788, 25.33 , 34.778, 29.033, 19.653,
       26.613, 18.545, 13.414, 23.407, 28.511, 15.295, 20.623, 26.352,
       11.203, 22.086, 22.191,  7.636, 19.847, 45.215, 10.873, 11.92 ,
       21.211, 11.761, 19.891, 10.117, 19.946, 28.459, 15.761, 23.749,
       23.866, 18.322, 22.13 ,  8.235, 18.733, 19.463, 26.742, 19.815,
       26.729, 11.367, 12.549, 12.766, 21.348, 23.435, 12.996, 20.699,
       21.318, 12.147, 19.371, 24.725, 20.386, 23.632,  9.021, 13.781,
       22.656, 23.234, 32.478, 14.419, 40.011, 16.283, 20.553, 24.755,
       19.324, 24.619,  8.307, 20.334, 24.373, 21.55 , 24.885])
'''
#  모델 성능 평가
from sklearn.metrics import mean_absolute_error, mean_squared_error
mae = mean_absolute_error(y_test,y_hat)
mse = mean_squared_error(y_test, y_hat)
print("MAE : ", round(mae,3), " MSE : ", round(mse,3))
# MAE :  2.005  MSE :  11.626
# 특성 변수의 중요도
print("Feature Importance : \n" ,extra.feature_importances_)
'''
Feature Importance : 
 [0.03590818 0.00648428 0.03026274 0.02392035 0.0346256  0.31874654
 0.01906524 0.02876165 0.02114097 0.0389133  0.05456434 0.01938018
 0.36822663]
'''
# OOB(out ob bag) 측정 값 : oob_score = True, bootstrap = True 인경우에 사용가능
# 처음 보는 데이터로 확인한 학습 모델의 OOB Score
print("OOB score: " , round(extra.oob_score_,3))
# OOB score:  0.867
# 각각 estimator의 단계마다 추정된 OOB의값 출력
print("OOB Score Each Traing process : \n",extra.oob_prediction_)
'''
OOB Score Each Traing process : 
 [35.8        17.51818182 45.58918919 16.29473684 22.43030303 19.06857143
 18.80857143 14.27692308 20.20930233 20.62571429 31.7875     27.90810811
 16.84871795 18.51891892 24.98157895 19.91891892 18.25625     7.91714286
 22.09259259 17.73333333 13.53243243 15.18181818 42.705      14.875
 16.22571429 25.81190476 20.81071429 20.55       21.48       18.03
 24.51290323 34.83488372  9.2025     22.29375    18.67380952 20.86521739
 23.12978723 23.56666667 21.20526316 45.04848485 15.45121951 17.11351351
 17.34634146 20.21891892 20.1475     21.34210526 35.99302326 19.75277778
 18.99411765 22.24878049 27.345      35.86956522 25.98709677 15.06764706
 12.36153846 10.716      19.3525     31.95       27.77727273 13.9375
 13.28214286 40.00697674 20.89473684 19.27096774 21.30909091 19.69756098
 12.8         8.5025     30.085      26.26888889 19.07435897 15.86060606
 16.75277778 24.93611111 22.70434783 25.93589744 33.7        22.75882353
 22.68857143 22.47428571 45.95869565 32.17619048 35.18       24.41714286
 22.9025     16.05365854 38.53658537 19.71428571 31.2425     23.28611111
 20.84516129 21.37948718 10.77209302 42.0862069  40.20540541 34.4804878
  7.6        16.44324324 21.06666667 36.40487805 15.48157895 29.85806452
 21.940625   22.80731707 10.75897436 20.04090909 24.46333333 30.73333333
 26.68863636 20.91538462 39.56956522 26.84117647 33.3675     23.76333333
 19.43055556 24.03846154 11.63333333 20.65227273 22.32682927 21.6725
  9.17435897 19.94594595 24.64680851 21.03103448 20.128125   20.54736842
 13.57222222 32.76666667 24.08235294 34.60833333 11.425      15.88125
 12.66666667 14.57837838 11.65454545 23.42702703 16.22368421 15.26875
 22.60294118 13.54878049 25.13571429 33.04090909 45.18611111 29.51818182
 18.73333333 28.23       43.19090909 21.52790698 21.16666667 25.775
 20.64444444 22.303125   10.64571429 21.61777778 19.24761905 26.44117647
 24.85897436 20.00606061 22.825      16.75294118 13.98       34.87575758
 20.18064516 11.82972973 21.6475     23.2969697  20.19230769 22.37837838
 42.6         9.37878788 18.61875    33.29767442 21.04090909 24.0804878
 24.52580645 22.115      15.44210526 44.28275862 25.34444444 20.36904762
 16.79090909 34.67428571 37.92       44.00555556 19.13023256 20.66363636
 21.7516129  44.93714286 13.98529412 19.91785714 16.44705882 24.09
  8.76764706 19.29189189 12.08       15.34857143 17.66458333 20.45945946
 22.1675     20.3        24.7325     42.56842105 26.60769231 22.12727273
 42.05217391  9.37777778 26.57428571 21.34193548 21.04       18.54651163
 19.18108108 16.7425     10.5        30.26170213 18.63658537 26.81315789
 21.63076923 27.23076923 11.41578947 11.52619048 28.05789474 31.5
 10.80227273 15.23235294 23.25277778 43.5        23.140625   10.46
 35.11785714 39.58918919 18.17272727 22.8804878  44.52820513 13.95
 23.4675     14.72285714 22.36666667 43.61219512 28.38235294 22.98
 18.8        19.82857143 15.98421053 44.20512821 20.76153846 34.44193548
 15.3097561  17.25128205 17.         21.98378378 33.95945946 23.
 47.46153846 11.25       15.65       23.53243243 17.73902439 18.54736842
 31.7        17.93414634 23.35142857 19.39777778 31.73793103 29.58529412
 18.89333333 16.74901961 33.6375     12.45517241 21.02       20.47777778
 21.665      30.23103448 11.83888889 25.44193548 19.96060606 27.89375
 19.49166667 21.5375     26.16315789 20.68181818 26.565      20.70277778
 21.         31.29393939 23.9516129  22.5972973  21.53333333 43.85897436
 23.88378378 24.04871795 10.12307692 30.74848485 21.00666667 33.17777778
 18.30810811 21.890625   11.28837209 24.03030303 15.53939394 15.17428571
 20.33888889 10.9        28.06666667 22.4255814  14.28139535 13.56206897
 34.93030303 14.66969697 21.00285714 24.12631579 19.80434783 20.4975
 26.9902439  20.88648649 33.67826087 20.87894737 20.12222222 20.62121212
 30.72857143  9.24871795 37.71351351 46.98947368 20.878      21.06111111
 16.25428571 35.97209302 19.79393939 20.7902439  24.90277778 19.81891892
 20.5        21.515625   20.6        21.125      10.94375    33.98108108
 17.090625   21.90555556 27.83548387 23.08666667 22.77435897 17.52121212
 30.34318182 22.2097561  31.92647059 17.67575758 22.39285714 18.75128205
 28.43076923 34.07575758 13.98095238 30.33823529  9.61282051 22.82972973
 11.27567568 26.31851852 46.92307692 29.884375   16.184375   18.90967742
 18.91612903 20.60333333 33.59393939 22.8        21.95641026 17.2
 21.64411765 18.40888889 14.45142857 20.87073171 20.59189189 14.71794872
 27.12142857 22.23939394 18.003125   21.49705882 26.23684211 11.7027027
 19.99534884]
'''




from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

boston = load_boston()
x_train, x_test, y_train, y_test = train_test_split(boston.data, boston.target, random_state=42)
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

rf = RandomForestRegressor(n_estimators=100, criterion='mse', max_depth=None, min_samples_split=2,
                           min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto',
                           max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None,
                           bootstrap=True, oob_score=True, n_jobs=None, random_state=None, verbose=0,
                           warm_start=False, ccp_alpha=0.0, max_samples=None)
# 모델 학습
rf.fit(x_train,y_train)
# 모델 적합성 검
rf.score(x_train,y_train)
print("Train R2 Score : ", round(rf.score(x_train,y_train),3))
# Train R2 Score :  0.977
rf.score(x_test,y_test)
print("Test R2 Score : ", round(rf.score(x_test, y_test),3))
# Test R2 Score :  0.845
y_hat = rf.predict(x_test)
'''
array([22.918, 31.36 , 16.915, 23.371, 17.02 , 21.324, 19.505, 15.946,
       21.567, 21.016, 20.071, 19.861,  8.312, 21.727, 19.608, 26.138,
       19.089,  8.432, 45.42 , 15.559, 24.098, 23.862, 14.727, 23.292,
       14.873, 14.992, 21.626, 13.91 , 19.049, 21.193, 20.049, 23.345,
       31.773, 20.128, 14.737, 16.261, 35.207, 19.16 , 20.881, 24.3  ,
       19.536, 28.94 , 45.162, 19.918, 22.695, 13.894, 15.202, 24.606,
       18.902, 28.941, 21.358, 34.009, 16.926, 26.153, 45.818, 21.52 ,
       15.422, 32.585, 22.26 , 21.134, 25.59 , 34.107, 29.977, 18.769,
       27.314, 17.444, 13.624, 23.119, 28.821, 16.168, 20.613, 28.919,
       10.59 , 21.169, 22.05 ,  7.039, 20.456, 45.516, 10.879, 12.615,
       21.482, 11.528, 19.722,  9.065, 20.578, 26.949, 15.974, 23.334,
       23.784, 17.701, 21.643,  7.835, 19.282, 18.785, 22.588, 19.593,
       37.306, 11.659, 12.68 , 12.179, 20.142, 23.571, 14.056, 20.082,
       20.181, 13.265, 19.203, 24.668, 20.241, 23.382,  8.915, 14.624,
       22.742, 25.195, 31.346, 14.315, 42.685, 16.319, 19.976, 23.903,
       19.263, 24.083,  8.102, 20.416, 24.143, 21.929, 24.126])
'''
#  모델 성능 평가
from sklearn.metrics import mean_absolute_error, mean_squared_error
mae = mean_absolute_error(y_test,y_hat)
mse = mean_squared_error(y_test, y_hat)
print("MAE : ", round(mae,3), " MSE : ", round(mse,3))
# MAE :  2.157  MSE :  10.848
# 특성 변수의 중요도
print("Feature Importance : \n" ,rf.feature_importances_)
'''
Feature Importance : 
 [0.03757992 0.00262323 0.00594842 0.00075916 0.01005151 0.4340106
 0.01580443 0.05846708 0.00444953 0.01101456 0.01556444 0.01120572
 0.3925214 ]
'''
# OOB(out ob bag) 측정 값 : oob_score = True, bootstrap = True 인경우에 사용가능
# 처음 보는 데이터로 확인한 학습 모델의 OOB Score
print("OOB score: " , round(rf.oob_score_,3))
# OOB score:  0.842
# 각각 estimator의 단계마다 추정된 OOB의값 출력
print("OOB Score Each Traing process : \n",rf.oob_prediction_)
'''
OOB Score Each Traing process : 
 [35.80526316 17.14102564 46.02592593 15.75526316 23.77073171 19.79444444
 19.04       13.58214286 21.41428571 19.53142857 31.68372093 28.27105263
 16.94285714 18.39090909 24.57575758 20.1        18.80285714  6.495
 22.33947368 17.14444444 13.68484848 15.79487179 43.14166667 15.72222222
 16.04761905 23.03793103 21.04722222 19.93421053 19.98536585 14.73414634
 23.32972973 34.9875      9.60833333 22.5875     20.12424242 20.50833333
 23.11627907 19.97105263 23.66388889 47.48947368 15.4625     17.61891892
 16.85609756 20.5875     20.17352941 20.99090909 32.34545455 19.994
 19.25789474 21.70810811 26.55714286 34.71785714 24.68125    14.63103448
 11.32941176  9.3969697  14.83333333 30.76944444 28.35135135 15.41333333
 14.246875   39.15333333 20.86041667 20.87857143 21.43947368 20.29677419
 11.56666667  8.65806452 31.62285714 26.41794872 18.62777778 13.74186047
 14.51052632 23.365      20.33714286 25.60344828 32.55384615 23.01538462
 23.1804878  22.1725     48.20740741 31.56451613 31.36428571 23.98333333
 23.91388889 15.6195122  41.1        20.6        31.81428571 23.21142857
 19.61860465 21.00454545 10.625      44.13333333 43.91304348 33.221875
  6.88292683 15.83555556 21.66578947 34.77380952 15.61081081 34.53846154
 22.64594595 23.18484848 10.24615385 20.19722222 25.73       30.82380952
 25.86842105 20.69772727 45.78       28.075      33.6        22.115
 20.58717949 23.075      13.23055556 20.74878049 22.45       19.6775
  9.         20.66046512 26.38157895 20.76756757 20.12368421 19.99090909
 14.35227273 33.44418605 25.1        32.83055556 12.84318182 15.32162162
 14.14871795 14.2547619  11.078125   23.64324324 14.572      15.34545455
 22.14347826 13.79189189 26.88484848 32.71388889 43.85277778 28.03333333
 19.30681818 26.8        44.0125     21.92142857 21.04473684 25.77941176
 18.87209302 27.62333333 10.6974359  21.96842105 18.28       25.88
 24.74864865 20.27826087 22.75135135 16.82162162 14.92972973 35.45833333
 18.76842105 25.06       21.4804878  21.059375   20.72941176 22.17878788
 38.66585366  8.76285714 19.53684211 33.26       20.790625   24.06363636
 23.86756757 21.78292683 16.66129032 43.384375   24.17391304 20.93939394
 17.66511628 30.84848485 34.87142857 46.590625   19.06111111 20.54666667
 22.47741935 46.40285714 13.23409091 19.85641026 16.53055556 20.290625
  8.8137931  19.84594595 11.78648649 15.26944444 17.54516129 19.09444444
 21.53846154 19.52631579 24.9625     43.89787234 26.8902439  22.309375
 40.19354839  9.71025641 24.75757576 20.87586207 21.02222222 20.01020408
 18.53157895 14.765625   10.61142857 30.34054054 17.84210526 26.54375
 21.6627907  27.25128205 11.24871795 11.38684211 27.54102564 31.68125
  9.39117647 15.740625   22.7372093  44.46774194 22.19302326 11.87741935
 33.87317073 38.3425     29.51282051 24.42195122 47.11219512 17.11304348
 23.44571429 15.00769231 22.16153846 42.29722222 28.13611111 23.88709677
 19.56666667 20.41578947 18.71       42.64444444 21.465      34.07647059
 15.56046512 16.83611111 14.22307692 21.55625    32.64411765 23.23428571
 48.52564103 11.54090909 16.06097561 23.35945946 17.15581395 16.79230769
 33.9745098  16.37105263 23.22058824 15.38421053 32.37073171 29.39189189
 19.81111111 16.32727273 34.90588235 12.18863636 20.978125   29.11333333
 20.65625    32.02307692 11.21538462 28.29142857 19.471875   26.07111111
 20.73947368 21.07222222 25.9972973  21.25833333 27.53       20.14117647
 22.13333333 33.46363636 22.58823529 22.07352941 22.51463415 45.33684211
 23.38       25.02727273 10.72941176 31.44411765 21.41071429 28.97692308
 19.471875   20.78205128 11.3175     21.70810811 15.17631579 16.96888889
 19.99705882 12.1755102  28.846875   21.37317073 14.59666667 13.50869565
 33.15641026 14.77647059 20.08666667 22.79736842 19.853125   19.89166667
 24.48780488 20.6        33.65333333 21.9625     19.14545455 20.28
 31.81428571  8.43953488 42.59411765 48.81714286 21.03666667 22.36153846
 15.6025641  35.37714286 18.82439024 21.76       24.39722222 19.86976744
 16.93       21.3225     18.00357143 21.26666667 11.2975     35.01842105
 17.74482759 21.82391304 28.37352941 22.88648649 23.05       17.73777778
 30.77037037 22.48611111 32.60810811 17.96097561 23.08157895 16.8575
 24.3        35.48125    13.99189189 30.83055556  8.74166667 23.4862069
  9.07352941 28.1        46.20967742 29.71052632 14.40697674 19.03333333
 20.20930233 21.45       33.32333333 24.5725     21.65517241 16.171875
 22.62413793 19.42432432 15.38571429 22.15128205 21.45517241 15.54444444
 28.68387097 22.40606061 18.73684211 22.15806452 27.70740741 12.68823529
 18.97272727]
'''



from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.ensemble import StackingRegressor

boston = load_boston()
x_train, x_test, y_train, y_test = train_test_split(boston.data, boston.target, random_state=42)
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

# Base Estimator 선언
from sklearn.linear_model import RidgeCV, SGDRegressor
estimators = [('lr', RidgeCV()), ('sgd', SGDRegressor(random_state=42))]

stack_rg = StackingRegressor(estimators, final_estimator=None, cv=None,
                             n_jobs=None, passthrough=False, verbose=0)
# 모델 학습
stack_rg.fit(x_train,y_train)
# 모델 적합성 검
stack_rg.score(x_train,y_train)
print("Train R2 Score : ", round(stack_rg.score(x_train,y_train),3))
# Train R2 Score :  -4.459244337993647e+25
stack_rg.score(x_test,y_test)
print("Test R2 Score : ", round(stack_rg.score(x_test, y_test),3))
# Test R2 Score :  -5.65183822466637e+25
y_hat = stack_rg.predict(x_test)




# Base Estimator 선언
from sklearn.ensemble import VotingRegressor
from sklearn.linear_model import RidgeCV, SGDRegressor
estimators = [('lr', RidgeCV()), ('sgd', SGDRegressor(random_state=42))]

voting_rg = VotingRegressor(estimators, weights=None, n_jobs=None)
# 모델 학습
voting_rg.fit(x_train,y_train)
# 모델 적합성 검
voting_rg.score(x_train,y_train)
print("Train R2 Score : ", round(voting_rg.score(x_train,y_train),3))
# Train R2 Score :  -4.459244337993647e+25
voting_rg.score(x_test,y_test)
print("Test R2 Score : ", round(voting_rg.score(x_test, y_test),3))
# Test R2 Score :  -5.65183822466637e+25
y_hat = voting_rg.predict(x_test)