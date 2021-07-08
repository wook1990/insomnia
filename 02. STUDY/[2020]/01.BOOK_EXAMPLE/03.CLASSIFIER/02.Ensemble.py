from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
# 유방암 진단 데이터셋 활용

bc_data = load_breast_cancer()
from bs4 import BeautifulSoup
soup = BeautifulSoup(bc_data.DESCR, 'html.parser')
print(soup)
x_train, x_test, y_train, y_test = train_test_split(bc_data.data, bc_data.target, random_state=42)
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)
# 모듈 호출
from sklearn.ensemble import AdaBoostClassifier
# 모델 선언
adaboost = AdaBoostClassifier(base_estimator=None, n_estimators=50, learning_rate=1.0,
                              algorithm='SAMME.R', random_state=None)
# 모델 학습
adaboost.fit(x_train, y_train)
# 모델 적합성 검증(Train Score)
adaboost.score(x_train,y_train)
# 모델 적합성 검증(Test Score)
adaboost.score(x_test,y_test)
# predict 함수는 예측된 클래스의 결과를 반환
y_hat = adaboost.predict(x_test)
# predict_proba는 확률값을 반환
y_hat_prob = adaboost.predict_proba(x_test)[:,1]

# 모델 평가 지표
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, precision_score, confusion_matrix, recall_score
from sklearn.metrics import roc_curve
print("F1_score : {0:.3f}".format(f1_score(y_test, y_hat)))
# F1_score : 0.960
print("Accuracy : {0: .3f}".format(accuracy_score(y_test,y_hat)))
# Accuracy :  0.951
print("precision : {0: .3f}".format(precision_score(y_test,y_hat)))
# precision :  0.988
print("recall : {0: .3f}".format(recall_score(y_test,y_hat)))
# recall :  0.933
print("AUC of ROC Curve : {0: .3f}".format(roc_auc_score(y_test,y_hat_prob)))
# AUC of ROC Curve :  0.988
print("Confusion Matix : \n", confusion_matrix(y_test, y_hat,labels=[1,0]))
'''
Confusion Matix : 
 [[83  6]
 [ 1 53]]
'''
fpr, tpr, thresholds = roc_curve(y_test,y_hat_prob)

# Attribute
print("Feature Importance : \n ",adaboost.feature_importances_)
'''
Feature Importance : 
  [0.   0.06 0.   0.04 0.02 0.08 0.   0.08 0.02 0.02 0.02 0.   0.   0.06
 0.02 0.08 0.   0.   0.06 0.04 0.02 0.04 0.04 0.08 0.04 0.   0.08 0.04
 0.06 0.  ]
'''




# Bagging
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingClassifier
# 유방암 진단 데이터셋 활용

bc_data = load_breast_cancer()
x_train, x_test, y_train, y_test = train_test_split(bc_data.data, bc_data.target, random_state=42)
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)
# 모듈 호출
# 모델 선언
bagging = BaggingClassifier(base_estimator=None, n_estimators=10,  max_samples=1.0, max_features=1.0, bootstrap=True,
                            bootstrap_features=False, oob_score=False, warm_start=False, n_jobs=None,
                            random_state=None, verbose=0)
# 모델 학습
bagging.fit(x_train, y_train)
# 모델 적합성 검증(Train Score)
bagging.score(x_train,y_train)
# 모델 적합성 검증(Test Score)
bagging.score(x_test,y_test)
# predict 함수는 예측된 클래스의 결과를 반환
y_hat = bagging.predict(x_test)
# predict_proba는 확률값을 반환
y_hat_prob = bagging.predict_proba(x_test)[:,1]

# 모델 평가 지표
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, precision_score, confusion_matrix, recall_score
from sklearn.metrics import roc_curve
print("F1_score : {0:.3f}".format(f1_score(y_test, y_hat)))
# F1_score : 0.966
print("Accuracy : {0: .3f}".format(accuracy_score(y_test,y_hat)))
# Accuracy :  0.958
print("precision : {0: .3f}".format(precision_score(y_test,y_hat)))
# precision :  0.977
print("recall : {0: .3f}".format(recall_score(y_test,y_hat)))
# recall :  0.955
print("AUC of ROC Curve : {0: .3f}".format(roc_auc_score(y_test,y_hat_prob)))
# AUC of ROC Curve :  0.990
print("Confusion Matix : \n", confusion_matrix(y_test, y_hat,labels=[1,0]))
'''
Confusion Matix : 
[[85  4]
 [ 2 52]]
'''
fpr, tpr, thresholds = roc_curve(y_test,y_hat_prob)

# Attribute
print("Feature Importance : \n ",bagging.feature_importances_)
'''
Feature Importance : 
  [0.   0.06 0.   0.04 0.02 0.08 0.   0.08 0.02 0.02 0.02 0.   0.   0.06
 0.02 0.08 0.   0.   0.06 0.04 0.02 0.04 0.04 0.08 0.04 0.   0.08 0.04
 0.06 0.  ]
'''


# ExtraTreesClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesClassifier
# 유방암 진단 데이터셋 활용

bc_data = load_breast_cancer()
x_train, x_test, y_train, y_test = train_test_split(bc_data.data, bc_data.target, random_state=42)
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)
# 모듈 호출
# 모델 선언
extra_clf = ExtraTreesClassifier(n_estimators=100, criterion='gini', max_depth=None, min_samples_split=2,
                               min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto',
                               max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None,
                               bootstrap=False, oob_score=False, n_jobs=None, random_state=None, verbose=0,
                               warm_start=False, class_weight=None, ccp_alpha=0.0, max_samples=None)
# 모델 학습
extra_clf.fit(x_train, y_train)
# 모델 적합성 검증(Train Score)
extra_clf.score(x_train,y_train)
# 모델 적합성 검증(Test Score)
extra_clf.score(x_test,y_test)
# predict 함수는 예측된 클래스의 결과를 반환
y_hat = extra_clf.predict(x_test)
# predict_proba는 확률값을 반환
y_hat_prob = extra_clf.predict_proba(x_test)[:,1]

# 모델 평가 지표
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, precision_score, confusion_matrix, recall_score
from sklearn.metrics import roc_curve
print("F1_score : {0:.3f}".format(f1_score(y_test, y_hat)))
# F1_score : 0.983
print("Accuracy : {0: .3f}".format(accuracy_score(y_test,y_hat)))
# Accuracy :  0.979
print("precision : {0: .3f}".format(precision_score(y_test,y_hat)))
# precision :  0.978
print("recall : {0: .3f}".format(recall_score(y_test,y_hat)))
# recall :  0.989
print("AUC of ROC Curve : {0: .3f}".format(roc_auc_score(y_test,y_hat_prob)))
# AUC of ROC Curve :  0.998
print("Confusion Matix : \n", confusion_matrix(y_test, y_hat,labels=[1,0]))
'''
Confusion Matix : 
 [[88  1]
 [ 2 52]]
'''
fpr, tpr, thresholds = roc_curve(y_test,y_hat_prob)

# Attribute
print("Feature Importance : \n ",extra_clf.feature_importances_)
'''
Feature Importance : 
  [0.06889303 0.01864199 0.06544649 0.066355   0.00901341 0.02790835
 0.05030664 0.09705546 0.01022562 0.00853061 0.01438313 0.00794999
 0.01930588 0.03425275 0.00669302 0.00685586 0.01295481 0.0109169
 0.00675591 0.00718216 0.08299192 0.03249281 0.06106668 0.06181918
 0.0190987  0.01780035 0.05670986 0.08912532 0.01770391 0.01156425]
'''


# GradientBoostingClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
# 유방암 진단 데이터셋 활용

bc_data = load_breast_cancer()
x_train, x_test, y_train, y_test = train_test_split(bc_data.data, bc_data.target, random_state=42)
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)
# 모듈 호출
# 모델 선언
gbm_clf = GradientBoostingClassifier(loss='deviance', learning_rate=0.1, n_estimators=100, subsample=1.0,
                                     criterion='friedman_mse', min_samples_split=2, min_samples_leaf=1,
                                     min_weight_fraction_leaf=0.0, max_depth=3, min_impurity_decrease=0.0,
                                     min_impurity_split=None, init=None, random_state=None, max_features=None,
                                     verbose=0, max_leaf_nodes=None, warm_start=False, presort='deprecated',
                                     validation_fraction=0.1, n_iter_no_change=None, tol=0.0001, ccp_alpha=0.0)
# 모델 학습
gbm_clf.fit(x_train, y_train)
# 모델 적합성 검증(Train Score)
gbm_clf.score(x_train,y_train)
# 모델 적합성 검증(Test Score)
gbm_clf.score(x_test,y_test)
# predict 함수는 예측된 클래스의 결과를 반환
y_hat = gbm_clf.predict(x_test)
# predict_proba는 확률값을 반환
y_hat_prob = gbm_clf.predict_proba(x_test)[:,1]

# 모델 평가 지표
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, precision_score, confusion_matrix, recall_score
from sklearn.metrics import roc_curve
print("F1_score : {0:.3f}".format(f1_score(y_test, y_hat)))
# F1_score : 0.966
print("Accuracy : {0: .3f}".format(accuracy_score(y_test,y_hat)))
# Accuracy :  0.958
print("precision : {0: .3f}".format(precision_score(y_test,y_hat)))
# precision :  0.966
print("recall : {0: .3f}".format(recall_score(y_test,y_hat)))
# recall :  0.966
print("AUC of ROC Curve : {0: .3f}".format(roc_auc_score(y_test,y_hat_prob)))
# AUC of ROC Curve :  0.994
print("Confusion Matix : \n", confusion_matrix(y_test, y_hat,labels=[1,0]))
'''
Confusion Matix : 
 [[86  3]
 [ 3 51]]
'''
fpr, tpr, thresholds = roc_curve(y_test,y_hat_prob)

# Attribute
print("Feature Importance : \n ",gbm_clf.feature_importances_)
'''
Feature Importance : 
  [5.63200022e-05 2.00511580e-02 2.17928987e-03 2.88979719e-05
 6.51579214e-05 3.94105867e-03 1.03612209e-03 4.67626798e-01
 8.34727193e-04 2.50878400e-04 1.13342774e-02 8.14903472e-03
 5.51602240e-04 3.75868356e-03 8.16524948e-04 3.02961239e-03
 1.33624054e-02 9.33734576e-03 1.60137242e-03 3.25594323e-04
 6.53191152e-02 4.96904137e-02 2.81742631e-02 4.17489701e-02
 3.96643598e-03 3.21361215e-04 1.71773726e-02 2.43421762e-01
 9.81317979e-04 8.62126620e-04]
'''



# RandomForestClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
# 유방암 진단 데이터셋 활용

bc_data = load_breast_cancer()
x_train, x_test, y_train, y_test = train_test_split(bc_data.data, bc_data.target, random_state=42)
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)
# 모델 선언
rf_clf = RandomForestClassifier(n_estimators=100, criterion='gini', max_depth=None, min_samples_split=2,
                                min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto',
                                max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None,
                                bootstrap=True, oob_score=False, n_jobs=None, random_state=None, verbose=0,
                                warm_start=False, class_weight=None, ccp_alpha=0.0, max_samples=None)
# 모델 학습
rf_clf.fit(x_train, y_train)
# 모델 적합성 검증(Train Score)
rf_clf.score(x_train,y_train)
# 모델 적합성 검증(Test Score)
rf_clf.score(x_test,y_test)
# predict 함수는 예측된 클래스의 결과를 반환
y_hat = rf_clf.predict(x_test)
# predict_proba는 확률값을 반환
y_hat_prob = rf_clf.predict_proba(x_test)[:,1]

# 모델 평가 지표
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, precision_score, confusion_matrix, recall_score
from sklearn.metrics import roc_curve
print("F1_score : {0:.3f}".format(f1_score(y_test, y_hat)))
# F1_score : 0.978
print("Accuracy : {0: .3f}".format(accuracy_score(y_test,y_hat)))
# Accuracy :  0.972
print("precision : {0: .3f}".format(precision_score(y_test,y_hat)))
# pprecision :  0.967
print("recall : {0: .3f}".format(recall_score(y_test,y_hat)))
# recall :  0.989
print("AUC of ROC Curve : {0: .3f}".format(roc_auc_score(y_test,y_hat_prob)))
# AUC of ROC Curve :  0.996
print("Confusion Matix : \n", confusion_matrix(y_test, y_hat,labels=[1,0]))
'''
Confusion Matix : 
 [[88  1]
 [ 3 51]]
'''
fpr, tpr, thresholds = roc_curve(y_test,y_hat_prob)

# Attribute
print("Feature Importance : \n ",rf_clf.feature_importances_)
'''
Feature Importance : 
  [0.00765357 0.01528392 0.04111946 0.05315443 0.0070303  0.01431716
 0.05799417 0.10223526 0.00426246 0.00608807 0.02132668 0.00470753
 0.01357827 0.05703214 0.00327027 0.0060552  0.00767488 0.00416662
 0.00542435 0.00713148 0.10282653 0.01831647 0.13747965 0.09552919
 0.01670498 0.01599118 0.02279942 0.13042368 0.016581   0.0038417 ]
'''


# StackingClassifier

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import StackingClassifier
# 유방암 진단 데이터셋 활용

bc_data = load_breast_cancer()
x_train, x_test, y_train, y_test = train_test_split(bc_data.data, bc_data.target, random_state=42)
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)
# 모델 선언
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
estimators = [
     ('rf', RandomForestClassifier(n_estimators=10, random_state=42)),
     ('svr', make_pipeline(StandardScaler(),
                           LinearSVC(random_state=42)))]


stack_clf = StackingClassifier(estimators, final_estimator=None,  cv=None,
              stack_method='auto', n_jobs=None, passthrough=False, verbose=0)
# 모델 학습
stack_clf.fit(x_train, y_train)
# 모델 적합성 검증(Train Score)
stack_clf.score(x_train,y_train)
# 모델 적합성 검증(Test Score)
stack_clf.score(x_test,y_test)
# predict 함수는 예측된 클래스의 결과를 반환
y_hat = stack_clf.predict(x_test)
# predict_proba는 확률값을 반환
y_hat_prob = stack_clf.predict_proba(x_test)[:,1]

# VotingClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import VotingClassifier
# 유방암 진단 데이터셋 활용

bc_data = load_breast_cancer()
x_train, x_test, y_train, y_test = train_test_split(bc_data.data, bc_data.target, random_state=42)
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)
# 모델 선언
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
estimators = [
     ('rf', RandomForestClassifier(n_estimators=10, random_state=42)),
     ('svr', make_pipeline(StandardScaler(),
                           LinearSVC(random_state=42)))]


voting_clf = VotingClassifier(estimators,voting='hard', weights=None, n_jobs=None,
                              flatten_transform=True)
# 모델 학습
voting_clf.fit(x_train, y_train)
# 모델 적합성 검증(Train Score)
voting_clf.score(x_train,y_train)
# 모델 적합성 검증(Test Score)
voting_clf.score(x_test,y_test)
# predict 함수는 예측된 클래스의 결과를 반환
y_hat = voting_clf.predict(x_test)
# predict_proba는 확률값을 반환
y_hat_prob = voting_clf.predict_proba(x_test)[:,1]