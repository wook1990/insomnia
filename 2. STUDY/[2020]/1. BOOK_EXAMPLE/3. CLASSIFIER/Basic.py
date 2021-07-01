from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
# 유방암 진단 데이터셋 활용

bc_data = load_breast_cancer()
x_train, x_test, y_train, y_test = train_test_split(bc_data.data, bc_data.target, random_state=42)
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)
# 모델 선언
decision_clf = DecisionTreeClassifier(criterion='gini', splitter='best', max_depth=None, min_samples_split=2,
                                    min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=None,
                                    random_state=None, max_leaf_nodes=None, min_impurity_decrease=0.0,
                                    min_impurity_split=None, class_weight=None, presort='deprecated', ccp_alpha=0.0)
# 모델 학습
decision_clf.fit(x_train, y_train)
# 모델 적합성 검증(Train Score)
decision_clf.score(x_train,y_train)
# 모델 적합성 검증(Test Score)
decision_clf.score(x_test,y_test)
# predict 함수는 예측된 클래스의 결과를 반환
y_hat = decision_clf.predict(x_test)
# predict_proba는 확률값을 반환
y_hat_prob = decision_clf.predict_proba(x_test)[:,1]


from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
# 유방암 진단 데이터셋 활용

bc_data = load_breast_cancer()
x_train, x_test, y_train, y_test = train_test_split(bc_data.data, bc_data.target, random_state=42)
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)
# (426, 30) (143, 30) (426,) (143,)

# 모델 선언
decision_clf = DecisionTreeClassifier(criterion='gini', splitter='best', max_depth=None, min_samples_split=2,
                                    min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=None,
                                    random_state=None, max_leaf_nodes=None, min_impurity_decrease=0.0,
                                    min_impurity_split=None, class_weight=None, presort='deprecated', ccp_alpha=0.0)
# 모델 학습
decision_clf.fit(x_train, y_train)
# 모델 적합성 검증(Train Score)
print("Train R2 score : ", round(decision_clf.score(x_train,y_train),3))
# Train R2 score :  1.0
# 모델 적합성 검증(Test Score)
print("Test R2 score : ", round(decision_clf.score(x_test,y_test),3))
# Test R2 score :  0.951
# predict 함수는 예측된 클래스의 결과를 반환
y_hat = decision_clf.predict(x_test)
'''
array([1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1,
       0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1,
       1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1,
       0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0,
       1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 1,
       0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0,
       1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 1])
'''
# predict_proba는 확률값을 반환
y_hat_prob = decision_clf.predict_proba(x_test)[:,1]

# 예측 결과의 정확도 검증
from sklearn.metrics import accuracy_score
print("accuracy : ", round(accuracy_score(y_test, y_hat),3))
# accuracy :  0.951



# ----------------------------------------------------------------------------#
# ExtraTree

from sklearn.tree import ExtraTreeClassifier

# 모델 선언
extra_clf = ExtraTreeClassifier(criterion='gini', splitter='random', max_depth=None, min_samples_split=2,
                                 min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto',
                                 random_state=None, max_leaf_nodes=None, min_impurity_decrease=0.0,
                                 min_impurity_split=None, class_weight=None, ccp_alpha=0.0)
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


from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import ExtraTreeClassifier
# 유방암 진단 데이터셋 활용

bc_data = load_breast_cancer()
x_train, x_test, y_train, y_test = train_test_split(bc_data.data, bc_data.target, random_state=42)
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)
# (426, 30) (143, 30) (426,) (143,)

# 모델 선언
extra_clf = ExtraTreeClassifier(criterion='gini', splitter='random', max_depth=None, min_samples_split=2,
                                 min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto',
                                 random_state=None, max_leaf_nodes=None, min_impurity_decrease=0.0,
                                 min_impurity_split=None, class_weight=None, ccp_alpha=0.0)
# 모델 학습
extra_clf.fit(x_train, y_train)
# 모델 적합성 검증(Train Score)
print("Train R2 score : ", round(extra_clf.score(x_train,y_train),3))
# Train R2 score :  1.0
# 모델 적합성 검증(Test Score)
print("Test R2 score : ", round(extra_clf.score(x_test,y_test),3))
# Test R2 score :  0.93
# predict 함수는 예측된 클래스의 결과를 반환
y_hat = extra_clf.predict(x_test)
'''
array([1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1,
       0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1,
       1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 1,
       0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0,
       1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 1,
       0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0,
       1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 1])
'''
# predict_proba는 확률값을 반환
y_hat_prob = extra_clf.predict_proba(x_test)[:,1]

# 예측 결과의 정확도 검증
from sklearn.metrics import accuracy_score
print("accuracy : ", round(accuracy_score(y_test, y_hat),3))
# accuracy :  0.93


#----------------------------------------------------------------------------------------#
# LogisticRegression
from sklearn.linear_model import LogisticRegression

# 모델 선언
logi_clf = LogisticRegression(penalty='l2', dual=False, tol=0.0001, C=1.0, fit_intercept=True, intercept_scaling=1,
                              class_weight=None, random_state=None, solver='lbfgs', max_iter=100, multi_class='auto',
                              verbose=0, warm_start=False, n_jobs=None, l1_ratio=None)

# 모델 학습
logi_clf.fit(x_train, y_train)

# 모델 적합성 검증(Train Score)
logi_clf.score(x_train,y_train)

# 모델 적합성 검증(Test Score)
logi_clf.score(x_test,y_test)

# predict 함수는 예측된 클래스의 결과를 반환
y_hat = logi_clf.predict(x_test)

# predict_proba는 확률값을 반환
y_hat_prob = logi_clf.predict_proba(x_test)[:,1]



from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
# 유방암 데이터
bc_data = load_breast_cancer()
x_train, x_test, y_train, y_test = train_test_split(bc_data.data, bc_data.target, random_state=42)
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)
# (426, 30) (143, 30) (426,) (143,)

# 모델 선언
logi_clf = LogisticRegression(penalty='l2', dual=False, tol=0.0001, C=1.0, fit_intercept=True, intercept_scaling=1,
                               class_weight=None, random_state=None, solver='lbfgs', max_iter=100, multi_class='auto',
                               verbose=0, warm_start=False, n_jobs=None, l1_ratio=None)


# 모델 학습
logi_clf.fit(x_train, y_train)
# 모델 적합성 검증(Train Score)
print("Train R2 score : ", round(logi_clf.score(x_train,y_train),3))
# Train R2 score :  0.944

# 모델 적합성 검증(Test Score)
print("Test R2 score : ", round(logi_clf.score(x_test,y_test),3))
# Test R2 score :  0.965

# predict 함수는 예측된 클래스의 결과를 반환
y_hat = logi_clf.predict(x_test)

# predict_proba는 확률값을 반환
y_hat_prob = logi_clf.predict_proba(x_test)[:,1]
y_hat = [1 if i > 0.5 else 0 for i in y_hat_prob]

# 예측 결과의 정확도 검증
from sklearn.metrics import accuracy_score
print("accuracy : ", round(accuracy_score(y_test, y_hat),3))
# accuracy :  0.965



#---------------------------------------#
# LogisticCV

from sklearn.linear_model import LogisticRegressionCV

# 모델 선언
logi_cv_clf = LogisticRegressionCV(Cs=10, fit_intercept=True, cv=None, dual=False, penalty='l2', scoring=None, solver='lbfgs',
                                   tol=0.0001, max_iter=100, class_weight=None, n_jobs=None, verbose=0, refit=True,
                                   intercept_scaling=1.0, multi_class='auto', random_state=None, l1_ratios=None)

# 모델 학습
logi_cv_clf.fit(x_train, y_train)

# 모델 적합성 검증(Train Score)
logi_cv_clf.score(x_train,y_train)

# 모델 적합성 검증(Test Score)
logi_cv_clf.score(x_test,y_test)

# predict 함수는 예측된 클래스의 결과를 반환
y_hat = logi_cv_clf.predict(x_test)

# predict_proba는 확률값을 반환
y_hat_prob = logi_cv_clf.predict_proba(x_test)[:,1]


from sklearn.linear_model import LogisticRegressionCV
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
# 유방암 데이터
bc_data = load_breast_cancer()
x_train, x_test, y_train, y_test = train_test_split(bc_data.data, bc_data.target, random_state=42)
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)
# (426, 30) (143, 30) (426,) (143,)

# 모델 선언
logi_cv_clf = LogisticRegressionCV(Cs=10, fit_intercept=True, cv=None, dual=False, penalty='l2', scoring=None, solver='lbfgs',
                                   tol=0.0001, max_iter=100, class_weight=None, n_jobs=None, verbose=0, refit=True,
                                   intercept_scaling=1.0, multi_class='auto', random_state=None, l1_ratios=None)


# 모델 학습
logi_cv_clf.fit(x_train, y_train)
# 모델 적합성 검증(Train Score)
print("Train R2 score : ", round(logi_cv_clf.score(x_train,y_train),3))
# Train R2 score :  0.974

# 모델 적합성 검증(Test Score)
print("Test R2 score : ", round(logi_cv_clf.score(x_test,y_test),3))
# Test R2 score :  0.965

# predict 함수는 예측된 클래스의 결과를 반환
y_hat = logi_cv_clf.predict(x_test)

# predict_proba는 확률값을 반환
y_hat_prob = logi_cv_clf.predict_proba(x_test)[:,1]
y_hat = [1 if i > 0.5 else 0 for i in y_hat_prob]

# 예측 결과의 정확도 검증
from sklearn.metrics import accuracy_score
print("accuracy : ", round(accuracy_score(y_test, y_hat),3))
# accuracy :  0.965



# -----------------------------------------------------------#

from sklearn.linear_model import RidgeClassifier

# 모델 선언
ridge_clf = RidgeClassifier(alpha=1.0, fit_intercept=True, normalize=False, copy_X=True, max_iter=None, tol=0.001,
                              class_weight=None, solver='auto', random_state=None)

# 모델 학습
ridge_clf.fit(x_train, y_train)

# 모델 적합성 검증(Train Score)
ridge_clf.score(x_train,y_train)

# 모델 적합성 검증(Test Score)
ridge_clf.score(x_test,y_test)

# predict 함수는 예측된 클래스의 결과를 반환
y_hat = ridge_clf.predict(x_test)

# predict_proba는 확률값을 반환
y_hat_prob = ridge_clf.predict_proba(x_test)[:,1]


from sklearn.linear_model import RidgeClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
# 유방암 데이터
bc_data = load_breast_cancer()
x_train, x_test, y_train, y_test = train_test_split(bc_data.data, bc_data.target, random_state=42)
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)
# (426, 30) (143, 30) (426,) (143,)

# 모델 선언
ridge_clf = RidgeClassifier(alpha=1.0, fit_intercept=True, normalize=False, copy_X=True, max_iter=None, tol=0.001,
                              class_weight=None, solver='auto', random_state=None)


# 모델 학습
ridge_clf.fit(x_train, y_train)
# 모델 적합성 검증(Train Score)
print("Train R2 score : ", round(ridge_clf.score(x_train,y_train),3))
# Train R2 score :  0.958

# 모델 적합성 검증(Test Score)
print("Test R2 score : ", round(ridge_clf.score(x_test,y_test),3))
# Test R2 score :  0.951

# predict 함수는 예측된 클래스의 결과를 반환
y_hat = ridge_clf.predict(x_test)

# predict_proba는 확률값을 반환
y_hat_prob = ridge_clf.predict_proba(x_test)[:,1]
y_hat = [1 if i > 0.5 else 0 for i in y_hat_prob]

# 예측 결과의 정확도 검증
from sklearn.metrics import accuracy_score
print("accuracy : ", round(accuracy_score(y_test, y_hat),3))
# accuracy :  0.951



# ------------------------------------------------------------------------------#
from sklearn.linear_model import RidgeClassifierCV

# 모델 선언
ridge_cv_clf = RidgeClassifierCV(alphas=(0.1, 1.0, 10.0), fit_intercept=True, normalize=False, scoring=None,
                                 cv=None, class_weight=None, store_cv_values=False)

# 모델 학습
ridge_cv_clf.fit(x_train, y_train)

# 모델 적합성 검증(Train Score)
ridge_cv_clf.score(x_train,y_train)

# 모델 적합성 검증(Test Score)
ridge_cv_clf.score(x_test,y_test)

# predict 함수는 예측된 클래스의 결과를 반환
y_hat = ridge_cv_clf.predict(x_test)

# predict_proba는 확률값을 반환
y_hat_prob = ridge_clf.ridge_cv_clf(x_test)[:,1]


from sklearn.linear_model import RidgeClassifierCV
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
# 유방암 데이터
bc_data = load_breast_cancer()
x_train, x_test, y_train, y_test = train_test_split(bc_data.data, bc_data.target, random_state=42)
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)
# (426, 30) (143, 30) (426,) (143,)

# 모델 선언
ridge_cv_clf = RidgeClassifierCV(alphas=(0.1, 1.0, 10.0), fit_intercept=True, normalize=False, scoring=None,
                                 cv=None, class_weight=None, store_cv_values=False)


# 모델 학습
ridge_cv_clf.fit(x_train, y_train)
# 모델 적합성 검증(Train Score)
print("Train R2 score : ", round(ridge_cv_clf.score(x_train,y_train),3))
# Train R2 score :  0.962

# 모델 적합성 검증(Test Score)
print("Test R2 score : ", round(ridge_cv_clf.score(x_test,y_test),3))
# Test R2 score :  0.972

# predict 함수는 예측된 클래스의 결과를 반환
y_hat = ridge_cv_clf.predict(x_test)

# predict_proba는 확률값을 반환
y_hat_prob = ridge_cv_clf._predict_proba(x_test)[:,1]
y_hat = [1 if i > 0.5 else 0 for i in y_hat_prob]

# 예측 결과의 정확도 검증
from sklearn.metrics import accuracy_score
print("accuracy : ", round(accuracy_score(y_test, y_hat),3))
# accuracy :  0.972


#------------------------------------------------------------------#
from sklearn.neural_network import MLPClassifier

mlp_clf = MLPClassifier(hidden_layer_sizes=(100, ), activation='relu', solver='adam', alpha=0.0001, 
                        batch_size='auto', learning_rate='constant', learning_rate_init=0.001, power_t=0.5, 
                        max_iter=200, shuffle=True, random_state=None, tol=0.0001, verbose=False, warm_start=False,
                        momentum=0.9, nesterovs_momentum=True, early_stopping=False, validation_fraction=0.1, beta_1=0.9, 
                        beta_2=0.999, epsilon=1e-08, n_iter_no_change=10, max_fun=15000)

# 모델 학습
mlp_clf.fit(x_train, y_train)

# 모델 적합성 검증(Train Score)
mlp_clf.score(x_train,y_train)

# 모델 적합성 검증(Test Score)
mlp_clf.score(x_test,y_test)

# predict 함수는 예측된 클래스의 결과를 반환
y_hat = mlp_clf.predict(x_test)

# predict_proba는 확률값을 반환
y_hat_prob = mlp_clf.predict_proba(x_test)[:,1]


from sklearn.neural_network import MLPClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
# 유방암 데이터
bc_data = load_breast_cancer()
x_train, x_test, y_train, y_test = train_test_split(bc_data.data, bc_data.target, random_state=42)
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)
# (426, 30) (143, 30) (426,) (143,)

# 모델 선언
mlp_clf = MLPClassifier(hidden_layer_sizes=(100, ), activation='relu', solver='adam', alpha=0.0001, 
                        batch_size='auto', learning_rate='constant', learning_rate_init=0.001, power_t=0.5, 
                        max_iter=200, shuffle=True, random_state=None, tol=0.0001, verbose=False, warm_start=False,
                        momentum=0.9, nesterovs_momentum=True, early_stopping=False, validation_fraction=0.1, beta_1=0.9, 
                        beta_2=0.999, epsilon=1e-08, n_iter_no_change=10, max_fun=15000)



# 모델 학습
mlp_clf.fit(x_train, y_train)
# 모델 적합성 검증(Train Score)
print("Train R2 score : ", round(mlp_clf.score(x_train,y_train),3))
# Train R2 score :  0.925

# 모델 적합성 검증(Test Score)
print("Test R2 score : ", round(mlp_clf.score(x_test,y_test),3))
# Test R2 score :  0.951

# predict 함수는 예측된 클래스의 결과를 반환
y_hat = mlp_clf.predict(x_test)

# predict_proba는 확률값을 반환
y_hat_prob = mlp_clf.predict_proba(x_test)[:,1]
y_hat = [1 if i > 0.5 else 0 for i in y_hat_prob]

# 예측 결과의 정확도 검증
from sklearn.metrics import accuracy_score
print("accuracy : ", round(accuracy_score(y_test, y_hat),3))
# accuracy :  0.951

#--------------------------------------------------#

from sklearn.linear_model import SGDClassifier

sgd_clf = SGDClassifier(loss='hinge', penalty='l2', alpha=0.0001, l1_ratio=0.15, fit_intercept=True, max_iter=1000, 
                        tol=0.001, shuffle=True, verbose=0, epsilon=0.1, n_jobs=None, random_state=None, 
                        learning_rate='optimal', eta0=0.0, power_t=0.5, early_stopping=False, validation_fraction=0.1, 
                        n_iter_no_change=5, class_weight=None, warm_start=False, average=False)

# 모델 학습
sgd_clf.fit(x_train, y_train)

# 모델 적합성 검증(Train Score)
sgd_clf.score(x_train,y_train)

# 모델 적합성 검증(Test Score)
sgd_clf.score(x_test,y_test)

# predict 함수는 예측된 클래스의 결과를 반환
y_hat = sgd_clf.predict(x_test)


from sklearn.linear_model import SGDClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
# 유방암 데이터
bc_data = load_breast_cancer()
x_train, x_test, y_train, y_test = train_test_split(bc_data.data, bc_data.target, random_state=42)
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)
# (426, 30) (143, 30) (426,) (143,)

# 모델 선언
sgd_clf = SGDClassifier(loss='hinge', penalty='l2', alpha=0.0001, l1_ratio=0.15, fit_intercept=True, max_iter=1000, 
                        tol=0.001, shuffle=True, verbose=0, epsilon=0.1, n_jobs=None, random_state=None, 
                        learning_rate='optimal', eta0=0.0, power_t=0.5, early_stopping=False, validation_fraction=0.1, 
                        n_iter_no_change=5, class_weight=None, warm_start=False, average=False)



# 모델 학습
sgd_clf.fit(x_train, y_train)
# 모델 적합성 검증(Train Score)
print("Train R2 score : ", round(sgd_clf.score(x_train,y_train),3))
# Train R2 score :  0.765

# 모델 적합성 검증(Test Score)
print("Test R2 score : ", round(sgd_clf.score(x_test,y_test),3))
# Test R2 score :  0.804

# predict 함수는 예측된 클래스의 결과를 반환
y_hat = sgd_clf.predict(x_test)

# 예측 결과의 정확도 검증
from sklearn.metrics import accuracy_score
print("accuracy : ", round(accuracy_score(y_test, y_hat),3))
# accuracy :  0.804



#-----------------------------------------------------------------#
import xgboost

xgb_clf = xgboost.XGBClassifier(base_score=0.5, colsample_bylevel=1, colsample_bytree=1,
                                gamma=0, learning_rate=0.1, max_delta_step=0, max_depth=10,
                                min_child_weight=1, missing=None, n_estimators=100, nthread=-1,
                                objective='binary:logistic', reg_alpha=0, reg_lambda=1,
                                scale_pos_weight=1, seed=0, silent=True, subsample=1)

# 모델 학습
xgb_clf.fit(x_train, y_train)

# 모델 적합성 검증(Train Score)
xgb_clf.score(x_train,y_train)

# 모델 적합성 검증(Test Score)
xgb_clf.score(x_test,y_test)

# predict 함수는 예측된 클래스의 결과를 반환
y_hat = xgb_clf.predict(x_test)

# predict_proba는 확률값을 반환
y_hat_prob = xgb_clf.predict_proba(x_test)[:,1]
y_hat = [1 if i > 0.5 else 0 for i in y_hat_prob]


import xgboost
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
# 유방암 데이터
bc_data = load_breast_cancer()
x_train, x_test, y_train, y_test = train_test_split(bc_data.data, bc_data.target, random_state=42)
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)
# (426, 30) (143, 30) (426,) (143,)

# 모델 선언
xgb_clf = xgboost.XGBClassifier(base_score=0.5, colsample_bylevel=1, colsample_bytree=1,
                                gamma=0, learning_rate=0.1, max_delta_step=0, max_depth=10,
                                min_child_weight=1, missing=None, n_estimators=100, nthread=-1,
                                objective='binary:logistic', reg_alpha=0, reg_lambda=1,
                                scale_pos_weight=1, seed=0, silent=True, subsample=1)



# 모델 학습
xgb_clf.fit(x_train, y_train)
# 모델 적합성 검증(Train Score)
print("Train R2 score : ", round(xgb_clf.score(x_train,y_train),3))
# Train R2 score :  1.0

# 모델 적합성 검증(Test Score)
print("Test R2 score : ", round(xgb_clf.score(x_test,y_test),3))
# Test R2 score :  0.972

# predict 함수는 예측된 클래스의 결과를 반환
y_hat = xgb_clf.predict(x_test)

# predict_proba는 확률값을 반환
y_hat_prob = xgb_clf.predict_proba(x_test)[:,1]
y_hat = [1 if i > 0.5 else 0 for i in y_hat_prob]


# 예측 결과의 정확도 검증
from sklearn.metrics import accuracy_score
print("accuracy : ", round(accuracy_score(y_test, y_hat),3))
# accuracy :  0.972