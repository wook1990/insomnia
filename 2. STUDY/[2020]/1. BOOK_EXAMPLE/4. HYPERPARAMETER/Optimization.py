from sklearn import datasets
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import numpy as np


# 1.Cross Validation + Grid Search
# 데이터를 train, validation, test 로 나누는 방법은 성능이 좋고 널리 사용되지만,
# 데이터를 나누는 방법에 매우 민감하여 일반화 성능을 평가하기 어렵다
# 일반화 성능 평가를 위해서 교차 검증을 사용하여 각 매개 변수의 조합의 성능을 평가할 수 있다.

# Load Data
iris = datasets.load_iris()

# Split Data
x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.25, random_state=0)

# Cross validaion + Grid Search
# Use SVM
best_score = 0

for gamma in [0.001, 0.01, 0.1, 1, 10, 100]:
    for C in [0.001, 0.01, 0.1, 1, 10, 100]:

        # Train SVM using gamma & C
        svm_clf = svm.SVC(gamma = gamma, C = C)

        # Evaluate model
        scores = cross_val_score(svm_clf, x_train, y_train, cv = 5)


        # Mean of CV
        score = np.mean(scores)

        print("gamma : ",gamma, ' C : ', C, " score : ", score)
        # Restore the highest score with its parameter
        if score > best_score:
            best_score = score
            best_param = {'gamma':gamma, 'C':C}


svm_clf = svm.SVC(**best_param)
svm_clf.fit(x_train, y_train)
train_score = svm_clf.score(x_train,y_train)
test_score = svm_clf.score(x_test, y_test)
print("train_score : " , train_score , " test_score : ", test_score)





# 2. GridSearchCV
# 교차 검증을 사용한 그리드 서치를 매개변수 조정 방법으로 널리 사용하므로
# sklearn에서는 GridSearchCV를 제공
# 검색 대상이 되는 매개변수를 dict 형식의 자료형으로 구성
from sklearn import datasets
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
# Load Data
iris = datasets.load_iris()
# 파라미터 그리드 생성
param_grid = {"C" : [0.001, 0.01, 0.1, 1, 10, 100], "gamma" : [0.001,0.01,0.1,1,10,100]}
clf = svm.SVC()

# GridSearchCV 객체 생성
grid_search = GridSearchCV(svm.SVC(), param_grid, cv = 5, return_train_score=True)

# 데이터 분할
# 데이터가 과적합 되는 것을 방지하기위해, 또는 최종 모델의 객관적인 정확도 평가를 위해 test data를 분리하고
# gridseach 및 cross-validation에 사용 되지 않도록 한다.
x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.25, random_state=0)

# gridSearch model
grid_search.fit(x_train,y_train)

# model evaluation
print('test set score : {}'.format(grid_search.score(x_test,y_test)))
# 교차 검증과 그리드 서치의 결과로 산정한 최적의 매개변수를 적용하여 전체 훈련 데이터 셋에 대해
# 훈련한 최종 모델에 테스트 데이터로 적용했을때 결과

# best parameter and best score
print("best parameter : {}".format(grid_search.best_params_))
print("best score : {}".format(grid_search.best_score_))
# 훈련 세트에서 수행한 교차검증의 평균 정확도의 score



# RandomSearchCV
# gridSearchCV화 활용방법은 동일
# 순차적으로 파라키터를 찾는 것이아니라 랜덤진행으로 파라미터튜닝
from sklearn import datasets
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
# Load Data
iris = datasets.load_iris()

# 파라미터 그리드 생성
random_param_grid = {"C" : [0.001, 0.01, 0.1, 1, 10, 100], "gamma" : [0.001,0.01,0.1,1,10,100]}
random_search = RandomizedSearchCV(clf, random_param_grid, cv = 5, return_train_score=True)

x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.25, random_state=0)
random_search.fit(x_train, y_train)
# model evaluation
print('test set score : {}'.format(random_search.score(x_test,y_test)))
# 교차 검증과 그리드 서치의 결과로 산정한 최적의 매개변수를 적용하여 전체 훈련 데이터 셋에 대해
# 훈련한 최종 모델에 테스트 데이터로 적용했을때 결과

# best parameter and best score
print("best parameter : {}".format(random_search.best_params_))
print("best score : {}".format(random_search.best_score_))
# 훈련 세트에서 수행한 교차검증의 평균 정확도의 score