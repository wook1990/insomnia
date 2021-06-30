import pandas as pd
import pycaret as pc

df = pd.read_csv("e:/train_data_fn.csv",index_col=False)
target_data = pd.read_csv("e:/AIchamp/validation.csv", index_col=False, low_memory=False)

target_data.drop("ord_dt_ch", axis=1,inplace=True)

# 데이터 분할
df_train = df.loc[df.ord_dt <= '2020-09-08']
df_unseen = df.loc[df.ord_dt > '2020-09-08']
df_train.to_csv("e:/train_input_data.csv",index=False,encoding="utf-8")
df_unseen.to_csv("e:/test_data.csv",index=False,encoding="utf-8")

# 354,586
# 정답 데잍 ㅓ분할
train_target_df = target_data[["shop_no","ord_dt","abuse_yn"]].groupby(["shop_no","ord_dt"]).max()
train_target_df.reset_index(inplace=True)
train_target_df.to_csv("e:/answer_data.csv",index=False,encoding="utf-8")


#####################################################################

test_data = pd.read_csv("e:/AIchamp/test_data.csv")
test_data.ord_dt.min()

test_anws = pd.read_csv("e:/AIchamp/validation.csv")
validation = test_anws[["shop_no","ord_dt","abuse_yn"]].groupby(["shop_no","ord_dt"]).max()

validation.reset_index(inplace=True)
prid_data = validation.loc[validation.ord_dt > "2020-09-08"]
# 학습데이터 정답 병합
# train_df = pd.merge(df_train, train_target_df, on=["shop_no","ord_dt"], how="left")
predict_df = pd.merge(test_data, prid_data, on=["shop_no","ord_dt"], how="left")
predict_df.to_csv("e:/AIchamp/predict_df.csv",encoding="utf-8", sep= ',')
# 서버 Dask 사용해서 해결
#########################################################################################
import pandas as pd
import numpy as np

from xgboost import *
from sklearn import *
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, cross_val_predict, cross_val_score
from sklearn.model_selection import StratifiedKFold
from imblearn.over_sampling import SMOTE

PATH = "C:/Users/wook1/Documents/WAI/2020/02.K스타트업_빅데이터경진대회/개발/ai_dat/final/"


train_df = pd.read_csv(PATH + "small_train_data.csv")
train_df = train_df.set_index(["shop_no","ord_dt"])
train_df.drop("Unnamed: 0",axis=1,inplace=True)
train_df = train_df.fillna(0)
feature, target = train_df.drop("abuse_yn",1), train_df["abuse_yn"]
x_train, x_test,y_train,y_test = train_test_split(feature, target, test_size=0.3, random_state=7)
len(train_df.columns)
x_train
random_clf =  RandomForestClassifier(
    n_estimators=50,
    criterion='gini',
    max_depth=5,
    min_samples_split=2,
    min_samples_leaf=1,
    min_weight_fraction_leaf=0.0,
    max_features='auto',
    max_leaf_nodes=None,
    min_impurity_decrease=0.0,
    min_impurity_split=None,
    bootstrap=True,
    oob_score=False,
    n_jobs=-1,
    random_state=0,
    verbose=0,
    warm_start=False,
    class_weight='balanced'
)

random_clf.fit(x_train, y_train)

print("train 데이터에 대한 성능 점수 : " ,random_clf.score(x_train,y_train))
# train 데이터에 대한 성능 점수 :  0.9991457381525236
print("검증 데이터의 성능 점수 : " ,random_clf.score(x_test,y_test))
# 검증 데이터의 성능 점수 :  0.9990285995433156

y_hat_score = random_clf.predict_proba(x_test)[:,1]
y_hat_score > 0.3
# [<the_expression> if <the_condition> else <other_expression> for <the_element> in <the_iterable>]
y_hat_score_list = [1 if i >=0.3 else 0 for i in y_hat_score ]
y_hat = random_clf.predict(x_test)

random_clf.feature_importances_


# predict data
pred_df = pd.read_csv(PATH +"small_predict_data.csv")
pred_x = pred_df.drop("abuse_yn",1)
pred_x = pred_x.set_index(["shop_no","ord_dt"])
pred_x.drop('Unnamed: 0',axis=1,inplace=True)
pred_x.drop('Unnamed: 0.1',axis=1,inplace=True)
pred_x = pred_x.fillna(0)
pred_yhat_score = random_clf.predict_proba(pred_x)[:,1]

pred_yhat_score_list = [1 if i >=0.3 else 0 for i in pred_yhat_score]


pred_df = pred_df.reset_index()
predict_df = pred_df[["shop_no","ord_dt"]]
predict_df["abuse_yn"] = pred_yhat_score_list
predict_df.columns = ["shop_no","ord_dt","pred_score"]

predict_df.to_csv(PATH + "predict_data.csv", index=False, encoding="utf-8")

def print_score(classifier,x_train, y_train, x_test,y_test, train=True):
    if train == True:
        print("Traiining result : \n")
        print('Accuracy Score: {0:.4f}\n'.format(metrics.accuracy_score(y_train,classifier.predict(x_train))))
        print('f1 Score : {0:4f}\n'.format(metrics.f1_score(y_train, classifier.predict(x_train))))
        print("Classification Report : \n {}\n".format(metrics.classification_report(y_train,classifier.predict(x_train))))
        print('Confusion Matrix: \n{}\n'.format(metrics.confusion_matrix(y_train,classifier.predict(x_train))))
        res = cross_val_score(classifier, x_train, y_train, cv=10, n_jobs=1,  scoring='f1')
        print('Average f1:\t{0:4f}\n'.format(res.mean()))
        print('Standard Deviation : \t{0:4f}\t'.format(res.std()))
    elif train == False:
        print("Test Result:\n")
        print('Accuracy Score : {0:.4f}\n'.format(metrics.accuracy_score(y_test,classifier.predict(x_test))))
        print('f1 Score : {0:4f}\n').format(metrics.f1_score(y_test,classifier.predict(x_test)))
        print('Classification Report:\n{}|n'.format(metrics.classification_report(y_test,classifier.predict(x_test))))
        print('confusion Matrix : \n{}\n'.format(metrics.confusion_matrix(y_test,classifier.predict(x_test))))

print_score(random_clf, x_train,y_train, x_test, y_test , train= True)
print_score(random_clf, x_train,y_train, x_test, y_test , train= False)


##############################################
# SMOTE smapling
sm= SMOTE(ratio='auto',kind='regular')
x_resampled, y_resampled = sm.fit_sample(feature, list(target))

##############################################
## Hyperparameter Tuned#
# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]
# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}

rf_random_CV = RandomizedSearchCV(estimator=random_clf, param_distributions=random_grid, n_ter=100, cv=3,
                                  verbose=2, random_state=42, n_jobs=-1)

# Fit the random search model
rf_random_CV.fit(x_train, y_train)








######************---- 별첨 ------************#######################
#############################데이터셋 축소 ############################
import re

import pandas as pd
# 훈련데이터
df = pd.read_csv("/workspace/train_final_data_x.csv")

col_list = df.columns
len(col_list)
tm_list = [word for word in col_list if "tm" in word]
print(tm_list)

rm_list=[]
for i in tm_list:
    if re.match("(^tm_)", i):
        rm_list.append(i)

rm_4tm_list=[]
for i in tm_list:
    if re.match("(tm4_)", i):
        rm_4tm_list.append(i)