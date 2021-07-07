
# Feature Selection에 들어가기 앞서 Data Preprocessing을 수행
# 0. Data Preprocessing

import pandas
import numpy
from sklearn.model_selection import train_test_split

paribas_data = pandas.read_csv("./data/3.FeatureSelection/01.Wrapper/train.csv", nrows = 20000)


print("----Raw Data Shape----")
print(paribas_data.shape)
# Numeric Columns 추출
num_columns = ['int16','int32', 'int64', 'float16', 'float32','float64']
numerical_columns = list(paribas_data.select_dtypes(include=num_columns).columns)
# 114개 연속형 변수만 추출
paribas_data = paribas_data[numerical_columns]
print("----Numerical Data Shape----")
print(paribas_data.shape)

# Split Train, Test Data Set
train_x, test_x, train_y, test_y = train_test_split(paribas_data.drop(labels=["ID","target"], axis=1),
                                                    paribas_data["target"],
                                                    test_size=0.2,
                                                    random_state=42)

# Correlation > |0.8| 인 컬럼 제거(Filter Method 사용)
correlated_features = set()
correlation_matrix = paribas_data.corr()

for idx, colname in enumerate(correlation_matrix):
    #print("-------")
    #print("{0} : {1}".format(idx, colname))
    for j in range(idx):
        #print(j)
        if abs(correlation_matrix.iloc[idx, j]) > 0.8:
            correlated_features.add(colname)

print(correlated_features)
print(len(correlated_features))

train_x.drop(labels=correlated_features, axis=1, inplace=True)
test_x.drop(labels=correlated_features, axis=1, inplace=True)

print("Train Shape : {0}, Test Shape : {1}".format(train_x.shape, test_x.shape))


# 2. Step Backward Feature Selection
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from mlxtend.feature_selection import SequentialFeatureSelector

feature_selector = SequentialFeatureSelector(RandomForestClassifier(n_estimators=10, n_jobs=-1),
                                             k_features=15,
                                             forward=False,
                                             verbose=2,
                                             scoring='roc_auc',
                                             cv=4)

features = feature_selector.fit(numpy.array(train_x.fillna(0)), train_y, custom_feature_names= train_x.columns)

filtered_features = train_x.columns[list(features.k_feature_idx_)]

# modeling
clf = RandomForestClassifier(n_estimators=10, random_state=41, max_depth=3)
clf.fit(train_x[filtered_features].fillna(0), train_y)

train_pred = clf.predict_proba(train_x[filtered_features].fillna(0))
print('Accuracy on training set: {}'.format(roc_auc_score(train_y, train_pred[:,1])))

test_pred = clf.predict_proba(test_y[filtered_features].fillna(0))
print('Accuracy on test set: {}'.format(roc_auc_score(test_y, test_pred [:,1])))





####################################################################
######--------------------------------------------------------######

# 3. Exhaustive Feature Selection

from mlxtend.feature_selection import ExhaustiveFeatureSelector
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

# Selector Algorithm
feature_selector = ExhaustiveFeatureSelector(RandomForestClassifier(n_jobs=-1),
                                             min_features = 1,
                                             max_features= 3,
                                             scoring = 'roc_auc',
                                             print_progress=True,
                                             cv=2)
# ExhaustiveFeatureSelector는 LogLevel도 없고 결과가 출력이 ??????
print("Seletor Fit")
# Selector Fit
features = feature_selector.fit(numpy.array(train_x.fillna(0)), train_y,custom_feature_names= train_x.columns)
print("select Feature adjust")
# Filtering Feature
filtered_features = train_x.columns[list(features.k_features_idx_)]



