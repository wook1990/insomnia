
# 데이터 준비
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split

boston = load_boston()
x_train, x_test, y_train, y_test = train_test_split(boston.data, boston.target, random_state=42)

from sklearn.tree import DecisionTreeRegressor
dt_rg = DecisionTreeRegressor(criterion='mse', splitter='best', max_depth=None, min_samples_split=2,
                              min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=None,
                              random_state=None, max_leaf_nodes=None, min_impurity_decrease=0.0,
                              min_impurity_split=None, presort='deprecated', ccp_alpha=0.0)
dt_rg.fit(x_train,y_train)
dt_rg.score(x_train,y_train)
print("Train R2 score : ", round(dt_rg.score(x_train,y_train),3))
# Train R2 score :  1.0
dt_rg.score(x_test, y_test)
print("Test R2 score : ", round(dt_rg.score(x_test,y_test),3))
# Test R2 score :  0.833
# 예측값 생성
y_hat = dt_rg.redict(x_test)
'''
array([18.5, 33.1, 15.2, 24.1, 19.4, 21. , 19.1, 16.7, 21.6, 21.2, 27.1,
       19.5,  8.5, 21. , 16.2, 25. , 20.5,  7.2, 50. , 13. , 23.3, 22.2,
       15.7, 22. , 13.1, 14.6, 22.7, 13.5, 16.7, 24.5, 27.1, 23.3, 17.8,
       19.9, 13.5, 15.6, 33.4, 19.3, 21.7, 24.7, 19.8, 32. , 50. , 16.2,
       22. , 13.1, 15.6, 24.1, 20. , 33.1, 22.9, 34.9, 16.6, 28.4, 44.8,
       22.2, 13.1, 22.8, 19.8, 22.5, 24.8, 33. , 30.1, 14.5, 26.6, 14.4,
       15.4, 23.2, 22.8, 19. , 22.6, 28.7,  8.4, 22.5, 24.5,  5. , 20.4,
       43.8, 10.2,  8.1, 22.6, 13.1, 18.7,  7.2, 20.3, 30.1, 14.1, 23. ,
       28.7, 18.1, 22.6,  8.5, 19.2, 17.5, 24.3, 18.4, 50. , 13.1, 13.5,
       10.2, 17.5, 24.5, 13.1, 20.4, 19.6, 19. , 20.6, 23. , 20.6, 28.7,
        7.2, 16.3, 23.2, 29.6, 29.6, 14.6, 50. , 14.8, 19.3, 22.3, 16.2,
       23.7,  8.5, 21.2, 24.1, 23.1, 28.7])
'''
from sklearn.metrics import mean_absolute_error, mean_squared_error
print("MAE : ",mean_absolute_error(y_test,y_hat), " MSE : " ,mean_squared_error(y_test,y_hat))
# MAE :  2.3642519685039365  MSE :  13.691666141732279

# 변수 중요도 산출
print("Feature Importance : \n",dt_rg.feature_importances_)
'''
Feature Importance : 
 [7.47064387e-02 1.46130375e-03 6.15234477e-03 1.88334696e-04
 5.33683764e-03 5.87099179e-01 1.52507219e-02 7.44219220e-02
 1.60312332e-03 6.76577516e-03 1.33808831e-02 1.26460737e-02
 2.00987062e-01]
'''
# 생성된 트리의 깊이
print("Tree Depth : ",dt_rg.get_depth())
# Tree Depth :  21
# 생덩된 트리의 leaf 수
print("Tree Leaf count : " , dt_rg.get_n_leaves())
# Tree Leaf count :  351


from sklearn.metrics import r2_score
sort_idx = x_train.ravel().argsort()
r2_train = r2_score(y_train, dt_rg.predict(x_train))
r2_test = r2_score(y_test, dt_rg.predict(x_test))


#------------------------------------------------------#

# 데이터 준비
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split

boston = load_boston()
x_train, x_test, y_train, y_test = train_test_split(boston.data, boston.target, random_state=42)

from sklearn.tree import ExtraTreeRegressor
ext_rg = ExtraTreeRegressor(criterion='mse', splitter='best', max_depth=None, min_samples_split=2,
                              min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=None,
                              random_state=None, max_leaf_nodes=None, min_impurity_decrease=0.0,
                              min_impurity_split=None,ccp_alpha=0.0)
ext_rg.fit(x_train,y_train)
ext_rg.score(x_train,y_train)
print("Train R2 score : ", round(ext_rg.score(x_train,y_train),3))
# Train R2 score :  1.0
ext_rg.score(x_test, y_test)
print("Test R2 score : ", round(ext_rg.score(x_test,y_test),3))
# Test R2 score :  0.808
# 예측값 생성
y_hat = ext_rg.predict(x_test)
'''
array([18.5, 32. , 15.2, 24.1, 19.1, 20.5, 19.3, 17.8, 21.4, 21.2, 19.5,
       27.1,  8.5, 20.5, 16.2, 23.9, 20.5,  7.2, 50. , 17.8, 22.1, 22.2,
       15.6, 22.8, 13.1, 14.5, 24.5, 14.1, 16.7, 24.3, 27.1, 23.1, 10.4,
       21.9, 13.8, 15.6, 33.4, 19.3, 20.4, 24.7, 19.8, 29.9, 50. , 22.8,
       22. , 13.1, 15.6, 23.7, 19.1, 35.1, 22.9, 36.1, 19.3, 29.9, 43.1,
       25. , 13.1, 22.8, 22. , 22.5, 24.5, 33. , 29.4, 18.2, 26.6, 14.3,
       15.4, 22.9, 22.8, 19. , 22. , 28.7,  8.3, 22.9, 28.1,  8.5, 20.2,
       43.8, 10.2,  8.1, 22. , 16.3, 18.7,  7.2, 20.3, 28.4, 14.1, 23.1,
       28.7, 18. , 19. ,  5.6, 19.2, 17.5, 23.3, 18.4, 50. , 16.3, 14.1,
       16.3, 19. , 26.4, 16.3, 20.4, 18.7, 19. , 19.6, 23. , 20.6, 28.7,
        7.2, 16.3, 22.9, 29.6, 28.7, 14.9, 50. , 13.2, 19.3, 22.3, 22.8,
       23.7,  5. , 21.2, 24.1, 23.1, 28.7])
'''
from sklearn.metrics import mean_absolute_error, mean_squared_error
print("MAE : ",round(mean_absolute_error(y_test,y_hat),3), " MSE : " ,round(mean_squared_error(y_test,y_hat),3))
# MAE :  2.865  MSE :  13.476

# 변수 중요도 산출
print("Feature Importance : \n",ext_rg.feature_importances_)
'''
Feature Importance : 
 [7.02161587e-02 1.72369431e-03 9.78951103e-03 4.16538033e-06
 7.07328871e-03 5.89106383e-01 1.49401024e-02 7.36841263e-02
 7.19238233e-04 3.91029210e-03 2.91906962e-02 9.25995193e-03
 1.90382392e-01]
'''
# 생성된 트리의 깊이
print("Tree Depth : ",ext_rg.get_depth())
# Tree Depth :  21
# 생덩된 트리의 leaf 수
print("Tree Leaf count : " , ext_rg.get_n_leaves())
# Tree Leaf count :  351


from sklearn.metrics import r2_score
sort_idx = x_train.ravel().argsort()
r2_train = r2_score(y_train, ext_rg.predict(x_train))
r2_test = r2_score(y_test, ext_rg.predict(x_test))

# ddd