import warnings
import os
import sys
import datetime
import pandas
import numpy
import glob
import joblib

from sklearn import *
from xgboost import *
from sklearn.preprocessing import MinMaxScaler
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

fixed_kernel = RBF(length_scale=1.0, length_scale_bounds='fixed')
kernels_gpc = [
    RBF(length_scale=0.1),
    fixed_kernel,
    RBF(length_scale=1.0, length_scale_bounds=(1e-3, 1e3)),
    C(1.0, (1e-2, 1e2)) * RBF(length_scale=1.0, length_scale_bounds=(1e-3, 1e3))
]


def create_models():
    list_models = list()
    model = linear_model.RidgeClassifier(fit_intercept=True, random_state=0, tol=0.0001, max_iter=10000, solver='auto', alpha=0.05)
    list_models.append({
        'name': 'linear_model.RidgeClassifier',
        'model': model,
    })
    model = tree.DecisionTreeClassifier(splitter='best', random_state=0, criterion='gini', max_depth=10, min_samples_split=10, min_samples_leaf=3)
    list_models.append({
        'name': 'tree.DecisionTreeClassifier',
        'model': model,
    })
    model = tree.ExtraTreeClassifier(splitter='best', random_state=0, criterion='gini', max_depth=10, min_samples_split=50, min_samples_leaf=3)
    list_models.append({
        'name': 'tree.ExtraTreeClassifier',
        'model': model,
    })
    model = linear_model.PassiveAggressiveClassifier(fit_intercept=True, random_state=0, warm_start=False, loss='hinge', C=1.0, max_iter=30000)
    list_models.append({
        'name': 'linear_model.PassiveAggressiveClassifier',
        'model': model,
    })
    model = ensemble.AdaBoostClassifier(random_state=0, n_estimators=100, algorithm='SAMME.R', learning_rate=0.5)
    list_models.append({
        'name': 'ensemble.AdaBoostClassifier',
        'model': model,
    })
    model = ensemble.ExtraTreesClassifier(random_state=0, n_estimators=100, criterion='gini', max_depth=100, min_samples_split=2, min_samples_leaf=1)
    list_models.append({
        'name': 'ensemble.ExtraTreesClassifier',
        'model': model,
    })
    model = ensemble.RandomForestClassifier(random_state=0, n_estimators=100, criterion='entropy', max_depth=50, min_samples_split=2, min_samples_leaf=2)
    list_models.append({
        'name': 'ensemble.RandomForestClassifier',
        'model': model,
    })
    model = linear_model.SGDClassifier(fit_intercept=True, random_state=0, warm_start=False, power_t=0.5, penalty='elasticnet', loss='squared_hinge', eta0=0.1, learning_rate='optimal', alpha=1e-05, l1_ratio=0.001, epsilon=0.05, max_iter=10000)
    list_models.append({
        'name': 'linear_model.SGDClassifier',
        'model': model,
    })
    model = XGBClassifier(verbosity=1, booster='gbtree', reg_alpha=0.5)
    list_models.append({
        'name': 'XGBClassifier',
        'model': model,
    })
    model = neural_network.MLPClassifier(power_t=0.5, random_state=0, activation='relu', solver='adam', max_iter=1000, hidden_layer_sizes=(10, 10, 10, 10), alpha=1e-05, learning_rate='adaptive')
    list_models.append({
        'name': 'neural_network.MLPClassifier',
        'model': model,
    })
    return list_models


def train_model(p_models, p_train_file_name, p_path_for_saving_model):
    df_train_x = pandas.read_csv(p_train_file_name, dtype={'raw_id':object})
    df_train_x = df_train_x.set_index(['raw_id'])

    df_train_x.loc[df_train_x['target'] < 1.0, 'target'] = 0.0
    df_train_x.loc[df_train_x['target'] >= 1.0, 'target'] = 1.0
    df_train_x['target'] = df_train_x['target'].astype(float)

    df_train_y = df_train_x['target']
    del df_train_x['target']
    for model_info in p_models:
        model_instance = model_info['model']
        model_file_name = p_path_for_saving_model + 'train_' + model_info['name'] + '.model'

        print(datetime.datetime.today().strftime('[%Y/%m/%d %H:%M:%S]'), 'Training ', model_info['name'], 'is started.', flush=True)
        model_instance.fit(df_train_x, df_train_y)
        with open(model_file_name, 'wb') as f:
            joblib.dump(model_instance, f)
        print(datetime.datetime.today().strftime('[%Y/%m/%d %H:%M:%S]'), 'Training ', model_info['name'], 'is finished.', flush=True)


if __name__ == '__main__':
    warnings.filterwarnings('ignore')

    #train_file_name = sys.argv[1]
    #path_for_saving_model = sys.argv[2]

    train_file_name = "C:/Users/WAI/Documents/kogas/data/kogas_train_model.csv"
    path_for_saving_model = "C:/Users/WAI/Documents/kogas/models/"
    if path_for_saving_model[-1:] != '/':
        path_for_saving_model += '/'

    print(datetime.datetime.today().strftime('[%Y/%m/%d %H:%M:%S]'), 'START', flush=True)
    train_model(create_models(), train_file_name, path_for_saving_model)
    print(datetime.datetime.today().strftime('[%Y/%m/%d %H:%M:%S]'), 'FINISH', flush=True)
