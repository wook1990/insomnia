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


def load_models(p_path_for_saving_model):
    list_models = list()
    files_model = glob.glob(p_path_for_saving_model + 'train_*.model')
    for file_name in files_model:
        model_name = file_name.split('train_')[1].replace('.model', '')
        with open(file_name, 'rb') as model_file:
            model = joblib.load(model_file)
            list_models.append({
                'name': model_name,
                'model': model,
            })
    return list_models


def execute_scoring(p_models, p_score_file_name, p_path_for_saving_model):

    df_score_x = pandas.read_csv(p_score_file_name)
    #df_score_x = p_score_file_name
    flag_assessment = False
    list_assessment = list()
    try:
        del df_score_x['target']
    except:
        pass
    df_result = df_score_x[['raw_id']]
    df_score_x = df_score_x.set_index(['raw_id'])
    for model_info in p_models:
        model_instance = model_info['model']
        model_name = model_info['name']

        print(datetime.datetime.today().strftime('[%Y/%m/%d %H:%M:%S]'), 'Scoring ', model_info['name'], 'is started.', flush=True)
        try:
            y_predict = model_instance.predict_proba(df_score_x)[:,1]
        except:
            y_predict = model_instance.decision_function(df_score_x)
        df_result['score'] = y_predict
        df_result.to_csv(p_path_for_saving_model + 'score_' + model_info['name'] + '.csv', index=None)

        print(datetime.datetime.today().strftime('[%Y/%m/%d %H:%M:%S]'), 'Scoring ', model_info['name'], 'is finished.', flush=True)


if __name__ == '__main__':
    warnings.filterwarnings('ignore')

    #score_file_name = sys.argv[1]
    #path_for_saving_model = sys.argv[2]

    score_file_name = "C:/Users/WAI/Documents/kogas/data/kogas_opt_model.csv"
    path_for_saving_model = "C:/Users/WAI/Documents/kogas/models/"
    if path_for_saving_model[-1:] != '/':
        path_for_saving_model += '/'

    print(datetime.datetime.today().strftime('[%Y/%m/%d %H:%M:%S]'), 'START', flush=True)
    execute_scoring(load_models(path_for_saving_model), score_file_name, path_for_saving_model)
    print(datetime.datetime.today().strftime('[%Y/%m/%d %H:%M:%S]'), 'FINISH', flush=True)
