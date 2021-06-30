import numpy as np
import pandas as pd
import warnings
from sklearn.linear_model import Lasso
from scipy import stats

warnings.filterwarnings(action="ignore")


# Data Load
# Change point
def _data_load(path):
    """
    Cautions
    1. Do not depend System Resource.
    2. Use File I/O, dask.DataFrame, pandas chunk
    :param path: path of train data set that consist raw columns and target
                 And it is only CSV(Comma-Seperated Values) file
    :return: DataFrame Type use Dask.DataFrame
    reference
    - https://docs.dask.org/en/latest/
    """
    return pd.read_csv(path, index_col=False)


# 1.랜덤 샘플링
def _random_sampling(df, size):
    sample_index = np.random.randint(0, len(df), size=size)
    rand_sample_df = df.loc[sample_index]

    return rand_sample_df


# 3. Lasso
# Change point
# How to find best parameter in lasso
def _lasso_feature_select(df, alpha):
    data = df.values
    y = data[:, -1]
    X = data[:, :-1]
    lasso_reg = Lasso(alpha=alpha)
    # lasso_reg = Lasso(alpha=0.01)
    lasso_reg.fit(X, y)
    coef = lasso_reg.coef_
    return coef


# coeffidence 의 T-test
def _t_test(coef_df):
    np.random.seed(100)
    col_list = []
    for col_name in coef_df.columns:
        test_df = coef_df[col_name]
        free_dgree = coef_df.iloc[0:, 2].size - 1
        rand_t = np.random.standard_t(df=free_dgree, size=30)
        result_ttest = stats.ttest_ind(test_df, rand_t, equal_var=False)
        if result_ttest[1] > 0.05:
            col_list.append(col_name)

    return pd.DataFrame(coef_df[col_list]).abs().sum()


if __name__ == "__main__":

    PATH = "G:/내 드라이브/WAI_WORK/Project_I/2020/03. 가스공사/data/train_set.csv"
    data = _data_load(PATH)
    colname = data.columns.tolist()

    colname.remove('target')
    colname.append('target')

    coef_list = []

    for i in range(0, 30):
        r_sample_df = _random_sampling(data, 600)
        var_coef = _lasso_feature_select(r_sample_df, 0.01)
        coef_list.append(var_coef)

    colname.remove('target')
    coef_df = pd.DataFrame(coef_list)
    coef_df.columns = colname

    _t_test(coef_df)
