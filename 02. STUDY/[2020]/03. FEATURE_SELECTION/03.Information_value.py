# Import packages
import pandas as pd
import numpy as np
import pandas.core.algorithms as algos
from pandas import Series
import scipy.stats.stats as stats
import re
import traceback
import warnings
warnings.filterwarnings('ignore')

MAX_BIN = 20
FORCE_BIN = 3



# Define a binning function for continuous independent variables
def mono_bin(Y, X, n = MAX_BIN):
    # 타겟, 연속형 변수 데이터 프레임 생성
    df1 = pd.DataFrame({"X": X,"Y": Y})

    # null 인경우 아닌 경우로 데이터 분리
    justmiss = df1[['X','Y']][df1.X.isnull()]
    notmiss = df1[['X', 'Y']][df1.X.notnull()]

    r = 0
    while np.abs(r) < 1:
        try:
            # null이 아닌 독립변수와, null이 아닌타겟을, 20개의 그룹으로 데이터를 분할
            d1 = pd.DataFrame({"X": notmiss.X , "Y": notmiss.Y, "Bucket" : pd.qcut(notmiss.X, n)})
            #print(d1.head())
            d2 = d1.groupby("Bucket", as_index=True)
            # 이부분에 대한 이해  필요
            # 스피어만 상관계수를 통해 그룹간 순서의 연관성 여부 판단
            r, p = stats.spearmanr(d2.mean().X, d2.mean().Y)
            print(r, p)
            n = n - 1
            print(n)
        except Exception as e:
            n = n-1
            print(n)

    if len(d2) == 1:
        n = FORCE_BIN
        bins = algos.quantile(notmiss.X, np.linspace(0,1,n))
        if len(np.unique(bins)) == 2:
            bins = np.insert(bins, 0 ,1)
            bins[1] = bins[1] - (bins[1]/2)
        d1 = pd.DataFrame({"X":notmiss.X, "Y": notmiss.Y, "Bucket" : pd.cut(notmiss.X, np.unique(bins), include_lowest=True)})
        d2 = d1.groupby("Bucket", as_index=True)

    d3 = pd.DataFrame({}, index=[])
    d3["MIN_VALUE"] = d2.min().X
    d3["MAX_VALUE"] = d2.max().X
    d3["COUNT"] = d2.count().Y
    d3["EVENT"] = d2.sum().Y
    d3["NONEVENT"] = d2.count().Y - d2.sum().Y
    d3 = d3.reset_index(drop=True)


    if len(justmiss.index) > 0 :
        d4 = pd.DataFrame({"MIN_VALUE":np.nan}, index =[0])
        d4["MAX_VALUE"] = np.nan
        d4["COUNT"] = justmiss.count().Y
        d4["EVENT"] = justmiss.sum().Y
        d4["NONEVENT"] = justmiss.count().Y - justmiss.sum().Y
        d3 = d3.append(d4, ignore_index=True)

    d3["EVENT_RATE"] = d3.EVENT/d3.COUNT
    d3["NON_EVENT_RATE"] = d3.NONEVENT/d3.COUNT
    d3["DIST_EVENT"] = d3.EVENT/d3.sum().EVENT
    d3["DIST_NON_EVENT"] = d3.NONEVENT/d3.sum().NONEVENT
    d3["WOE"] = np.log(d3.DIST_EVENT/d3.DIST_NON_EVENT)
    d3["IV"] = (d3.DIST_EVENT - d3.DIST_NON_EVENT) * np.log(d3.DIST_EVENT/d3.DIST_NON_EVENT)
    d3["VAR_NAME"] = "VAR"
    d3 = d3[["VAR_NAME" , "MIN_VALUE", "MAX_VALUE", "COUNT", "EVENT", "EVENT_RATE", "NONEVENT" ,
             "NON_EVENT_RATE", "DIST_EVENT", "DIST_NON_EVENT" ,"WOE", "IV"]]
    d3 = d3.replace([np.inf, -np.inf], 0)
    d3.IV = d3.IV.sum()

    return d3


# Define a binning function for categorical independent variables
def char_bin(Y,X) :

    df1 = pd.DataFrame({"X":X, "Y":Y})
    justmiss = df1[['X','Y']][df1.X.isnull()]
    notmiss = df1[['X','Y']][[df1.X.notnull()]]
    df2 = notmiss.groupby('X',as_index=True)

    d3 = pd.DataFrame({}, index=[])
    d3["COUNT"] = df2.count().Y
    d3["MIN_VALUE"] = df2.sum().Y.index
    d3["MAX_VALUE"] = d3["MIN_VALUE"]
    d3["EVENT"] = df2.sum().Y
    d3["NONEVENT"] = df2.count().Y - df2.sum().Y

    if len(justmiss.index) > 0 :
        d4 = pd.DataFrame({"MIN_VALUE":np.nan}, index=[0])
        d4["MAX_VALUE"] = np.nan
        d4["COUNT"] = justmiss.count().Y
        d4["EVENT"] = justmiss.sum().Y
        d4["NONEVENT"] = justmiss.count().Y - justmiss.sum().Y
        d3 = d3.append(d4, ignore_index=True)

    d3["EVENT_RATE"] = d3.EVENT/d3.COUNT
    d3["NON_EVENT_RATE"] = d3.NONEVENT/d3.COUNT
    d3["DIST_EVENT"] = d3.EVENT/d3.sum().EVENT
    d3["DIST_NON_EVENT"] = d3.NONEVENT/d3.sum().NONEVENT
    d3["WOE"] = np.log(d3.DIST_EVENT/d3.DIST_NON_EVENT)
    d3["IV"] = (d3.DIST_EVENT-d3.DIST_NON_EVENT)*np.log(d3.DIST_EVENT/d3.DIST_NON_EVENT)
    d3["VAR_NAME"] = "VAR"
    d3 = d3[["VAR_NAME","MIN_VALUE", "MAX_VALUE","COUNT","EVENT","EVENT_RATE","NONEVENT","NON_EVENT_RATE",
            "DIST_EVENT","DIST_NON_EVENT","WOE","IV"]]
    d3 = d3.replace([np.inf, -np.inf],0)
    d3.IV = d3.IV.sum()
    d3 = d3.reset_index(drop=True)

    return d3


def data_vars(df1, target):

    stack = traceback.extract_stack()
    filename, lineno, function_name , code = stack[-2]
    vars_name = re.compile(r'\((.*?)\).*$').search(code).groups()[0]
    final = (re.findall(r"[\w']+", vars_name))[-1]

    x = df1.dtypes.index
    count = -1

    for i in x:
        print(i)
        if i.upper() not in (final.upper()):
            if np.issubdtype(df1[i], np.number) and len(Series.unique((df1[i]))) > 2:
                conv = mono_bin(target, df1[i])
                conv["VAR_NAME"] = i
                count = count +1
            else:
                conv = char_bin(target, df1[i])
                conv["VAR_NAME"] = i
                count = count + 1

            if count == 0 :
                iv_df = conv
            else:
                iv_df = iv_df.append(conv, ignore_index=True)

    iv = pd.DataFrame({"IV":iv_df.groupby("VAR_NAME").IV.max()})
    iv = iv.reset_index()

    return iv_df, iv


if __name__ == "__main__":
    # data load
    df = pd.read_csv("D:/git_study/Haram/hospital-open-close/data/train_mod_example.csv", index_col=False, low_memory=False)
    df = df.set_index('inst_id')
    df.drop(["score"],inplace=True, axis=1)


    final_iv, IV = data_vars(df, df["OC"])

    final_iv.to_csv("D:\\99.study\\Reg_Feauture\\final_iv.csv",index=False)
    IV.to_csv("D:\\99.study\\Reg_Feauture\\IV.csv", index=False)

