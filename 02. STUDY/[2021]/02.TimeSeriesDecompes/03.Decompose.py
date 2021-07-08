import pandas
import numpy
from statsmodels.tsa.seasonal import STL

# target에 대한 시계열 분해
# Using STL in Statsmodel

DATA_PATH= "/data/2.TimeSeriesDecompose\\"
FILE_NAME="d1_202_train.csv"

def line_logging(*messages):
    import datetime
    import sys
    today = datetime.datetime.today()
    log_time = today.strftime('[%Y/%m/%d %H:%M:%S]')
    log = list()
    for message in messages:
        log.append(str(message))
    print(log_time + ':[' + ' '.join(log) + ']', flush=True)

def decompose_stl(index:str, target:str, period:int):
    '''
    기능 : STL LOWESS를 사용하여 시계열 요소 분해
    파라미터 : index : {string}, target : string, period : int
    index : Timeseries Index Column Name
    target : Target Column Name
    period : Use in STL Decompose Func
    '''

    df = pandas.read_csv(DATA_PATH + FILE_NAME, index_col=False)
    df.set_index(index, inplace=True)
    stats_df = df[target]

    #Decomposition
    stl_result = STL(stats_df, period).fit()
    stl_res_df = pandas.concat([stl_result.trend, stl_result.seasonal, stl_result.resid], axis=1)
    stl_df = pandas.merge(df, stl_res_df, left_index=True, right_index=True, how="left")
    stl_df.to_csv(DATA_PATH + "decompose_{0}days.csv".format(period))
    line_logging("\n",stl_df.head(1))

if __name__ == '__main__':

    period = [7, 14, 30, 60, 90, 120, 180, 365]
    for i in iter(period):
        decompose_stl("eod_date","target",i)
        line_logging("분해주기 {0}일 - 데이터 생성완료".format(i))



