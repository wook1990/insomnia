# Pandas Interpolate -> 단일 열에 대한, Series 대한 보간
# 필요 모듈 import
import pandas
import warnings
import scipy
import numpy

warnings.filterwarnings(action='ignore')


def line_logging(*messages):
    import datetime
    import sys
    today = datetime.datetime.today()
    log_time = today.strftime('[%Y/%m/%d %H:%M:%S]')
    log = []
    for message in messages:
        log.append(str(message))
    print(log_time + '::' + ','.join(log) + '')
    sys.stdout.flush()


# 데이터 보간 함수
def auto_interpolate(df: pandas.DataFrame, target_col: str, index_col: str, null_ratio: int, interp_method: str):

    """
    입력받은 데이터에서 null 포함된 비율 기준으로 데이터를 뽑아 데이터를 보간하는 함수
    :parameter df: pandas.DataFrame : Input data type DataFrame
    :parameter target_col:str : Target column name for removing
    :parameter index_col:str : Index column name
    :parameter null_ratio : int : Criteria for selecting data to interpolate
    :parameter interp_method : str : Data interpolation methodology
     Parameters in to scipy.interpolate.interp1d
      ‘linear’, ‘nearest’, ‘zero’, ‘slinear’, ‘quadratic’,
      ‘cubic’, ‘previous’, ‘next’, where ‘zero’, ‘slinear’, ‘quadratic’, ‘cubic’ 등선언
    TO-BE
    1. 맨처음 값이 Null인 경우 처리방법 추가
    2. 기초통계값을 이용한 보간처리 추가
    3. 영향도 분석 등등 추가
    """
    # null 값 확인
    df.set_index(index_col, inplace=True)
    target_y = df[target_col]
    df = df.drop(columns=[target_col], axis=0)
    total_null_ratio = (round(df.isnull().sum() / 301 * 100, 3))
    null_df = pandas.DataFrame(total_null_ratio, columns=["null_ratio"])
    # 기준 범위 내 null을 가지고 있는 데이터 보간 방법법
    list_intp = null_df.loc[(null_df["null_ratio"] != 0) & (null_df["null_ratio"] <= null_ratio)].index.to_list()
    for intp_col in iter(list_intp):
        # line_logging(intp_col)
        prep_x = df[intp_col].values
        if interp_method in ["polynomial", "spline"]:
            # 3차 다항, 스플라인 사용 그이상의 차수는 고려
            # To-Be
            intp_x = pandas.Series(prep_x).interpolate(interp_method, order=3)
            df[intp_col] = intp_x.values
        else:
            intp_x = pandas.Series(prep_x).interpolate(interp_method)
            df[intp_col] = intp_x.values

    # 2. 기준 비율 이상의 null을 가지고 있는 데이터 보간 방법
    check_col = null_df.loc[null_df["null_ratio"] > null_ratio].index.to_list()

    # TO_DO
    # 기준치 이상의 null을 보유한 컬럼의 정규성 검증
    for colname in iter(check_col):
        # line_logging(colname)
        if len(df) <= 2000:
            # Shapirowilk Test
            test_stats, p_val = scipy.stats.shapiro(df[colname].dropna().values)
        else:
            # Kolmogorov-Smirnov Test
            test_stats, p_val = scipy.stats.kstest(df[colname].dropna().values, 'norm')

        if p_val > 0.5:
            # TODO
            # 정규성을 따르지 않는 컬럼에 대한 처리
            # print("정규성을 따르지 않음")
            check_col.remove(colname)

        else:
            # null이 아닌 데이터의 간격이 균일한지
            not_null_dt = numpy.where(df[colname].notnull())
            sum_diff = 0
            for i in range(0, len(not_null_dt[0])):
                if i < len(not_null_dt[0]) - 1:
                    sum_diff += not_null_dt[0][i + 1] - not_null_dt[0][i]
            # ADI 의 개념 사용
            # Average Demand Interval ( 평균 수요 구간)
            # MAPE 오차율 비교하여 인덱스 차이 검증
            mape_interval = round(numpy.abs((len(df) / len(not_null_dt[0])) - (sum_diff / len(not_null_dt[0]))) / (
                        len(df) / len(not_null_dt[0])) * 100, 3)

            if mape_interval < 10:
                prep_x = df[colname].values
                if interp_method in ["polynomial", "spline"]:
                    # 3차 다항, 스플라인 사용 그이상의 차수는 고려
                    # To-Be
                    intp_x = pandas.Series(prep_x).interpolate(interp_method, order=3)
                    df[intp_col] = intp_x.values
                else:
                    intp_x = pandas.Series(prep_x).interpolate(interp_method)
                    df[intp_col] = intp_x.values
            else:
                # TODO
                # 등간격을 유지하지 않는 데이터의 처리
                df.drop(columns=colname, inplace=True, axis=1)

    df[target_col] = target_y
    return df
