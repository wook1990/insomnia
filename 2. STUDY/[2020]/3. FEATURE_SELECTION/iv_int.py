# Import packages
import pandas as pd
import numpy as np
import pandas.core.algorithms as algos
from pandas import Series
import scipy.stats.stats as stats
import warnings
warnings.filterwarnings('ignore')


MAX_BIN = 20
FORCE_BIN = 3

# 이산형분류 문제 예시 데이터
file_path = "D:/git_study/Haram/hospital-open-close/data/train_mod_example.csv"
df1 = pd.read_csv(file_path, low_memory=False)
df1 = df1.set_index("inst_id")
df1.drop(["score"], axis = 1, inplace=True)
target = df1["OC"]

"""
# 이부분의 역할 정의 필요
# LIFO 구조의 스택사용
# excetion 처리를 위한 trackback 실행 스택 관리
stack = traceback.extract_stack()
filename, lineno, function_name , code = stack[-2]
vars_name = re.compile(r'\((.*?)\).*$').search(code).groups()[0]
final = (re.findall(r"[\w']+", vars_name))[-1]
"""
# 변수명 추출
x = df1.dtypes.index
count = -1

# 독립변수별 종속변수와 iv, woe 계산
for i in x:
    print(i)
    #if i.upper() not in (final.upper()):
    # 연속형, 범주형 분리 조건
    # 변수의 데이터 타입이 number이고 유니크한 값의 길이가 2초과인 경우로 분리
    if np.issubdtype(df1[i], np.number) and len(Series.unique((df1[i]))) > 2:
        print("mono_bin : " + i)
        # 일대일 변수 영향도를 측정하는 지표이므로
        # 독립변수 한개와 종속변수 한개에 대한 데이터 프레임 생성
        df_mono = pd.DataFrame({"X" : df1[i], "Y": target})

        # null이 존재하는 데이터와 존재하지 않는 데이터 셋으로 분리
        justmiss = df_mono[['X', 'Y']][df_mono.X.isnull()]
        notmiss = df_mono[['X', 'Y']][df_mono.X.notnull()]

        # 맨 위에 선언된 bin의 크기만큼 데이터 분할후 분할된 범주의 스피어만상관계수를 확인하여
        # 상관계수가 절대값 1에 가까운 범주크기로 분할
        # null 이아닌 경우에 대한 데이터 분할
        # WOE는 결측치에 대해서 다른 그룹으로 구분하여 계산하기 때문에
        r = 0
        while np.abs(r) < 1:
            try:
                # null이 아닌 독립변수와, null이 아닌타겟을 20개의 그룹으로 데이터 분할
                d1 = pd.DataFrame({"X": notmiss.X, "Y" : notmiss.Y, "Bucket" : pd.qcut(notmiss.X, MAX_BIN)})
                # 분할된 범주로 그룹핑
                d2 = d1.groupby("Bucket", as_index=True)

                # 그룹핑이 가능해지면
                # 범주의 독립변수 평균과 종속변수의 평균의 스피어만 순위 상관계수를 비교하여
                # 최적의 binning 구간 찾기
                r, p = stats.spearmanr(d2.mean().X, d2.mean().Y)
                print(r, p)
                MAX_BIN = MAX_BIN - 1
                print(MAX_BIN)
            except Exception as e:
                # 그룹핑이 가능한 bin이 되도록 bin 범위 축소
                MAX_BIN = MAX_BIN-1
                print(MAX_BIN)

        # 범주가 하나인 경우
        if len(d2) == 1:
            # 강제적으로 최소 3개의 빈 선언
            n = FORCE_BIN
            # 0,1사이에 3개의 점을 직어 선형분할 bins생성
            bins = algos.quantile(notmiss.X, np.linspace(0,1,n))
            # bin의 길이가 2인경우
            if len(np.unique(bins)) == 2:
                # bin array 0번째 자리에 1을 삽입
                bins = np.insert(bins,0,1)
                bins[1] = bins[1] - (bins[1]/2)

            d1 = pd.DataFrame(
                {"X": notmiss.X, "Y": notmiss.Y, "Bucket": pd.cut(notmiss.X, np.unique(bins), include_lowest=True)})
            d2 = d1.groupby("Bucket", as_index=True)
        # 범주별 최대 최소 갯수 이벤트 비이벤트 갯수 생성하여 결과 저장
        d3 = pd.DataFrame({}, index=[])
        d3["MIN_VALUE"] = d2.min().X
        d3["MAX_VALUE"] = d2.max().X
        d3["COUNT"] = d2.count().Y
        d3["EVENT"] = d2.sum().Y
        d3["NONEVENT"] = d2.count().Y - d2.sum().Y
        d3 = d3.reset_index(drop=True)

        # null 값인 경우 각 케이스 값 추출
        if len(justmiss.index) > 0:
            d4 = pd.DataFrame({"MIN_VALUE": np.nan}, index=[0])
            d4["MAX_VALUE"] = np.nan
            d4["COUNT"] = justmiss.count().Y
            d4["EVENT"] = justmiss.sum().Y
            d4["NONEVENT"] = justmiss.count().Y - justmiss.sum().Y
            d3 = d3.append(d4, ignore_index=True)

        d3["EVENT_RATE"] = d3.EVENT / d3.COUNT
        d3["NON_EVENT_RATE"] = d3.NONEVENT / d3.COUNT
        d3["DIST_EVENT"] = d3.EVENT / d3.sum().EVENT
        d3["DIST_NON_EVENT"] = d3.NONEVENT / d3.sum().NONEVENT
        d3["WOE"] = np.log(d3.DIST_EVENT / d3.DIST_NON_EVENT)
        d3["IV"] = (d3.DIST_EVENT - d3.DIST_NON_EVENT) * np.log(d3.DIST_EVENT / d3.DIST_NON_EVENT)
        d3["VAR_NAME"] = "VAR"
        d3 = d3[["VAR_NAME", "MIN_VALUE", "MAX_VALUE", "COUNT", "EVENT", "EVENT_RATE", "NONEVENT",
                 "NON_EVENT_RATE", "DIST_EVENT", "DIST_NON_EVENT", "WOE", "IV"]]
        d3 = d3.replace([np.inf, -np.inf], 0)
        d3.IV = d3.IV.sum()

        # 결과 저장
        conv = d3
        conv["VAR_NAME"] = i
        count = count + 1

    else:
        # 범주형 변수 WOE, IV 계산 로직은 연속형 변수와 동일
        print("char_bin : " + i)
        df_char = pd.DataFrame({"X" : df1[i], "Y": target})
        justmiss = df_char[['X', 'Y']][df_char.X.isnull()]
        notmiss = df_char[['X', 'Y']][df_char.X.notnull()]
        df2 = notmiss.groupby('X', as_index=True)

        d3 = pd.DataFrame({}, index=[])
        d3["COUNT"] = df2.count().Y
        d3["MIN_VALUE"] = df2.sum().Y.index
        d3["MAX_VALUE"] = d3["MIN_VALUE"]
        d3["EVENT"] = df2.sum().Y
        d3["NONEVENT"] = df2.count().Y - df2.sum().Y

        if len(justmiss.index) > 0:
            d4 = pd.DataFrame({"MIN_VALUE": np.nan}, index=[0])
            d4["MAX_VALUE"] = np.nan
            d4["COUNT"] = justmiss.count().Y
            d4["EVENT"] = justmiss.sum().Y
            d4["NONEVENT"] = justmiss.count().Y - justmiss.sum().Y
            d3 = d3.append(d4, ignore_index=True)

        d3["EVENT_RATE"] = d3.EVENT / d3.COUNT
        d3["NON_EVENT_RATE"] = d3.NONEVENT / d3.COUNT
        d3["DIST_EVENT"] = d3.EVENT / d3.sum().EVENT
        d3["DIST_NON_EVENT"] = d3.NONEVENT / d3.sum().NONEVENT
        d3["WOE"] = np.log(d3.DIST_EVENT / d3.DIST_NON_EVENT)
        d3["IV"] = (d3.DIST_EVENT - d3.DIST_NON_EVENT) * np.log(d3.DIST_EVENT / d3.DIST_NON_EVENT)
        d3["VAR_NAME"] = "VAR"
        d3 = d3[["VAR_NAME", "MIN_VALUE", "MAX_VALUE", "COUNT", "EVENT", "EVENT_RATE", "NONEVENT", "NON_EVENT_RATE",
                 "DIST_EVENT", "DIST_NON_EVENT", "WOE", "IV"]]
        d3 = d3.replace([np.inf, -np.inf], 0)
        d3.IV = d3.IV.sum()
        d3 = d3.reset_index(drop=True)


        conv = d3
        conv["VAR_NAME"] = i
        count = count + 1

    if count == 0:
        iv_df = conv
    else:
        iv_df = iv_df.append(conv, ignore_index=True)

    MAX_BIN = 20

iv = pd.DataFrame({"IV": iv_df.groupby("VAR_NAME").IV.max()})
iv = iv.reset_index()

IV_DF = iv_df
IV = iv

IV_DF.to_csv("D:/git_study/Haram/Feature_select/final_iv.csv",index=False)
IV.to_csv("D:/git_study/Haram/Feature_select/iv.csv",index=False)
