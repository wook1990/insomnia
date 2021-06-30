# 필수 라이브러리 임포트

import pandas
import warnings

warnings.filterwarnings(action="ignore")
DATA_PATH = "/data/2.TimeSeriesDecompose\\"

def data_merge(code_equity):
    # 종목 코드
    code_equity = code_equity
    # 시장 구분 코드
    code_market = 'DJI@DJI'
    name_market = code_market.replace('@', '_')
    # 수집 데이터 로딩
    df_loaded_price = pandas.read_csv(DATA_PATH + 'd1_101_price_' + code_equity + '.csv')
    df_loaded_price = df_loaded_price.set_index(['eod_date'])
    del df_loaded_price['item_code']
    df_loaded_price.columns = [code_equity + '-' + str(col) for col in df_loaded_price.columns.values]

    df_loaded_group = pandas.read_csv(DATA_PATH +  'd1_102_group_' + code_equity + '.csv')
    df_loaded_group = df_loaded_group.set_index(['eod_date'])
    del df_loaded_group['item_code']
    df_loaded_group.columns = [code_equity + '-' + str(col) for col in df_loaded_group.columns.values]

    df_loaded_market = pandas.read_csv(DATA_PATH + 'd1_103_market_' + name_market + '.csv')
    df_loaded_market = df_loaded_market.set_index(['eod_date'])
    del df_loaded_market['item_code']
    df_loaded_market.columns = [name_market + '-' + str(col) for col in df_loaded_market.columns.values]

    # 데이터 결합
    df_train = df_loaded_price.join(df_loaded_group, how='inner')
    df_train = df_train.join(df_loaded_market, how='left')
    df_train.to_csv(DATA_PATH + 'd1_201_preprocess.csv')

def make_target():

    # 데이터 로딩 및 타겟 생성
    df_train = pandas.read_csv(DATA_PATH + 'd1_201_preprocess.csv')
    df_train = df_train.set_index(['eod_date'])
    df_train = df_train.sort_index(ascending=True)
    df_train['target'] = df_train['005930-price_close'].shift(-1)

    # 5년 정도의 데이터 생성
    df_train = df_train.tail(1305)
    list_train_columns = list(df_train.columns)
    # Null 컬럼들의 바로 위/아래 행 값을 세팅
    df_count_null = df_train.isnull().sum().reset_index()
    df_count_null.columns = ['col_name', 'null_count']
    list_null = df_count_null[df_count_null['null_count'] != 0]['col_name'].tolist()
    for col_name in list_null:
        if col_name != 'target':
            df_train[col_name + '_shift-1'] = df_train[col_name].shift(-1)
            df_train[col_name + '_shift-1'] = df_train[col_name + '_shift-1'].astype(float)

            df_train[col_name + '_shift+1'] = df_train[col_name].shift(1)
            df_train[col_name + '_shift+1'] = df_train[col_name + '_shift+1'].astype(float)
    df_train.to_csv(DATA_PATH + 'test.csv')

    # Null 값 보정
    df_train['DJI_DJI-price_close'] = df_train['DJI_DJI-price_close'].astype(str)
    df_train_null = df_train[df_train['DJI_DJI-price_close'] == 'nan']

    for col_name in list_null:
        if col_name != 'target':
            df_train_null["DJI_DJI-price_open"] = (df_train["DJI_DJI-price_open" + '_shift-1'] + df_train["DJI_DJI-price_open" + '_shift+1']) / 2
            df_train_null[col_name] = (df_train[col_name + '_shift-1'] + df_train[col_name + '_shift+1']) / 2
            del df_train_null[col_name + '_shift-1']
            del df_train_null[col_name + '_shift+1']
    df_train_null = df_train_null[list_train_columns]

    # 보정 결과 반영
    set_null_indice = set(df_train_null.index.tolist())
    df_train_not_null = df_train[~df_train.index.isin(set_null_indice)]
    df_train_not_null = df_train_not_null[list_train_columns]

    df_train = pandas.concat([df_train_not_null, df_train_null], sort=False)
    df_train = df_train.sort_index(ascending=True)

    # 학습/테스트 데이터 분리 (결과 확인을 위해 최근일 -1 일 데이터 사용)
    df_train.drop(df_train.tail(1).index, inplace=True)
    df_tests = df_train.tail(1)
    df_tests.to_csv(DATA_PATH + 'd1_203_tests.csv')
    df_train.drop(df_train.tail(1).index, inplace=True)
    df_train.to_csv(DATA_PATH + 'd1_202_train.csv')
    print(df_train[['005930-price_close', 'target']].head())
    print(df_train[['005930-price_close', 'target']].tail())
    print(df_tests[['005930-price_close', 'target']].tail())

if __name__ == "__main__":

    code_equity = "005930"
    data_merge(code_equity)
    make_target()