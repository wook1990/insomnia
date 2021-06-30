# import packages
import pandas as pd
import numpy as np
import seaborn as sns

import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import matplotlib
from matplotlib.pyplot import figure


# plot 설정
plt.style.use('ggplot')
matplotlib.rcParams['figure.figsize'] = (12,8)
pd.options.mode.chained_assignment = None

# 데이터 로드
df = pd.read_csv("E:/Data/sberbank/train.csv")

# 데이터 shape 및 type 확인
print(df.shape)
# (30471, 292)
print(df.dtypes)


# 숫자형 변수 선택
df_numeric = df.select_dtypes(include=[np.number])
numeric_cols = df_numeric.columns.values
print(numeric_cols)
len(numeric_cols)
# 범주형 변수 선택
df_non_numeric = df.select_dtypes(exclude=[np.number])
non_numeric_cols = df_non_numeric.columns.values
print(non_numeric_cols)


# Missing Data Heatmap
# 처음의 30개 컬럼
cols = df.columns[:30]
# 누락된 값은 노란색, 정상적인 값은 파란색
colours = ['#000099','#ffff00']
# seaborn의 heatmap 사용하여 누락된 컬럼 확인
sns.heatmap(df[cols].isnull(), cmap=sns.color_palette(colours))

# Missing Data Percentage List
# 각 컬럼별 누락된 값의 비율 확인
for col in df.columns:
    pct_missing = np.mean(df[col].isnull())
    print('{} - {}%'.format(col, round(pct_missing*100)))


# 누락된 데이터 히스토 그램
# 각 행별로 결측컬럼이 몇개 존재하는지 식별
for col in df.columns:
    # 각 컬럼의 누락된 값의 합
    missing = df[col].isnull()
    num_missing = np.sum(missing)
    # 누락된 값이 있다면, 누락된 컬럼으로 컬럼 여부값 생성
    if num_missing > 0:
        print('created missing indicator for: {}'.format(col))
        df['{}_ismissing'.format(col)] = missing

# 누락 여부 컬럼을 추출해 전체 행중 누락된 데이터가 몇건 인지 식별
ismissing_cols = [col for col in df.columns if 'ismissing' in col]
df['num_missing'] = df[ismissing_cols].sum(axis=1)

df['num_missing'].value_counts().reset_index().sort_values(by='index').plot.bar(x='index', y='num_missing')

# 결측치 컬럼이 많은 행 삭제
ind_missing =df[df['num_missing'] > 35].index
df_less_missing_rows = df.drop(ind_missing,axis=0)
df_less_missing_rows.shape
# 29779, 344


# 결측컬럼삭제
#  앞서 확인한 컬럼별 결측비율에서
# hospital_beds_raion은 47%의 많은 결측치를 포함
cols_to_drop = ["hospital_beds_raion"]
df_less_hos_beds_raion = df.drop(cols_to_drop, axis=1)
df_less_hos_beds_raion.shape
# (30471, 343)


# life_sq의 변수의 결측치를 중앙값으로 대치
med = df['life_sq'].median()
print(med)
df["life_sq"] = df["life_sq"].fillna(med)
print("{0} 의 결측치 개수 : {1}".format("life_sq",df["life_sq"].isnull().sum()))
# life_sq 의 결측치 개수 : 0

# 연속형 변수는 중앙 값으로 대체
# 연속형 변수만 추출
df_numeric = df.select_dtypes(include=[np.number])
numeric_cols = df_numeric.columns.values
# 연속형 변수의 결측치 총 합
for col in numeric_cols:
    missing = df[col].isnull()
    num_missing = np.sum(missing)

    if num_missing > 0:  # 결측값이 존재하는 컬럼에 한하여 중앙값으로 대체
        print('imputing missing values for: {}'.format(col))
        df['{}_ismissing'.format(col)] = missing
        med = df[col].median()
        df[col] = df[col].fillna(med)
'''        
imputing missing values for: floor
imputing missing values for: max_floor
imputing missing values for: material
imputing missing values for: build_year
imputing missing values for: num_room
imputing missing values for: kitch_sq
imputing missing values for: state
imputing missing values for: preschool_quota
imputing missing values for: school_quota
imputing missing values for: hospital_beds_raion
imputing missing values for: raion_build_count_with_material_info
imputing missing values for: build_count_block
imputing missing values for: build_count_wood
imputing missing values for: build_count_frame
imputing missing values for: build_count_brick
imputing missing values for: build_count_monolith
imputing missing values for: build_count_panel
imputing missing values for: build_count_foam
imputing missing values for: build_count_slag
imputing missing values for: build_count_mix
imputing missing values for: raion_build_count_with_builddate_info
imputing missing values for: build_count_before_1920
imputing missing values for: build_count_1921-1945
imputing missing values for: build_count_1946-1970
imputing missing values for: build_count_1971-1995
imputing missing values for: build_count_after_1995
imputing missing values for: metro_min_walk
imputing missing values for: metro_km_walk
imputing missing values for: railroad_station_walk_km
imputing missing values for: railroad_station_walk_min
imputing missing values for: ID_railroad_station_walk
imputing missing values for: cafe_sum_500_min_price_avg
imputing missing values for: cafe_sum_500_max_price_avg
imputing missing values for: cafe_avg_price_500
imputing missing values for: cafe_sum_1000_min_price_avg
imputing missing values for: cafe_sum_1000_max_price_avg
imputing missing values for: cafe_avg_price_1000
imputing missing values for: cafe_sum_1500_min_price_avg
imputing missing values for: cafe_sum_1500_max_price_avg
imputing missing values for: cafe_avg_price_1500
imputing missing values for: cafe_sum_2000_min_price_avg
imputing missing values for: cafe_sum_2000_max_price_avg
imputing missing values for: cafe_avg_price_2000
imputing missing values for: cafe_sum_3000_min_price_avg
imputing missing values for: cafe_sum_3000_max_price_avg
imputing missing values for: cafe_avg_price_3000
imputing missing values for: prom_part_5000
imputing missing values for: cafe_sum_5000_min_price_avg
imputing missing values for: cafe_sum_5000_max_price_avg
imputing missing values for: cafe_avg_price_5000
'''

# 범주형 변수만 추출
df_non_numeric = df.select_dtypes(exclude=[np.number])
non_numeric_cols = df_non_numeric.columns.values
# 범주형 변수의 결측치 판단
for col in non_numeric_cols:
    missing = df[col].isnull()
    num_missing = np.sum(missing)

    if num_missing > 0:  # 결측이 존재하는 컬럼의 결측값을 최빈값으로 대치
        print('imputing missing values for: {}'.format(col))
        df['{}_ismissing'.format(col)] = missing

        top = df[col].describe()['top']
        df[col] = df[col].fillna(top)


# 2) 이상치
# life_sq 데이터를 통한 확인
# 히스토그램
df["life_sq"].hist(bins=100)
# 상자그림
df.boxplot(column=['life_sq'])

# 기술통계
df["life_sq"].describe()
'''
count    30471.000000
mean        33.480883
std         46.522251
min          0.000000
25%         22.000000
50%         30.000000
75%         38.000000
max       7478.000000
'''

# 범주형
# 막대그림
df['ecology'].value_counts().plot.bar()



# 3) 불필요한 데이터
# 1. 정보 없음/중복
# 전체 데이터 수
num_rows = len(df.index)
# 정보력이 낮은 컬럼을 담을 리스트 선언
low_information_cols = []

# 전체 컬럼의 값 추출(na포함)
for col in df.columns:
    cnts = df[col].value_counts(dropna=False)
    top_pct = (cnts/num_rows).iloc[0]

    # 컬럼의 동일 값의 비율이 95프로 이상이 컬럼 추출
    if top_pct > 0.95:
        low_information_cols.append(col)
        print('{0} : {1:.5f}%'.format(col, top_pct*100))
        print(cnts)
        print()


'''oil_chemistry_raion : 99.02858%
no     30175
yes      296
Name: oil_chemistry_raion, dtype: int64

railroad_terminal_raion : 96.27187%
no     29335
yes     1136
Name: railroad_terminal_raion, dtype: int64

nuclear_reactor_raion : 97.16780%
no     29608
yes      863
Name: nuclear_reactor_raion, dtype: int64

build_count_foam : 95.35624%
0.0     29056
1.0       969
11.0      262
2.0       184
Name: build_count_foam, dtype: int64

big_road1_1line : 97.43691%
no     29690
yes      781
Name: big_road1_1line, dtype: int64

railroad_1line : 97.06934%
no     29578
yes      893
Name: railroad_1line, dtype: int64

cafe_count_500_price_high : 97.25641%
0    29635
1      787
2       38
3       11
Name: cafe_count_500_price_high, dtype: int64

mosque_count_500 : 99.51101%
0    30322
1      149
Name: mosque_count_500, dtype: int64

cafe_count_1000_price_high : 95.52689%
0    29108
1     1104
2      145
3       51
4       39
5       15
6        8
7        1
Name: cafe_count_1000_price_high, dtype: int64

mosque_count_1000 : 98.08342%
0    29887
1      584
Name: mosque_count_1000, dtype: int64

mosque_count_1500 : 96.21936%
0    29319
1     1152
Name: mosque_count_1500, dtype: int64

floor_ismissing : 99.45194%
False    30304
True       167
Name: floor_ismissing, dtype: int64

metro_min_walk_ismissing : 99.91795%
False    30446
True        25
Name: metro_min_walk_ismissing, dtype: int64

metro_km_walk_ismissing : 99.91795%
False    30446
True        25
Name: metro_km_walk_ismissing, dtype: int64

railroad_station_walk_km_ismissing : 99.91795%
False    30446
True        25
Name: railroad_station_walk_km_ismissing, dtype: int64

railroad_station_walk_min_ismissing : 99.91795%
False    30446
True        25
Name: railroad_station_walk_min_ismissing, dtype: int64

ID_railroad_station_walk_ismissing : 99.91795%
False    30446
True        25
Name: ID_railroad_station_walk_ismissing, dtype: int64

cafe_sum_3000_min_price_avg_ismissing : 96.74773%
False    29480
True       991
Name: cafe_sum_3000_min_price_avg_ismissing, dtype: int64

cafe_sum_3000_max_price_avg_ismissing : 96.74773%
False    29480
True       991
Name: cafe_sum_3000_max_price_avg_ismissing, dtype: int64

cafe_avg_price_3000_ismissing : 96.74773%
False    29480
True       991
Name: cafe_avg_price_3000_ismissing, dtype: int64

prom_part_5000_ismissing : 99.41584%
False    30293
True       178
Name: prom_part_5000_ismissing, dtype: int64

cafe_sum_5000_min_price_avg_ismissing : 99.02530%
False    30174
True       297
Name: cafe_sum_5000_min_price_avg_ismissing, dtype: int64

cafe_sum_5000_max_price_avg_ismissing : 99.02530%
False    30174
True       297
Name: cafe_sum_5000_max_price_avg_ismissing, dtype: int64

cafe_avg_price_5000_ismissing : 99.02530%
False    30174
True       297
Name: cafe_avg_price_5000_ismissing, dtype: int64
'''


# 중복행 제거
# key값인 ID 컬럼 제거
df_dedupped = df.drop('id',axis=1).drop_duplicates()
print(df.shape)
# (30471, 344)
print(df_dedupped.shape)
# (30461, 343)

# 주요 Feature grouping을 사용한 중복 제거
key = ['timestamp', 'full_sq', 'life_sq', 'floor', 'build_year', 'num_room', 'price_doc']

df.fillna(-999).groupby(key)['id'].count().sort_values(ascending=False).head(20)
'''
timestamp   full_sq  life_sq  floor  build_year  num_room  price_doc
2012-10-22  61       30.0     18.0   1979.0      2.0       8248500      2
2014-12-17  62       30.0     9.0    1979.0      2.0       6552000      2
2014-01-22  46       28.0     1.0    1968.0      2.0       3000000      2
2013-04-03  42       30.0     2.0    1979.0      2.0       3444000      2
2013-09-23  85       30.0     14.0   1979.0      3.0       7725974      2
2012-08-27  59       30.0     6.0    1979.0      2.0       4506800      2
2014-04-15  134      134.0    1.0    0.0         3.0       5798496      2
2013-05-22  68       30.0     2.0    1979.0      2.0       5406690      2
2013-12-05  40       30.0     5.0    1979.0      1.0       4414080      2
2012-09-05  43       30.0     21.0   1979.0      2.0       6229540      2
2015-03-30  41       41.0     11.0   2016.0      1.0       4114580      2
2013-12-18  39       30.0     6.0    1979.0      1.0       3700946      2
2013-06-24  40       30.0     12.0   1979.0      2.0       4112800      2
2013-08-30  40       30.0     12.0   1979.0      1.0       4462000      2
2014-12-09  40       30.0     17.0   1979.0      1.0       4607265      2
2015-03-14  62       30.0     2.0    1979.0      2.0       6520500      2
2013-08-29  60       60.0     11.0   0.0         2.0       6518400      1
            56       41.0     2.0    1961.0      3.0       8380000      1
            52       52.0     9.0    1979.0      2.0       5619060      1
            58       58.0     13.0   2013.0      2.0       5764128      1
'''

# drop duplicates based on an subset of variables.

key = ['timestamp', 'full_sq', 'life_sq', 'floor', 'build_year', 'num_room', 'price_doc']
df_dedupped2 = df.drop_duplicates(subset=key)

print(df.shape)
(30471, 344)
print(df_dedupped2.shape)
(30455, 344)


# 4) 일관되지 않은 데이터
df['sub_area'].value_counts(dropna=False)
'''
Poselenie Sosenskoe               1776
Nekrasovka                        1611
Poselenie Vnukovskoe              1372
Poselenie Moskovskij               925
Poselenie Voskresenskoe            713
                                  ... 
Molzhaninovskoe                      3
Poselenie Kievskij                   2
Poselenie Shhapovskoe                2
Poselenie Klenovskoe                 1
Poselenie Mihajlovo-Jarcevskoe       1
'''

# make everything lower case.
df['sub_area_lower'] = df['sub_area'].str.lower()
df['sub_area_lower'].value_counts(dropna=False)

'''
poselenie sosenskoe               1776
nekrasovka                        1611
poselenie vnukovskoe              1372
poselenie moskovskij               925
poselenie voskresenskoe            713
                                  ... 
molzhaninovskoe                      3
poselenie shhapovskoe                2
poselenie kievskij                   2
poselenie mihajlovo-jarcevskoe       1
poselenie klenovskoe                 1
'''


# make everything lower case.
df['sub_area_lower'] = df['sub_area'].str.lower()
df['sub_area_lower'].value_counts(dropna=False)

'''
poselenie sosenskoe               1776
nekrasovka                        1611
poselenie vnukovskoe              1372
poselenie moskovskij               925
poselenie voskresenskoe            713
                                  ... 
molzhaninovskoe                      3
poselenie shhapovskoe                2
poselenie kievskij                   2
poselenie mihajlovo-jarcevskoe       1
poselenie klenovskoe                 1
'''


# 유형2. 형식
df['timestamp_dt'] = pd.to_datetime(df['timestamp'], format='%Y-%m-%d')
df['year'] = df['timestamp_dt'].dt.year
df['month'] = df['timestamp_dt'].dt.month
df['weekday'] = df['timestamp_dt'].dt.weekday
df[['timestamp', 'timestamp_dt','year','month','weekday']]
'''
        timestamp timestamp_dt  year  month  weekday
0      2011-08-20   2011-08-20  2011      8        5
1      2011-08-23   2011-08-23  2011      8        1
2      2011-08-27   2011-08-27  2011      8        5
3      2011-09-01   2011-09-01  2011      9        3
4      2011-09-05   2011-09-05  2011      9        0
           ...          ...   ...    ...      ...
30466  2015-06-30   2015-06-30  2015      6        1
30467  2015-06-30   2015-06-30  2015      6        1
30468  2015-06-30   2015-06-30  2015      6        1
30469  2015-06-30   2015-06-30  2015      6        1
30470  2015-06-30   2015-06-30  2015      6        1
'''
print(df['year'].value_counts(dropna=False))
'''
2014    13662
2013     7978
2012     4839
2015     3239
2011      753
'''
print()
print(df['month'].value_counts(dropna=False))
'''
12    3400
4     3191
3     2972
11    2970
10    2736
6     2570
5     2496
9     2346
'''

from nltk.metrics import edit_distance

df_city_ex = pd.DataFrame(data={'city': ['torontoo', 'toronto', 'tronto', 'vancouver', 'vancover', 'vancouvr', 'montreal', 'calgary']})


df_city_ex['city_distance_toronto'] = df_city_ex['city'].map(lambda x: edit_distance(x, 'toronto'))
df_city_ex['city_distance_vancouver'] = df_city_ex['city'].map(lambda x: edit_distance(x, 'vancouver'))
df_city_ex
'''
        city  city_distance_toronto  city_distance_vancouver
0   torontoo                      1                        8
1    toronto                      0                        8
2     tronto                      1                        8
3  vancouver                      8                        0
4   vancover                      7                        1
5   vancouvr                      7                        1
6   montreal                      7                        8
7    calgary                      7                        8
'''

msk = df_city_ex['city_distance_toronto'] <= 2
df_city_ex.loc[msk, 'city'] = 'toronto'

msk = df_city_ex['city_distance_vancouver'] <= 2
df_city_ex.loc[msk, 'city'] = 'vancouver'

df_city_ex



# 유형4. 주소

df_add_ex  =  pd . DataFrame ([ '123 MAIN St Apartment 15' , '123 Main Street Apt 12' , '543 FirSt Av' , '876 FIRst Ave.' ], columns = [ 'address' ])
df_add_ex
'''
                    address
0  123 MAIN St Apartment 15
1    123 Main Street Apt 12
2              543 FirSt Av
3            876 FIRst Ave.
'''

# 전부 소문자로 변경
df_add_ex['address_std'] = df_add_ex['address'].str.lower()
# 선행 및 후행 공백을 제거
df_add_ex['address_std'] = df_add_ex['address_std'].str.strip() # remove leading and trailing whitespace.
# 마침표를 제거
df_add_ex['address_std'] = df_add_ex['address_std'].str.replace('\\.', '') # remove period.
# 거리를 st 변경
df_add_ex['address_std'] = df_add_ex['address_std'].str.replace('\\bstreet\\b', 'st') # replace street with st.df_add_ex [ 'address_std' ] =  df_add_ex [ 'address_std' ].str.replace ( ' \\ bstreet \\ b' , 'st' )
# 아파트를 apt로 변경
df_add_ex['address_std'] = df_add_ex['address_std'].str.replace('\\bapartment\\b', 'apt') # replace apartment with apt.
# 거리를 ave로 변경
df_add_ex['address_std'] = df_add_ex['address_std'].str.replace('\\bav\\b', 'ave')

df_add_ex
'''
                    address         address_std
0  123 MAIN St Apartment 15  123 main st apt 15
1    123 Main Street Apt 12  123 main st apt 12
2              543 FirSt Av       543 first ave
3            876 FIRst Ave.       876 first ave
'''





