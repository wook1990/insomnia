import pandas as pd
# 1) Series
s1 = pd.Series([4,5,6,7])
print(s1)
'''
0    4
1    5
2    6
3    7
dtype: int64
'''


# 주요 메소드

# i) values
# 해당 시리즈의 객제의 값들만 반환
print(s1.values)
#>>> [4 5 6 7]
print(type(s1.values))
# >>> <class 'numpy.ndarray'>


# ii) index
# 해당 시리즈의 색인(index)만 반환
print(s1.index)
# >>> RangeIndex(start=0, stop=4, step=1)
print(type(s1.index))
# >>> <class 'pandas.core.indexes.range.RangeIndex'>

# iii) Series[index]
# 배열에서 값을 선택하거나 대입할때 색인을 사용
# 일반적인 Python의 자료 접근 방식과 유사
# 값 인덱싱
s1[1]
# 값 추가
s1[4] = 5
print(s1)
'''
0    4
1    5
2    6
3    7
4    5
dtype: int64
'''

#  데이터 프레임 만들기
# 1. 딕셔너리에서 데이터프레임으로 변환
dic_data = {'country': ['벨기에', '인도', '브라질'],
'capital': ['브뤼셀', '뉴델리', '브라질리아'],
'population': [11190846, 1303171035, 207847528]}
df = pd.DataFrame(dic_data)
print(df)
'''
  country capital  population
0     벨기에     브뤼셀    11190846
1      인도     뉴델리  1303171035
2     브라질   브라질리아   207847528

'''

# 2. 시리즈에서 데이터 프레임으로 변환
series = {'one': pd.Series([1., 2., 3.], index=['a', 'b', 'c']),
'two': pd.Series([1., 2., 3., 4.], index=['a', 'b', 'c', 'd'])}
df = pd.DataFrame(series)
print(df)
'''
   one  two
a  1.0  1.0
b  2.0  2.0
c  3.0  3.0
d  NaN  4.0
'''

# 3. ndArray & Lists에서 데이터프레임으로 변환
ndArrays = {'one': [1., 2., 3., 4.],
'two': [4., 3., 2., 1.]}
df = pd.DataFrame(ndArrays)
print(df)
'''
   one  two
0  1.0  4.0
1  2.0  3.0
2  3.0  2.0
3  4.0  1.0
'''



# 데이터 다루기
import pandas as pd
exam_dic = {'국어' : [80, 70, 90, 87],
            "영어" : [100, 80, 50, 90],
            '수학' : [85, 75, 65, 100],
            "과학" : [89, 67, 90, 85]}
data = pd.DataFrame(exam_dic, index = ['Evan', 'Chloe', 'Alice', 'John'])
print(data)
'''
       국어   영어   수학  과학
Evan   80  100   85  89
Chloe  70   80   75  67
Alice  90   50   65  90
John   87   90  100  85
'''

print(data['국어'])
'''
Evan     80
Chloe    70
Alice    90
John     87
Name: 국어, dtype: int64
'''

print(data.국어)
'''
Evan     80
Chloe    70
Alice    90
John     87
Name: 국어, dtype: int64
'''

print(data[['국어','수학']])
'''
       국어   수학
Evan   80   85
Chloe  70   75
Alice  90   65
John   87  100
'''


# 언어영역합계 열 추가
data['언어영역합계'] = data['국어'] + data['영어']
print(data)
'''
       국어   영어   수학  과학  언어영역합계
Evan   80  100   85  89     180
Chloe  70   80   75  67     150
Alice  90   50   65  90     140
John   87   90  100  85     177
'''

# 수리영역합계 열 추가
data['수리영역합계'] = data['수학'] + data['과학']
print(data)
'''
       국어   영어   수학  과학  언어영역합계  수리영역합계
Evan   80  100   85  89     180     174
Chloe  70   80   75  67     150     142
Alice  90   50   65  90     140     155
John   87   90  100  85     177     185
'''


# 수업태도 점수 추가
data["수업태도점수"] = 10
print(data)
'''
       국어   영어   수학  과학  언어영역합계  수리영역합계  수업태도점수
Evan   80  100   85  89     180     174      10
Chloe  70   80   75  67     150     142      10
Alice  90   50   65  90     140     155      10
John   87   90  100  85     177     185      10
'''


del data["수업태도점수"]
print(data)
'''
       국어   영어   수학  과학  언어영역합계  수리영역합계
Evan   80  100   85  89     180     174
Chloe  70   80   75  67     150     142
Alice  90   50   65  90     140     155
John   87   90  100  85     177     185
'''


data.pop('수리영역합계')
'''
Evan     174
Chloe    142
Alice    155
John     185
Name: 수리영역합계, dtype: int64
'''
print(data)
'''
       국어   영어   수학  과학  언어영역합계
Evan   80  100   85  89     180
Chloe  70   80   75  67     150
Alice  90   50   65  90     140
John   87   90  100  85     177
'''


data.drop('언어영역합계',axis=1, inplace=True)
print(data)
'''
       국어   영어   수학  과학
Evan   80  100   85  89
Chloe  70   80   75  67
Alice  90   50   65  90
John   87   90  100  85
'''






# 열다루기

import pandas as pd

exam_dic = {'이름' : ['Evan', 'Chloe', 'Alice', 'John'],
            '국어' : [80, 70, 90, 87],
            '영어' : [100, 80, 50, 90],
            '수학' : [85, 75, 65, 100],
            '과학' : [89, 67, 90, 85]}
data = pd.DataFrame(exam_dic)
print(data)
'''
   이름  국어   영어   수학  과학
0   Evan  80     100    85    89
1  Chloe  70      80    75    67
2  Alice  90      50    65    90
3   John  87      90   100    85

'''


data.loc[1]
'''
이름    Chloe
국어       70
영어       80
수학       75
과학       67
Name: 1, dtype: object

'''

data.loc[2]
'''
이름    Alice
국어       90
영어       50
수학       65
과학       90
Name: 2, dtype: object
'''


data.set_index('이름',inplace=True)
print(data)
'''
       국어    영어   수학   과학
이름                     
Evan    80     100     85    89
Chloe   70     80      75    67
Alice   90     50      65    90
John    87     90     100    85
'''


data.loc['Chloe']
'''
국어    70
영어    80
수학    75
과학    67
Name: Chloe, dtype: int64
'''

data.loc["Alice"]
'''
국어    90
영어    50
수학    65
과학    90
Name: Alice, dtype: int64
'''


data.loc[['Chloe', 'Alice']]
'''
       국어  영어  수학  과학
이름                   
Chloe  70     80    75    67
Alice  90     50    65    90
'''


# 행 추가\
data.loc['Chris'] = [90, 85, 80 ,50]
print(data)
'''
       국어   영어   수학  과학
이름                     
Evan   80  100   85  89
Chloe  70   80   75  67
Alice  90   50   65  90
John   87   90  100  85
Chris  90   85   80  50
'''

exam_dic2 = {'국어' : [80, 70],
             '영어' : [100, 80],
             '수학' : [85, 75],
             '과학' : [89, 67]}
new_row = pd.DataFrame(exam_dic2, index=['대한이', '민국'])
print(new_row)
'''
       국어   영어  수학  과학
대한이  80    100    85    89
민국    70     80    75    67
'''
data2 = pd.concat([data, new_row])
print(data2)
'''
       국어   영어  수학  과학
Evan     80   100     85    89
Chloe    70    80     75    67
Alice    90    50     65    90
John     87    90    100    85
Chris    90    85     80    50
대한이   80   100     85    89
민국     70    80     75    67
'''

data.drop(["Evan", "Chloe"], inplace = True)
print(data)
'''
       국어  영어   수학  과학
이름                    
Alice  90    50     65    90
John   87    90    100    85
Chris  90    85     80    50
'''

# 행과 열 다루기
# 최대 줄 수 설정
pd.set_option('display.max_rows', 5000)
# 최대 열 수 설정
pd.set_option('display.max_columns', 5000)
# 표시할 가로의 길이
pd.set_option('display.width', 10000)

url = 'https://assets.datacamp.com/production/repositories/502/datasets/502f4eedaf44ad1c94b3595c7691746f282e0b0a/pennsylvania2012_turnout.csv'
election = pd.read_csv(url, index_col = "county")
print(election.head())
'''
          state   total      Obama     Romney  winner  voters    turnout     margin
county                                                                             
Adams        PA   41973  35.482334  63.112001  Romney   61156  68.632677  27.629667
Allegheny    PA  614671  56.640219  42.185820   Obama  924351  66.497575  14.454399
Armstrong    PA   28322  30.696985  67.901278  Romney   42147  67.198140  37.204293
Beaver       PA   80015  46.032619  52.637630  Romney  115157  69.483401   6.605012
Bedford      PA   21444  22.057452  76.986570  Romney   32189  66.619031  54.929118
'''

election.loc['Allegheny', 'winner']
# >>>  'Obama'

election.iloc[1,4]
# >>> 'Obama'

# column 선택 및 재정렬
results = election[['winner','total']]
print(results.head(10))
'''
           winner   total
county                   
Adams      Romney   41973
Allegheny   Obama  614671
Armstrong  Romney   28322
Beaver     Romney   80015
Bedford    Romney   21444
Berks      Romney  163253
Blair      Romney   47631
Bradford   Romney   22501
Bucks       Obama  319407
Butler     Romney   88924
'''

p_counties = election.loc['Perry':'Potter']
print(p_counties)
'''
             state   total      Obama     Romney  winner   voters    turnout     margin
county                                                                                 
Perry           PA   18240  29.769737  68.591009  Romney    27245  66.948064  38.821272
Philadelphia    PA  653598  85.224251  14.051451   Obama  1099197  59.461407  71.172800
Pike            PA   23164  43.904334  54.882576  Romney    41840  55.363289  10.978242
Potter          PA    7205  26.259542  72.158223  Romney    10913  66.022175  45.898681
'''

p_counties_rev = election.loc['Potter':'Perry':-1]
print(p_counties_rev)
'''
             state   total      Obama     Romney  winner   voters    turnout     margin
county                                                                                 
Potter          PA    7205  26.259542  72.158223  Romney    10913  66.022175  45.898681
Pike            PA   23164  43.904334  54.882576  Romney    41840  55.363289  10.978242
Philadelphia    PA  653598  85.224251  14.051451   Obama  1099197  59.461407  71.172800
Perry           PA   18240  29.769737  68.591009  Romney    27245  66.948064  38.821272
'''

to_Obama = election.loc[:, :'Obama']
print(to_Obama.head())
'''
          state   total      Obama
county                            
Adams        PA   41973  35.482334
Allegheny    PA  614671  56.640219
Armstrong    PA   28322  30.696985
Beaver       PA   80015  46.032619
Bedford      PA   21444  22.057452
'''

Obama_to_Voters = election.loc[:, 'Obama':'voters']
print(Obama_to_Voters.head())

'''
               Obama     Romney  winner  voters
county                                         
Adams      35.482334  63.112001  Romney   61156
Allegheny  56.640219  42.185820   Obama  924351
Armstrong  30.696985  67.901278  Romney   42147
Beaver     46.032619  52.637630  Romney  115157
Bedford    22.057452  76.986570  Romney   32189
'''

subselected_election = election.loc['Potter':'Perry':-1, 'Obama':'voters']
print(subselected_election)
'''
                  Obama     Romney  winner   voters
county                                             
Potter        26.259542  72.158223  Romney    10913
Pike          43.904334  54.882576  Romney    41840
Philadelphia  85.224251  14.051451   Obama  1099197
Perry         29.769737  68.591009  Romney    27245
'''


# 데이터 요약
import pandas as pd
import seaborn as sns

df = sns.load_dataset("titanic")
print(df)
'''
     survived  pclass     sex    age  sibsp  parch      fare embarked   class    who  adult_male deck  embark_town alive  alone
0           0       3    male  22.00      1      0    7.2500        S   Third    man        True  NaN  Southampton    no  False
1           1       1  female  38.00      1      0   71.2833        C   First  woman       False    C    Cherbourg   yes  False
2           1       3  female  26.00      0      0    7.9250        S   Third  woman       False  NaN  Southampton   yes   True
3           1       1  female  35.00      1      0   53.1000        S   First  woman       False    C  Southampton   yes  False
4           0       3    male  35.00      0      0    8.0500        S   Third    man        True  NaN  Southampton    no   True
5           0       3    male    NaN      0      0    8.4583        Q   Third    man        True  NaN   Queenstown    no   True
...
886         0       2    male  27.00      0      0   13.0000        S  Second    man        True  NaN  Southampton    no   True
887         1       1  female  19.00      0      0   30.0000        S   First  woman       False    B  Southampton   yes   True
888         0       3  female    NaN      1      2   23.4500        S   Third  woman       False  NaN  Southampton    no  False
889         1       1    male  26.00      0      0   30.0000        C   First    man        True    C    Cherbourg   yes   True
890         0       3    male  32.00      0      0    7.7500        Q   Third    man        True  NaN   Queenstown    no   True
'''


df.head(10)
'''
   survived  pclass     sex   age  sibsp  parch     fare embarked   class    who  adult_male deck  embark_town alive  alone
0         0       3    male  22.0      1      0   7.2500        S   Third    man        True  NaN  Southampton    no  False
1         1       1  female  38.0      1      0  71.2833        C   First  woman       False    C    Cherbourg   yes  False
2         1       3  female  26.0      0      0   7.9250        S   Third  woman       False  NaN  Southampton   yes   True
3         1       1  female  35.0      1      0  53.1000        S   First  woman       False    C  Southampton   yes  False
4         0       3    male  35.0      0      0   8.0500        S   Third    man        True  NaN  Southampton    no   True
5         0       3    male   NaN      0      0   8.4583        Q   Third    man        True  NaN   Queenstown    no   True
6         0       1    male  54.0      0      0  51.8625        S   First    man        True    E  Southampton    no   True
7         0       3    male   2.0      3      1  21.0750        S   Third  child       False  NaN  Southampton    no  False
8         1       3  female  27.0      0      2  11.1333        S   Third  woman       False  NaN  Southampton   yes  False
9         1       2  female  14.0      1      0  30.0708        C  Second  child       False  NaN    Cherbourg   yes  False
'''

df.tail(10)
'''
     survived  pclass     sex   age  sibsp  parch     fare embarked   class    who  adult_male deck  embark_town alive  alone
881         0       3    male  33.0      0      0   7.8958        S   Third    man        True  NaN  Southampton    no   True
882         0       3  female  22.0      0      0  10.5167        S   Third  woman       False  NaN  Southampton    no   True
883         0       2    male  28.0      0      0  10.5000        S  Second    man        True  NaN  Southampton    no   True
884         0       3    male  25.0      0      0   7.0500        S   Third    man        True  NaN  Southampton    no   True
885         0       3  female  39.0      0      5  29.1250        Q   Third  woman       False  NaN   Queenstown    no  False
886         0       2    male  27.0      0      0  13.0000        S  Second    man        True  NaN  Southampton    no   True
887         1       1  female  19.0      0      0  30.0000        S   First  woman       False    B  Southampton   yes   True
888         0       3  female   NaN      1      2  23.4500        S   Third  woman       False  NaN  Southampton    no  False
889         1       1    male  26.0      0      0  30.0000        C   First    man        True    C    Cherbourg   yes   True
890         0       3    male  32.0      0      0   7.7500        Q   Third    man        True  NaN   Queenstown    no   True
'''

df.dtypes
'''
survived          int64
pclass            int64
sex              object
age             float64
sibsp             int64
parch             int64
fare            float64
embarked         object
class          category
who              object
adult_male         bool
deck           category
embark_town      object
alive            object
alone              bool
dtype: object
'''

print(df.describe())
'''
         survived      pclass         age       sibsp       parch        fare
count  891.000000  891.000000  714.000000  891.000000  891.000000  891.000000
mean     0.383838    2.308642   29.699118    0.523008    0.381594   32.204208
std      0.486592    0.836071   14.526497    1.102743    0.806057   49.693429
min      0.000000    1.000000    0.420000    0.000000    0.000000    0.000000
25%      0.000000    2.000000   20.125000    0.000000    0.000000    7.910400
50%      0.000000    3.000000   28.000000    0.000000    0.000000   14.454200
75%      1.000000    3.000000   38.000000    1.000000    0.000000   31.000000
max      1.000000    3.000000   80.000000    8.000000    6.000000  512.329200
'''

# 범주형 데이터
# 범주형 데이터 여부 확인
cat = (df.dtypes == "object")
print(cat)
'''
survived       False
pclass         False
sex             True
age            False
sibsp          False
parch          False
fare           False
embarked        True
class          False
who             True
adult_male     False
deck           False
embark_town     True
alive           True
alone          False
'''
# 범주형 변수 추출
cat_var = df.dtypes[cat]
print(cat_var)
'''
sex            object
embarked       object
who            object
embark_town    object
alive          object
dtype: object
'''

df.groupby("sex").agg(count_sex = ("sex", 'count'))
'''
        count_sex
sex              
female        314
male          577
'''
df.groupby("embarked").agg(count_embarked = ("embarked", 'count'))
'''
          count_embarked
embarked                
C                    168
Q                     77
S                    644
'''
df.groupby("embark_town").agg(count_embark_town = ("embark_town", 'count'))
'''
             count_embark_town
embark_town                   
Cherbourg                  168
Queenstown                  77
Southampton                644
'''
df.groupby("who").agg(count_who = ("who", 'count'))
'''
       count_who
who             
child         83
man          537
woman        271
'''
df.groupby("alive").agg(count_alive = ("alive", 'count'))
'''
       count_alive
alive             
no             549
yes            342
'''

print(df.isnull().sum())
'''
survived         0
pclass           0
sex              0
age            177
sibsp            0
parch            0
fare             0
embarked         2
class            0
who              0
adult_male       0
deck           688
embark_town      2
alive            0
alone            0
'''

# 나이 컬럼만 추출
a = df["age"]
# 40 세 미만 조건 추출
b = a < 40
print(b.head(10))
'''
0      True
1      True
2      True
3      True
4      True
6     False
7      True
8      True
9      True
10     True
'''
print(b.sum())
# >>> 551

df.columns
# 각 범주에 다른 fare 최대 최소값
df[['sex','fare']].groupby('sex', dropna=True).agg(min_fare = ('fare',min), max_fare = ('fare',max))
'''
        min_fare  max_fare
sex                       
female      6.75  512.3292
male        0.00  512.3292
'''
df[['embarked','fare']].groupby('embarked', dropna=True).agg(min_fare = ('fare',min), max_fare = ('fare',max))
'''
          min_fare  max_fare
embarked                    
C           4.0125  512.3292
Q           6.7500   90.0000
S           0.0000  263.0000
'''
df[['who','fare']].groupby('who', dropna=True).agg(min_fare = ('fare',min), max_fare = ('fare',max))
'''
       min_fare  max_fare
who                      
child     7.225  211.3375
man       0.000  512.3292
woman     6.750  512.3292
'''
df[['embark_town','fare']].groupby('embark_town', dropna=True).agg(min_fare = ('fare',min), max_fare = ('fare',max))
'''
             min_fare  max_fare
embark_town                    
Cherbourg      4.0125  512.3292
Queenstown     6.7500   90.0000
Southampton    0.0000  263.0000
'''
df[['alive','fare']].groupby('alive', dropna=True).agg(min_fare = ('fare',min), max_fare = ('fare',max))
'''
       min_fare  max_fare
alive                    
no          0.0  263.0000
yes         0.0  512.3292
'''

# 성별에 따른 전체 수치형 변수의 평균
df.groupby('sex').mean()
'''
        survived    pclass        age     sibsp     parch       fare  adult_male     alone
sex                                                                                       
female  0.742038  2.159236  27.915709  0.694268  0.649682  44.479818    0.000000  0.401274
male    0.188908  2.389948  30.726645  0.429809  0.235702  25.523893    0.930676  0.712305
'''
df.groupby('embarked').mean()
'''
          survived    pclass        age     sibsp     parch       fare  adult_male     alone
embarked                                                                                    
C         0.553571  1.886905  30.814769  0.386905  0.363095  59.954144    0.535714  0.505952
Q         0.389610  2.909091  28.089286  0.428571  0.168831  13.276030    0.480519  0.740260
S         0.336957  2.350932  29.445397  0.571429  0.413043  27.079812    0.636646  0.610248
'''
df.groupby('who').mean()
'''
       survived    pclass        age     sibsp     parch       fare  adult_male     alone
who                                                                                      
child  0.590361  2.626506   6.369518  1.734940  1.265060  32.785795         0.0  0.072289
man    0.163873  2.372439  33.173123  0.296089  0.152700  24.864182         1.0  0.763501
woman  0.756458  2.084871  32.000000  0.601476  0.564576  46.570711         0.0  0.446494
'''
df.groupby('embark_town').mean()
'''
             survived    pclass        age     sibsp     parch       fare  adult_male     alone
embark_town                                                                                    
Cherbourg    0.553571  1.886905  30.814769  0.386905  0.363095  59.954144    0.535714  0.505952
Queenstown   0.389610  2.909091  28.089286  0.428571  0.168831  13.276030    0.480519  0.740260
Southampton  0.336957  2.350932  29.445397  0.571429  0.413043  27.079812    0.636646  0.610248
'''
df.groupby('alive').mean()
'''
       survived    pclass        age     sibsp     parch       fare  adult_male     alone
alive                                                                                    
no          0.0  2.531876  30.626179  0.553734  0.329690  22.117887    0.817851  0.681239
yes         1.0  1.950292  28.343690  0.473684  0.464912  48.395408    0.257310  0.476608
'''


# concat
import numpy as np
# 임의의 난수로 데이서 생성
df = pd.DataFrame(np.random.randn(10, 4))
print(df)
'''
          0         1         2         3
0 -1.321014 -1.576978 -0.101584 -0.902394
1 -2.066681  0.313877 -0.045289 -0.341854
2  1.060721 -1.196711 -0.179277  0.430177
3 -0.111604 -0.906930  0.795542  1.681785
4  0.297519  0.725870  0.538120  0.479057
5 -0.793504  0.461562  0.078431  2.185964
6 -0.492519  1.521511  0.551097 -0.721491
7 -0.017354  1.405763  1.170045 -0.774836
8  0.973632  0.000167 -1.069122 -0.200101
9  0.112477 -1.562122 -0.410183 -1.463272
'''
# 슬라이싱을 통해 데이터 프레임 3덩어리로 분할
pieces = [df[:3], df[3:7], df[7:]]
print(pieces[0])
'''
          0         1         2         3
0 -1.321014 -1.576978 -0.101584 -0.902394
1 -2.066681  0.313877 -0.045289 -0.341854
2  1.060721 -1.196711 -0.179277  0.430177
'''
print(pieces[1])
'''
          0         1         2         3
3 -0.111604 -0.906930  0.795542  1.681785
4  0.297519  0.725870  0.538120  0.479057
5 -0.793504  0.461562  0.078431  2.185964
6 -0.492519  1.521511  0.551097 -0.721491
'''
print(pieces[2])
'''
          0         1         2         3
7 -0.017354  1.405763  1.170045 -0.774836
8  0.973632  0.000167 -1.069122 -0.200101
9  0.112477 -1.562122 -0.410183 -1.463272
'''
pd.concat(pieces)
'''
          0         1         2         3
0 -1.321014 -1.576978 -0.101584 -0.902394
1 -2.066681  0.313877 -0.045289 -0.341854
2  1.060721 -1.196711 -0.179277  0.430177
3 -0.111604 -0.906930  0.795542  1.681785
4  0.297519  0.725870  0.538120  0.479057
5 -0.793504  0.461562  0.078431  2.185964
6 -0.492519  1.521511  0.551097 -0.721491
7 -0.017354  1.405763  1.170045 -0.774836
8  0.973632  0.000167 -1.069122 -0.200101
9  0.112477 -1.562122 -0.410183 -1.463272
'''


#merge
# Example_1
left = pd.DataFrame({'key': ['foo', 'foo'], 'lval': [1, 2]})
print(left)
'''
   key  lval
0  foo     1
1  foo     2
'''
right = pd.DataFrame({'key': ['foo', 'foo'], 'rval': [4, 5]})
print(right)
'''
   key  rval
0  foo     4
1  foo     5
'''

t_df = pd.merge(left, right, on='key')
print(t_df)
'''
   key  lval  rval
0  foo     1     4
1  foo     1     5
2  foo     2     4
3  foo     2     5
'''

# Example_2
left = pd.DataFrame({'key': ['foo', 'bar'], 'lval': [1, 2]})
print(left)
'''
   key  lval
0  foo     1
1  bar     2
'''
right = pd.DataFrame({'key': ['foo', 'bar'], 'rval': [4, 5]})
print(right)
'''
   key  rval
0  foo     4
1  bar     5
'''
t_df = pd.merge(left, right, on='key')
print(t_df)
'''
   key  lval  rval
0  foo     1     4
1  bar     2     5
'''
