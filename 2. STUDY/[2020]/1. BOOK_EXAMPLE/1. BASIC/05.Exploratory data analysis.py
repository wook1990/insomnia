import numpy as np
import pandas as pd
from pandas import Series
import matplotlib.pyplot as plt
import seaborn as sns

# 최대 줄 수 설정
pd.set_option('display.max_rows', 500)
# 최대 열 수 설정
pd.set_option('display.max_columns', 500)
# 표시할 가로의 길이
pd.set_option('display.width', 1000)

plt.style.use('seaborn')
sns.set(font_scale=1.5)
# 이 두줄은  matplotlib 의 기본 scheme 말고 seaborn scheme 을 세팅하고, 일일이 graph 의
# font size 를 지정할 필요 없이 seaborn 의 font_scale 을 사용하면 편합니다.

import missingno as msno

#ignore warnings
import warnings
warnings.filterwarnings('ignore')

# 데이터 셋 확인
df = pd.read_csv("G:/내 드라이브/WAI_WORK/분석교재/data/titanic.csv")
# df.head() 메소드를 활용하여 데이터의 개략적인 형태 파악
df.head()
'''
   PassengerId  Survived  Pclass                                               Name     Sex   Age  SibSp  Parch            Ticket     Fare Cabin Embarked
0            1         0       3                            Braund, Mr. Owen Harris    male  22.0      1      0         A/5 21171   7.2500   NaN        S
1            2         1       1  Cumings, Mrs. John Bradley (Florence Briggs Th...  female  38.0      1      0          PC 17599  71.2833   C85        C
2            3         1       3                             Heikkinen, Miss. Laina  female  26.0      0      0  STON/O2. 3101282   7.9250   NaN        S
3            4         1       1       Futrelle, Mrs. Jacques Heath (Lily May Peel)  female  35.0      1      0            113803  53.1000  C123        S
4            5         0       3                           Allen, Mr. William Henry    male  35.0      0      0            373450   8.0500   NaN        S

'''

# describe()  메소드를 활용하여 연속형 변수의 통계치 확인
df.describe()
'''
       PassengerId    Survived      Pclass         Age       SibSp       Parch        Fare
count   891.000000  891.000000  891.000000  714.000000  891.000000  891.000000  891.000000
mean    446.000000    0.383838    2.308642   29.699118    0.523008    0.381594   32.204208
std     257.353842    0.486592    0.836071   14.526497    1.102743    0.806057   49.693429
min       1.000000    0.000000    1.000000    0.420000    0.000000    0.000000    0.000000
25%     223.500000    0.000000    2.000000   20.125000    0.000000    0.000000    7.910400
50%     446.000000    0.000000    3.000000   28.000000    0.000000    0.000000   14.454200
75%     668.500000    1.000000    3.000000   38.000000    1.000000    0.000000   31.000000
max     891.000000    1.000000    3.000000   80.000000    8.000000    6.000000  512.329200
'''

# 데이터를 확인해보니 전체 건수에 Null 값이 있는 컬럼 존재

# 1) Null Data Check
# 컬럼별로 Null 값이 전체 대비 몇 퍼센트 존재하는지 확인
for col in df.columns:
    msg = 'columns : {:>10}\t Percent of NaN Value : {:.2f}%'.format(col,100 * (df[col].isnull().sum()/df[col].shape[0]))
    print(msg)
'''
columns : PassengerId	 Percent of NaN Value : 0.00%
columns :   Survived	 Percent of NaN Value : 0.00%
columns :     Pclass	 Percent of NaN Value : 0.00%
columns :       Name	 Percent of NaN Value : 0.00%
columns :        Sex	 Percent of NaN Value : 0.00%
columns :        Age	 Percent of NaN Value : 19.87%
columns :      SibSp	 Percent of NaN Value : 0.00%
columns :      Parch	 Percent of NaN Value : 0.00%
columns :     Ticket	 Percent of NaN Value : 0.00%
columns :       Fare	 Percent of NaN Value : 0.00%
columns :      Cabin	 Percent of NaN Value : 77.10%
columns :   Embarked	 Percent of NaN Value : 0.22%
'''

# age 에서 약 20%의 Null 값
# Cabin 에서 약 77%의 Null 값
# Embarked 에서 약 0.22%의 Null 값이 존재

# MANO라는 라이브러리를 사용하여 null data를 시각화하여 확인
msno.matrix(df=df.iloc[:,:], figsize=(8, 8), color=(0.8,0.5,0.2))

msno.bar(df=df.iloc[:,:], figsize=(8, 8), color=(0.8,0.5,0.2))

# 1.2 Target Label 확인
# 타겟이 어떠한 분포를 가지고 있는지 확인하는 작업
# 이진 분류의 문제에 있어 1과 0의 분포가 어떠냐에 따라 모델의 평가 방법이 달라짐

fig = plt.figure()
ax1 = fig.add_subplot(1,2,1)
ax2 = fig.add_subplot(1,2,2)
df['Survived'].value_counts().plot.pie(explode=[0, 0.1], autopct='%1.1f%%', ax=ax1, shadow=True)
ax1.set_title('Pie plot - Survived')
ax1.set_ylabel('')
sns.countplot('Survived', data=df, ax=ax2)
ax2.set_title('Count plot - Survived')

plt.show()

# 생존한 사람 보다 죽은 사람이 많다.
# 1 : 38.4% , 0: 61.6%
# 어느정도 균일한 분포를 띄고 있기에 주어진 데이터를 그대로 활용해도 괜찮다.

# 2. 탐색적 데이터 분석
# 데이터 안의 숨겨진 의미를 찾기 위해, 시각화 라이브러리를 사용하여
# 각 데이터를 컬럼별로 탐색해보고자 한다.

# 2.1 Pclass
# Pclass에 대해서 살펴보자, Pclass는 ordinal, 서수형 데이터이다.
# 카테고리이면서, 순서가 존재하는 데이터 타입이다.

# I) Pclass에 따른 생존률의 차이를 살펴보자
# 엑셀의 피벗차트와 유사한 작업이며, pandas dataframe에서는
# groupby, pivot이라는 메소드를 사용한다.
# 아래와 같이 pclass별 count()를 통해 클래스별로 총 승객수를 알 수 있고
# sum()을 통해 생존한 승객수를 pclass별로 알아낼 수 있다.

df[['Pclass','Survived']].groupby('Pclass', as_index=True).count()
'''
        Survived
Pclass          
1            216
2            184
3            491
'''
df[['Pclass','Survived']].groupby('Pclass', as_index=True).sum()
'''
        Survived
Pclass          
1            136
2             87
3            119
'''

# pandas의 crosstab을 사용하면 위의 과정을 조금더 수월하게 확인할 수 있다
sur_mat = pd.crosstab(df['Pclass'], df['Survived'], margins=True)
'''
Survived    0    1  All
Pclass                 
1          80  136  216
2          97   87  184
3         372  119  491
All       549  342  891
'''

# 그룹객체에 mean()메소드를 사용하면, 각 클래스별 생존률을 알 수 있다.
class_ratio= df[['Pclass', 'Survived']].groupby(['Pclass'], as_index=True).mean().sort_values(by='Survived', ascending=False)
'''
        Survived
Pclass          
1       0.629630
2       0.472826
3       0.242363
'''
# plot.bar() 메소드를 통해 그래프로 확인할 수 있다.
class_ratio.plot.bar()

# 그래프의 결과를 보면 Pclass가 좋을수록(1st) 생존률이 높은 것을 알 수 있다.
# 조금더 보기 쉽게 seaborn의 countplot을 이용하면, 특정 label에 따른 개수를 확인할 수 있다.
y_position = 1.02
f, ax = plt.subplots(1, 2, figsize=(18, 8))
df['Pclass'].value_counts().plot.bar(color=['#CD7F32','#FFDF00','#D3D3D3'], ax=ax[0])
ax[0].set_title('Number of Passengers By Pclass', y=y_position)
ax[0].set_ylabel('Count')
sns.countplot('Pclass', hue='Survived', data=df, ax=ax[1])
ax[1].set_title('Pclass: Survived vs Dead', y=y_position)
plt.show()

# 클래스가 높을 수록 생존확률이 높은 것을 알 수 있다.
# 1 : 63% , 2 : 48% , 3: 25% 로 생존여부를 파악하는 모델을 만들고자 할때
# PClass는 큰 영향을 주는 feature로 사용하는것이 좋다고 판단할 수 있는 결론을 내릴 수 있다.


# 2.2 Sex
# 성별로 생존률이 어떻게 달라지는지 확인해 보자
# groupby 와 seaborn countplot을 사용하도록 하겠다.
f, ax = plt.subplots(1, 2, figsize=(18, 8))
df[['Sex', 'Survived']].groupby(['Sex'], as_index=True).mean().plot.bar(ax=ax[0])
ax[0].set_title('Survived vs Sex')
sns.countplot('Sex', hue='Survived', data=df, ax=ax[1])
ax[1].set_title('Sex: Survived vs Dead')
plt.show()

#  결과로 보아 여자가 남자보다 생존율이 높다는 것을 알 수 있다.
df[['Sex','Survived']].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False)
'''
      Sex  Survived
0  female  0.742038
1    male  0.188908
'''
pd.crosstab(df['Sex'], df['Survived'], margins=True)
'''
Survived    0    1  All
Sex                    
female     81  233  314
male      468  109  577
All       549  342  891
'''

# Sex 컬럼도 Pclass와 마찬가지로 생존율에 영향력이 큰 변수라는 것을 판단할 수 있다.

# 2.3 Sex 와 Pclass두가지에 따른 생존율을 비교해보자
# seaborn의 factorplot을 사용하여 손쉽게 3차원 그래프를 통해 확인해보도록하자.

sns.factorplot('Pclass','Survived', hue='Sex', data=df, size =6, aspect=1.5)

# pclass의 모든 항목에서 여성이 남성보다 살 확률이 높다는 것을 알 수 있다.
# 또한 남자, 여자 상관없이 클래스가 높을 수록 살확률이 높다는것도 알 수 있다.
# factorplot의 파라미터중 hue 대신에 column으로 변경하면 아래와 같은 그림로 확인할 수 있다.
sns.factorplot(x='Sex',y='Survived',col='Pclass', data=df, satureation=.5, size=9, aspect=1)


# 2.4 Age
# 나이컬럼에 대하여 확인해 보자
print("나이가 제일 많은 탑승객의 나이 :  {:.1f} Years".format(df['Age'].max()))
# 나이가 제일 많은 탑승객의 나이 :  80.0 Years
print("나이가 제일 어린 탑승객의 나이 :  {:.1f} Years".format(df['Age'].min()))
# 나이가 제일 어린 탑승객의 나이 :  0.4 Years
print("탑승객 평균 나이 :  {:.1f} Years".format(df['Age']. mean()))
# 탑승객 평균 나이 :  29.7 Years

# 그렇다면 생존에 따른 나이 Histogram을 그려보자.

fig, ax = plt.subplots(1, 1, figsize=(9, 5))
sns.kdeplot(df[df['Survived'] == 1]['Age'], ax=ax)
sns.kdeplot(df[df['Survived'] == 0]['Age'], ax=ax)
plt.legend(['Survived == 1', 'Survived == 0'])
plt.show()

# 40살 미만에서 생존자의 수가 많은 것을 알 수 있다.

# Pclass에 따른 나이 hitogram을 그려보자
plt.figure(figsize=(8, 6))
df['Age'][df['Pclass'] == 1].plot(kind='kde')
df['Age'][df['Pclass'] == 2].plot(kind='kde')
df['Age'][df['Pclass'] == 3].plot(kind='kde')

plt.xlabel('Age')
plt.title('Age Distribution within classes')
plt.legend(['1st Class', '2nd Class', '3rd Class'])

# 그래프를 확인해보면, Class가 높을 수록 나이가 많은 사람의 비중이 높아지는것을 확인할 수 있다.
# 나이대가 ㅏ변하면 생존률이 어떻게 되는지 확인해보고싶다.
# 나이 범위를 점점 넓혀가며 생존률의 변화율을 확인해보자
cummulate_survival_ratio = []
for i in range(1, 80):
    cummulate_survival_ratio.append(df[df['Age'] < i]['Survived'].sum() / len(df[df['Age'] < i]['Survived']))

plt.figure(figsize=(7,7))
plt.plot(cummulate_survival_ratio)
plt.title('Survival rate change depending on ragne of Age', y=1.02)
plt.ylabel('Survival rate')
plt.xlabel('Range of Age(0~80)')
plt.show()

# 나이가 어릴 수록 생존률이 확실히 높은 것을 확인할 수 있다.
# 이를 통하여 Age 컬럼 역시 생존율을 예측하는데 중요한 Feature로 사용될 수 있음을 확인할 수 있다.

# 2.5 Pcalss, Sex, Age
# 지금까지 살펴본 Pclass, Sex, Age는 모두 생존여부에 영향력이 큰 변수라고 판단하였다.
# 세가지 컬럼 모두에 대하여 생존여부에 대한 정보를 확인해보고자 한다.
# seaborn의 violonplot을 사용하면 위의 그래프를 쉽게 확인해 볼수 있다.
# x 축은 나눠서 보고싶어하는 case(Pclass, Sex)를 나타내며, y축은 보고싶은 분포(Age)를 나타낼 것이다.
f, ax= plt.subplots(1, 2, figsize=(18, 8))
sns.violinplot("Pclass", "Age", hue="Survived", data=df, scale='count', split=True, ax = ax[0])
ax[0].set_title('Pclass and Age VS Survived')
ax[0].set_yticks(range(0,110,10))
sns.violinplot("Sex", "Age", hue='Survived', data=df, scale='count', split=True, ax=ax[1])
ax[1].set_title('Sex and Age VS Survived')
ax[1].set_yticks(range(0,110,10))
plt.show()


# 2.6 Embarked
# 탑승한 항구에 따른 생존률을 확인해 보도록 하자
f, ax = plt.subplots(1, 1, figsize=(7, 7))
df[['Embarked', 'Survived']].groupby(['Embarked'], as_index=True).mean().sort_values(by='Survived', ascending=False).plot.bar(ax=ax)

# 그래프를 보았을때 각 탑승 항구마다 생존률의 차이는 존재하나
# 앞의 다른 Feature들에 비해서는 차이가 적다는 것을 알 수 있다.
# 어느정도 생존율에 영향을 줄 수 있다고 보이기 때문에 사용하도록한다.

# 이번에는 countplot을 사용하여 다른 feature들로 나누어 살펴보도록하자
f, ax = plt.subplots(2, 2, figsize=(20,15))
sns.countplot('Embarked', data=df, ax=ax[0,0])
ax[0,0].set_title('(1) Number Of Passengers Boarded') # 항구별 탑승자수
sns.countplot('Embarked', hue='Sex', data=df, ax = ax[0,1])
ax[0,1].set_title("(2) Male-Female Split for Embarked") # 항구별 여성 남성 탑승자 수
sns.countplot('Embarked', hue= 'Survived', data=df, ax=ax[1,0])
ax[1,0].set_title("(3) Embarked vs Survived") # 탑승항구별 생존, 사망자수
sns.countplot('Embarked', hue='Pclass', data=df, ax=ax[1,1])
ax[1,1].set_title("(4) Embarked vs Pclass") # 탑승항구별 클래스 탑승자수


# (1) S 항구에서 가장 많은 사람이 탑승했다.
# (2) C, Q 항구에서는 남녀 탑승객 비율이 비슷하며, S 항구에서는 남자 탑승객 수가 많다.
# (3) S항구에서 탄 탑승객의 생존자수가 가장 많은 것을 알 수 있다.
# (4) Class별로 분할하니 C class의 생존자가 많은 이유는 1st class의 탑승객이 많고,
# S 항구의 생존율이 낮은 이유는 3rd Class의 탑승객이 많기 때문이라는 사실을 알 수있다.

# 2.7 Family - SibSp(형제 자매) + Parch(부모, 자녀)
# SibSp 와 Parch를 합하면 가족수를 알 수 있을 것이다.
# 가족수에 대하여 데이터를 분석해 보도록 하자
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1  # 자기 자신이 포함되어야 하므로 1을 더한다.
print("최대 가족 구성원 수: ", df["FamilySize"].max())
# 최대 가족 구성원 수:  11
print("최소 가족 구성원 수: ", df["FamilySize"].min())
# 최소 가족 구성원 수:  1

# 그렇다면 가족구성원 수에 따른 생존 관계를 분석해보자
f, ax = plt.subplots(1,3, figsize=(40,10))
sns.countplot('FamilySize', data=df, ax=ax[0])
ax[0].set_title('(1) Number Of passengers Boarded', y=1.02) # 가족구성원수에 따른 탑승객 수

sns.countplot('FamilySize', hue='Survived', data=df, ax=ax[1])
ax[1].set_title('(2) Survived countplot depending on FamilySize', y= 1.02) # 가족구성원수에 따른 생존,사망 승객 수

df[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=True).mean().sort_values(by='Survived', ascending=False).plot.bar(ax=ax[2])
ax[2].set_title('(3) Survived rate depending on FamilySize', y= 1.02) # 가족구성원 수에 따른 생존율

# (1) 가족구성원은 1~11까지 있음을 알 수 있다. 대부분 1~3명의 가족구성원의 탑승객에 제일 많다.
# (2),(3) - 가족구성원수에 따른 생존비교한 그래프이다. 가족구성원수가 4명인 경우가 가장 생존확률이 높은것을
# 알 수있다. 가족수가 많아질 수록 생존확률이 낮아진다는 사실을 알수 있다. 또한 1명일 경우의 생존율도 낮다는
# 사실을 통해 가족구성원 수는 생존확률에 영향을 주는 feature라고 생각할 수 있다.

# 2.8 Fare
# 탑승요금이며, 연속형 Feature이다. Histogram을 통해 Feature의 분포를 확인해보자
fig, ax = plt.subplots(1, 1, figsize=(8,8))
g = sns.distplot(df['Fare'], color='b', label='Skeness : {:.2f}'.format(df['Fare'].skew()), ax = ax)
g = g.legend(loc='best')

# Histogram을 확인해보면 매우 비대칭인 모습의 분포를 가지고 있다는 것을 알 수 있다.
# Fare를 그대로 모델 학습에 사용한다면, 모델이 잘못 학습할 수 도 있는 위험성이 존재한다.
# 몇 없는 outlier에 민감하게 반응해, 실제 예측시 좋지 못한 결과를 부를수 있다.
# 이런 outlier의 영향을 줄이기 위해, Feature Engineering의 방법중 하나인 Scalingㅇ를 사용하도록 하겠다.
# 여기서는 log 스케일을 취하도록 하겠다.
# pandas의 컬럼에 공통 함수를 적용하기 위해, map, apply 함수를 사용할 것이다.
df.loc[df.Fare.isnull(), 'Fare'] = df['Fare'].mean()
# Null 값이 존재하는 컬럼을 Fare의 평균으로 채우도록 하겠다.

# 값이 0보다 크면 값에 log를 취하고, 0보다 작으면 0으로 값 대체
df['Fare'] = df['Fare'].map(lambda i: np.log(i) if i > 0 else 0)

fig, ax = plt.subplots(1,1, figsize=(8,8))
g = sns.distplot(df['Fare'], color='b', label='Skewness : {:.2f}'.format(df['Fare'].skew()), ax= ax )
g = g.legend(loc='best')

# log를 취해 두 분포를 비교해보면, 비대칭성이 많이 사라진 것을 확인할 수 있다.
# 이러한 Scale 작업을 통해 모델이 좀더 좋은 성능을 내도록 만들 수 있다.
# 모델의 성능을 높히기 위해 feature를 추가하는 것을 feature engineering이라고 한다.



# 2.9 Cabin
# 맨처음 Feature별 NaN값을 확인햇을 때 Cabin은 대략 80%의 컬럼에 값이 존재하지 않았다.
# 생존에 영향을 미칠 중요한 정보를 얻어내기 어려운 Feature이므로 학습에서 제외하도록 하겠다.
print("Cabin의 NaN의 비율 : {:.2f}%".format(df['Cabin'].isnull().sum()/len(df)*100))
# Cabin의 NaN의 비율 : 77.10%


# 2.10 Ticket
# Ticket Feature는 NaN은 존재하지 않으나, string 타입의 Feature이므로, 추가적인 작업을 통해 의미를 찾아야한다.
# 이를 위해서는 다양한 아이디어가 필요하며, 데이터 분석의 경험이 깊어질 수록 다양한 방법의 노하우가 생길 것이다.
df['Ticket'].value_counts()
'''
CA. 2343           7
347082             7
1601               7
3101295            6
CA 2144            6
                  ..
36967              1
PC 17754           1
365222             1
SOTON/OQ 392089    1
237798             1
'''

#https://kaggle-kr.tistory.com/17?category=868316

