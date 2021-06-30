import numpy as np
import pandas as pd
from pandas import Series
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('seaborn')
sns.set(font_scale=1.5)
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls

#ignore warnings
import warnings
warnings.filterwarnings('ignore')

# 최대 줄 수 설정
pd.set_option('display.max_rows', 500)
# 최대 열 수 설정
pd.set_option('display.max_columns', 500)
# 표시할 가로의 길이
pd.set_option('display.width', 1000)




df = pd.read_csv("G:/내 드라이브/WAI_WORK/분석교재/data/titanic.csv")

# 가족구성원수라는 새로운 Feature를 생성
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1 # 자신을 포함해야하니 1을 더합니다

# 탐색적 데이터 분석을 통해 한쪽으로 비대칭적인 Fare Feature를 균일한 분포로 변환하기위한
# Log Scaling 적용
df.loc[df.Fare.isnull(), 'Fare'] = df['Fare'].mean()
df['Fare'] = df['Fare'].map(lambda i: np.log(i) if i > 0 else 0)
df.columns
# 3. Feature enginnering
# 탐색적 데이터 분석을 통해 각각의 Feature들에 대한 정보를 파악하였다.
# 지금부터 본격적인 Feature engineering을 시작해보려한다.
# 가장 먼저 dataset에 존재한s null data를 채우는 것부터 시작한다.
# 아무 숫자나 채우는 것이아니라, null data를 포함하는 feature의 통계치를 참고하거나, 다양한 기법을 사용하여 채울 수 있다.
# null 값을 어떻게 채우느냐에 따라 모델의 성능이 좌지우지 될수 있다. 교재 앞부분에서 설명한 기본적인 내용을 바탕으로
# 실제 null data를 채우는 feature engineering을 수행하도록 하겠다.
# feature engineering은 실제 모델 학습에 쓰려고 하는 것이기 때문에, train set 과 test set 도 동일 하게 적용해야 한다.
# 이점을 주의하며 시작하도록 하자.


# https://kaggle-kr.tistory.com/18?category=868316



# 3.1 Fill Null
# 3.1.1 Age 컬럼의 Null 값을 title을 이용하여 채우기
# Age 에는 177개의 Null 값이 존재한다는 것을 탐색적 데이터 분석을 통해 알 수 있다.
# Age를 채우는 값을 우리는 title이라는 변수를 만들어 age의 통계값과 함께 사용하는 방법으로 Null 값을 채울 것이다.
# Name의 str을 정규표현식으로 추출하여, Initial을 추출해 새로운 컬럼으로 만들겠다.

df['Initial'] = df.Name.str.extract('([A-Za-z]+)\.')
# crosstab을 이용하여 Initail과 Sex별 갯수를 확안해보자
pd.crosstab(df['Initial'], df['Sex'])

# 남녀에 따른 호칭을 구분해 볼 수 있다.
# 다양한 분류의 호칭을 남성, 여성, 기타 3가지로 변환하여 단순화 시키도록하자

df['Initial'].replace(['Mlle','Mme','Ms','Dr','Major','Lady','Countess','Jonkheer','Col','Rev','Capt','Sir','Don', 'Dona'],
                        ['Miss','Miss','Miss','Mr','Mr','Mrs','Mrs','Other','Other','Other','Mr','Mr','Mr', 'Mr'],inplace=True)

df.groupby('Initial').mean()
'''
         PassengerId  Survived    Pclass        Age     SibSp     Parch      Fare  FamilySize
Initial                                                                                      
Master    414.975000  0.575000  2.625000   4.574167  2.300000  1.375000  3.340710    4.675000
Miss      411.741935  0.704301  2.284946  21.860000  0.698925  0.537634  3.123713    2.236559
Mr        455.880907  0.162571  2.381853  32.739609  0.293006  0.151229  2.651507    1.444234
Mrs       456.393701  0.795276  1.984252  35.981818  0.692913  0.818898  3.443751    2.511811
Other     564.444444  0.111111  1.666667  45.888889  0.111111  0.111111  2.641605    1.222222
'''

# 여성과 관계있는 Miss, Mrs의 생존율이 다른 호칭들 보다 높은 것을 알 수 있다.

df.groupby('Initial')['Survived'].mean().plot.bar()

# 이제 Age 의 Null값을 채우기 위한 기준을 정하기 위한 데이터인 Initial을 만들었다.
# 본격적으로 Null 값을 채워보자. Null 값을 채우는 방법은 정말 다양하다.
# 통계값을 활용하는 방법이 있고, null data가 없는 데이터를 기반으로 새로운 알고리즘을 만들어 예측하는 방법도 있다.
# 우리는 기본적으로 통계값을 활용한 방법을 사용할 것이다.
# train data의 통계값을 활용하여, train, test의 null 값을 보정해야한다.
# 이유는 test data는 언제나 unseen으로 두어야 하는 데이터이기 때문이다.

df.groupby('Initial').mean()
'''
         PassengerId  Survived    Pclass        Age     SibSp     Parch      Fare  FamilySize
Initial                                                                                      
Master    414.975000  0.575000  2.625000   4.574167  2.300000  1.375000  3.340710    4.675000
Miss      411.741935  0.704301  2.284946  21.860000  0.698925  0.537634  3.123713    2.236559
Mr        455.880907  0.162571  2.381853  32.739609  0.293006  0.151229  2.651507    1.444234
Mrs       456.393701  0.795276  1.984252  35.981818  0.692913  0.818898  3.443751    2.511811
Other     564.444444  0.111111  1.666667  45.888889  0.111111  0.111111  2.641605    1.222222
'''
# 각 호칭의 Age 평균을 이용하여 null 값을 채우도록 해보자
df.loc[(df.Age.isnull())&(df.Initial=='Mr'),'Age'] = 33
df.loc[(df.Age.isnull())&(df.Initial=='Mrs'),'Age'] = 36
df.loc[(df.Age.isnull())&(df.Initial=='Master'),'Age'] = 5
df.loc[(df.Age.isnull())&(df.Initial=='Miss'),'Age'] = 22
df.loc[(df.Age.isnull())&(df.Initial=='Other'),'Age'] = 46

# 위 코드의 loc + boolean + column 을 사용하여 값을 치환하는 방법은 자주쓰이는 방법으로
# 익숙해져야한다.
# 각 호칭의 조건별로 Age 의 평균으로 null 값은 채우는 코드이다.


# 3.1.2 Fill Null in Embarked
print("Embarked에는 총 ", df["Embarked"].isnull().sum(), "개의 Null 값이 존재한다.")
# Embarked에는 총  2 개의 Null 값이 존재한다.
# Embarked는 가장 많이 존재하는 값으로 Null 값을 채울 것이다.
# 즉, 최빈값을 사용하여 Null 값은 채우는 방법이다.

df[['Embarked', 'PassengerId']].groupby(['Embarked']).count()
'''
Embarked             
C                 168
Q                  77
S                 644

'''
# Embarked에는 S의 값이 가장 많이 존재하므로, S로 값을 채우도록 하겠다.
df['Embarked'].fillna('S' , inplace=True)

# 3.2 Change Age(연속형 변수의 범주화)
# Age는 현재 연속형 Feature이다. 그대롤 사용하여 모델을 생성할 수 있지만,
# Age를 몇개의 group으로 나누어 범주화 시켜줄 수 있다.
# 연속형 변수를 범주화 시키게 되면 정보의 손실이 발생할 수 도 있다는 점을 고려해야한다.
# loc함수를 사용하거나, apply 함수를 사용하여 만들고자 하는 간격대로 범주화 해줄 수 있다.

# 10간격으로 범주를 나누어 보도록 하자
# loc메소드를 사용하여 조건에 의한 범주를 나누는 방법이다.
df['Age_cat'] = 0
df.loc[df['Age'] < 10, 'Age_cat'] = 0
df.loc[(10 <= df['Age']) & (df['Age'] < 20), 'Age_cat'] = 1
df.loc[(20 <= df['Age']) & (df['Age'] < 30), 'Age_cat'] = 2
df.loc[(30 <= df['Age']) & (df['Age'] < 40), 'Age_cat'] = 3
df.loc[(40 <= df['Age']) & (df['Age'] < 50), 'Age_cat'] = 4
df.loc[(50 <= df['Age']) & (df['Age'] < 60), 'Age_cat'] = 5
df.loc[(60 <= df['Age']) & (df['Age'] < 70), 'Age_cat'] = 6
df.loc[70 <= df['Age'], 'Age_cat'] = 7

# 각 나이의 구간을 부등호로 표현해야한다는 번거로움이 존재한다.
# apply 함수를 사용하여, 사용자 정의 함수를 만들어 작업하는 방법은 다음과 같다
def category_age(x):
    if x < 10:
        return 0
    elif x < 20:
        return 1
    elif x < 30:
        return 2
    elif x < 40:
        return 3
    elif x < 50:
        return 4
    elif x < 60:
        return 5
    elif x <70:
        return 6
    else:
        return 7

df[''] = df['Age'].apply(category_age)

# 두가지 방법을 사용하여 연속형 변수를 범주화 시켜주었다.
# 두 결과가 동일한지 확인해 보도록하자
print('첫번째 방법, 두번째 방법 둘다 같은 결과이면 True 반환 -> ', (df["Age_cat"] == df["Age_cat_2"]).all())
# 첫번째 방법, 두번째 방법 둘다 같은 결과이면 True 반환 ->  True

# all() 메소드를 사용하여 전체 데이터에 대한 비교후 모든값이 True이면 True를 반환하고
# 하나라도 False 가 존재하면 False를 반환한다.
# 모두 True인것이 확인 되었으므로 둘중 편한 방법을 사용하면된다.
# 그렇다면 중복되는 컬럼인 Age_cat_2와 원래 Age를 제거하도록 하겠다.
df.drop(['Age','Age_cat_2'], axis=1, inplace=True)

# 3.3 Initial, Embarked and Sex 변환
# 문자를 숫자로 변환
# 현재 Initial은 Mr, Miss, Mrs, Master, Other 총 5개의 범주로 이루어져 있다.
# 이러한 string Feature를 그대로 사용하며 모델에 사용할 수 없다.
# 컴퓨터가 인식할 수 있도록 string변수들을 수치화 시켜주어야 한다.
# map() 메소드를 통하여 간단히 변환해 줄 수 있다.
df['Initial'] = df['Initial'].map({'Master':0, 'Miss' : 1, 'Mr':2., 'Mrs': 3, 'Other': 5 })

print("Embarked의 범주는 {} 로 이루어져 있다.".format(df['Embarked'].unique()))
# Embarked의 범주는 ['S' 'C' 'Q'] 로 이루어져 있다.
# 동일하게 map 메소드를 활용하여 수치형 변수로 변환해주도록 하겠다.
df['Embarked'] = df['Embarked'].map({'C': 0, 'Q':1, 'S':2})

# any() 메소드를 활용하여 Embarked 컬럼에 Null 이 존재하는지 확인해보도록 하자.
df['Embarked'].isnull().any()
# False를 반환하기 때문에 Null 값이 모두 처리된 것을 확인할 수 있다.

# Sex는 female 과 male로 이루어져있다. map을 활용하여 동일하게 변환해 보자.
df['Sex'] = df['Sex'].map({'female':0, 'male' : 1})

# 모델이 학습을 할 수 있도록 데이터의 String 형태를 수치형으로 변환하는 작업을 수행하였다.
# 매우 기본적인 Feature engineering 기법이고 필수적으로 사용되는 방법이기 때문에 꼭 익혀두길 바란다.

# 다음으로 각 feature간의 상관관계를 확인해 보자.
# 분류 모델이기 때문에 각 Feature들의 상관성을 고려하는 것은 의미있는 작업은 아니지만,
# 상관계수분석결과를 손쉽게 확인할 수 있는 방법을 설명하기 위해 다루도록 하겠다.
# dataframe의 corr() 메소드를 통해 상관계수를 구하고 seaborn의 heatmap plot을 사용하면,
# 각 feature들간의 상관계수를 -1~1까지 쉽게 판단할 수 있다.

heatmap_data = df[['Survived', 'Pclass', 'Sex', 'Fare', 'Embarked', 'FamilySize', 'Initial', 'Age_cat']]

colormap = plt.cm.RdBu
plt.figure(figsize=(14,12))
plt.title('Pearson Correlation of Feature', y=1.05, size = 15)
sns.heatmap(heatmap_data.astype(float).corr(), linewidths=0.1, vmax=1.0, square=True, cmap=colormap,
            linecolor='White', annot=True, annot_kws={"size":16})

del heatmap_data

# 결과를 확인해보면, 앞서 탐색적 데이터 분석을 통해 살펴보았듯이
# Sex 와 Pclass가 Survived에 상관관계가 어느정도 있다는 것을 알 수 있다.
# 생각보다 fare 와 Embarked도 상관관계가 있음도 알 수 있다.
# 또한 상관관계분석을 통해 알 수 있는 것은 매우 강한 상관관계를 가지는 feature가 없다는 것을 파악할 수 있다.
# 이를 통해 모델을 학습 시킬때, 불필요한 feature가 없다는 것을 의미한다는 것을 알 수 있다.

# 실제로 모델을 학습시키기 이전에 data preprocessing(전처리)를 진행해보자

# 3.4 One-hot encoding on Initial and Embarked
# 수치화시킨 카테고리 데이터를 그대로 사용해도 괜찮지만 모델의 성능을 높히기 위해 one-hot encoding을 해줄 수 있다.
# 수치화는 간단하게 각 범주별 데이터를 각각 수치로 변화하여 매핑해주는 것을 의미하지만
# One-hot encoding은 각 카테고리를 범주의 갯수만큼 차원 벡터를 늘려서 표현하는 것을 의미한다.

# 각 범주수만큼 차원을 늘려 벡터를 생성할 수도 있으나.
# pandas의 get_dummies 메소드를 사용하여 매우 쉽게 생성할 수 있다.
# Initial의 범주는 총 5개이므로 5개의 새로운 컬럼이 생성되고, prefix를 Initial로 지정하여 구분하기 쉽게 할 수 있다.
df = pd.get_dummies(df, columns=['Initial'], prefix='Initial')
df.head()
'''
   Initial_0.0  Initial_1.0  Initial_2.0  Initial_3.0  Initial_5.0
0            0            0            1            0            0
1            0            0            0            1            0
2            0            1            0            0            0
3            0            0            0            1            0
4            0            0            1            0            0
'''

# Embarked도 마찬가지로 one-Hot Encoding을 사용하여 더미변수를 생성해보자
df = pd.get_dummies(df, columns=['Embarked'], prefix='Embarked')
df.head()
'''
  Embarked_0  Embarked_1  Embarked_2
0          0           0           1
1          1           0           0
2          0           0           1
3          0           0           1
4          0           0           1
'''

# 아주 쉽게 one-hot encoding이 적용된 것을 확인할 수 있다.
# sklearn의 Labelencoder + OneHotencoder를 사용하여 처리할 수 도있다.
# 하지만 범주가 100개 이상되는 데이터도 존재하기 때문에, one-hot encoding을 하게되면 새로운 컬럼이 100개 추가된다.
# 이렇게 될경우 모델 학습 성능의 문제가 발생할 수 도 있기 때문에, 차원을 축소하거나, 범주를 줄이는 방법이 사용된다.

# 3.5 Drop columns
# 이제 필요한 학습 데이터만 남기고 필요없는 데이터는 삭제하도록 하자

df.drop(['PassengerId','Name','SibSp','Parch','Ticket','Cabin'], axis=1, inplace=True)
df.head()
'''
   Survived  Pclass  Sex      Fare  FamilySize  Age_cat  Initial_0.0  Initial_1.0  Initial_2.0  Initial_3.0  Initial_5.0  Embarked_0  Embarked_1  Embarked_2
0         0       3    1  1.981001           2        2            0            0            1            0            0           0           0           1
1         1       1    0  4.266662           2        3            0            0            0            1            0           1           0           0
2         1       3    0  2.070022           1        2            0            1            0            0            0           0           0           1
3         1       1    0  3.972177           2        3            0            0            0            1            0           0           0           1
4         0       3    1  2.085672           1        3            0            0            1            0            0           0           0           1

'''
# Index를 나타내는 PassengersID, Name, 각 파생변수를 만들기 위해 사용된 변수, Null 값이 많아 정보가 없는 feature를
# 제거하여 최종적인 학습데이터를 생성하였다.
# 매우 기본적인 내용을 다루었으며, 각 데이터와 도메인에 맞는 방법으로 다양한 Feature Engineering 방법들이 존재한다.
# 다양한 데이터와 도메인에 대한 분석을 통하여 다양한 방법을 터득할 수 있기 위해서는 기본이 가장 중요하다는 사실을
# 잊지말자.
