import pandas
import numpy

PATH = "C:/Users/WAI/OneDrive/03.WORKSPACE/00.DATA/99.KAGGLE/insurance/14. Learning from Imbalaced Insurance Data/"
DNAME = "aug_train.csv"

train_df = pandas.read_csv(PATH + "aug_train.csv", index_col=False)

# 전체 데이터 셋 요약 정보
print("{0} 은 {1} 건의 데이터와 {2} 개의 컬럼으로 구성되어 있습니다.".format(DNAME.split(".")[0], train_df.shape[0], train_df.shape[1]))
type(train_df.info())
pandas.DataFrame(train_df.info())
train_df.shape[0]

# 변수 데이터 타입 확인
# 연속형, 범주형 변수 구분





import inspect
inspect.getsource()
