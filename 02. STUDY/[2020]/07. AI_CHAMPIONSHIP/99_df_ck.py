import pandas as pd
import re
DATA_PATH = "/data/prep/train_rep.csv"
PATH = "C:/Users/wook1/Documents/WAI/2020/02.K스타트업_빅데이터경진대회/개발/"


df = pd.read_csv(PATH + DATA_PATH, index_col=False, low_memory=False)
df.groupby("review_yn").size().reset_index(name="count")

miss_columns_data  = []
for idx, str in enumerate(df["ord_no"]):
    if re.match("[가-힣]",str):
        print("{0}은 {1}로 잘못된 데이터 입니다.".format(idx,str))
        miss_columns_data.append(idx)

df.drop(miss_columns_data, inplace= True)


# validation set과 동일한 기간의 data 추출
train_df = df.loc[df["ord_dt"]>= "2020-08-15"]
train_df["ord_dt"].min()
train_df["ord_dt"].max()
train_df.to_csv(PATH + "data/prep/train_data_rep.csv", index=False, encoding="utf-8")

# estimator set
estimator_df = df.loc[df["ord_dt"] < "2020-08-15"]
estimator_df.to_csv(PATH + "data/prep/estimator_data_rep.csv", index=False, encoding="utf-8")


del df
del estimator_df

for i in range(0,10000):
    print(i)
