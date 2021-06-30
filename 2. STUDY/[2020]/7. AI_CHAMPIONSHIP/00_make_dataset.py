import pandas as pd

PATH = "C:/Users/wook1/Documents/WAI/2020/02.K스타트업_빅데이터경진대회/개발/"

#################################################
'''  

TEST 데이터 생성
validation 데이터의 끝의 일주일을 검증 데이터로 활용
validation max ord_dt 20200915
일별 업체별 어뷰징 케이스 상관없이 어뷰징 여부 판단
그룹핑 : shop_no, ord_dt, max(abuse_yn) 
기간 : 2020-09-09 ~ 2020-09-15

'''

# 1. 데이터 로드 및 데이터 확인
valid_df = pd.read_csv(PATH + "data/validation.csv",index_col=False, error_bad_lines=False, low_memory= False)

len(valid_df)
# validation data : 4,116,382 건
valid_df.columns
# Index(['shop_no', 'ord_dt', 'abuse_yn', 'abuse_class'], dtype='object')
valid_df["ord_dt"].max()
# 2020-09-15
# 기간 연산을 위한 날짜 커럼 임시 처리
valid_df["ord_dt_ch"] = valid_df["ord_dt"].str.replace("-","").apply(int)

answer_df = valid_df.loc[valid_df.ord_dt >= "2020-09-09"]
answer_df.drop("abuse_class", axis = 1,inplace=True)
answer_df = answer_df.groupby(["shop_no","ord_dt"]).max()
answer_df= answer_df.reset_index()
print(answer_df.groupby("abuse_yn").size().reset_index(name="count"))
answer_df.to_csv(PATH + "answer_df.csv",index=False, encoding="utf-8")
# 2. Validation set & Test Set Split
# valid data
valid_df = valid_df.loc[valid_df["ord_dt_ch"] < valid_df["ord_dt_ch"].max()-6]
valid_df.to_csv(PATH + "data/prep/validation.csv",index=False, encoding="utf-8")

# test data
# 끝의 1주일만 뽑기
test_sub_df = valid_df.loc[valid_df["ord_dt_ch"] >= valid_df["ord_dt_ch"].max()-6]
# test data  889,973건
len(test_sub_df)
print(test_sub_df.groupby("abuse_yn").size().reset_index(name = 'count'))
# abusing case를 고려한 target
#    abuse_yn   count
# 0         0  888117
# 1         1    1856
# 일별 업체별 어뷰징 여부
test_df = test_sub_df.groupby(["shop_no","ord_dt","abuse_yn"]).max()
len(test_df)
# 79,190 건 전체
print(test_df.groupby("abuse_yn").size().reset_index(name = 'count'))
# 일별 업체별 어뷰징 여부
#    abuse_yn  count
# 0         0  79165
# 1         1     25
# test data 는 max ord_dt 마지막 일주일 생성
test_df.ord_dt_ch.unique()
test_df = test_df.sort_values("ord_dt_ch")
test_df = test_df.reset_index()
test_df.drop(columns = {"ord_dt_ch","abuse_class"}, inplace=True, axis=0)
len(test_df)
test_df.head()
print(test_df.groupby("abuse_yn").size().reset_index(name = 'count'))
test_df.to_csv(PATH + "data/prep/test_data.csv", index=False, encoding="utf-8")



