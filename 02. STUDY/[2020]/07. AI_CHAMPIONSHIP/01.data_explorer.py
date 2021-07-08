import pandas as pd
import numpy as np
import seaborn as sns
import warnings
warnings.filterwarnings(action="ignore")

PATH = "C:/Users/wook1/Documents/WAI/2020/02.K스타트업_빅데이터경진대회/개발/"

# Train Data Load

train_df = pd.read_csv(PATH + "/data/prep/train_data_rep.csv",index_col=False, error_bad_lines=False,low_memory=False)

#train_df = train_df.reset_index()
# train_df.drop("index",inplace=True, axis=1)
'''
df = pd.read_csv(PATH + "/data/prep/train_rep.csv",index_col=False, error_bad_lines=False,low_memory=False)
df.groupby("purch_method_cd").count()
'''
# 결제 수단 13가지
'''
purch_method_cd                                     ...                   
0D030528F9669EC43F5BF4EACB2D61F0    70670    70670  ...     70670    70670
0F7EED938F73E3923772B491020201D7   539950   539950  ...    539950   539950
1ADE919D61FC2A11B45B6DEF7D562BE7   754472   754472  ...    754472   754472
2F694937A67A24F1A30D75A46A7A3995   323549   323549  ...    323549   323549
3959395607506196088098ABA999DDA9  1216453  1216453  ...   1216453  1216453
4DBF08AB9B8B7E6987637525DB9C3169     3866     3866  ...      3866     3866
68A43CD87222DF230160D6654DF02F67  4164124  4164124  ...   4164124  4164124
69DAB047525629EAF3544F084C72FAAA      237      237  ...       237      237
9DC7B78948C851A91A15EFF3483D62C1  1610722  1610722  ...   1610722  1610722
BE7E1ECC7A6641814F3AA56BE5E23BC6     3831     3831  ...      3831     3831
EAA0B719F8BCF1559B7EFA288FED88C4   876777   876777  ...    876777   876777
EE695C70C5C8D75FC4CB2F89350CD6D2   395677   395677  ...    395677   395677
F1881B8847BD259DC2314120426646AA   684143   684143  ...    684143   684143
[13 rows x 23 columns]
'''
# 컬럼 정보
'''
ord_no : 주문정보
ci_seq : 주문자의 ci 순번
mem_no : 주문자의 회원번호
dvc_id : 주문자의 디바이스 번호
shop_no : 주문한 가게의 가게번호
shop_owner_no : 주문한 가게의 업주번호 
rgn1_cd : 주문지역
rgn2_cd : 주문지역
rgn3_cd : 주문지역
ord_msg : 주문 관련 요청 메세지 내용
ord_tm : 주문시간
item_name : 주문메뉴
item_quantity : 총 메뉴 주문 수량
cpn_use_cnt : 주문자의 쿠폰사용수
ord_price : 메뉴별 주문 수량 * 메뉴 금액(할인금액 미반영, 최종결제금액 아님)
purch_method_cd : 주문자의 결재수단(암호화) 카드/만나서카드/카카오페이 등
review_yn : 주문자의 리뷰작성 여부
rating : 주문자가 리뷰작성시 생성한 리뷰 점수
review_created_tm : 주문 작성 시간
image_review_yn : 리뷰 이미지 삽입여부
delevery_yn : 배달/배민오더 여부 확인
ord_prog_cd : 주문 완료/최소 등에 대한 정보
ord_dt : 주문 일자
'''

# validation set 과 동일한 기간의 train_df 생성
# 기본적인 형태를 validation set의 형태로 변환
# 주문에 대한 어뷰징은 알 수 없고 업체별 일별 어뷰징유형별 여부 최종목표
# train을 업체별 일별 컬럼으로 전처리 필요
train_df = pd.read_csv(PATH + "/data/prep/train_data_rep.csv", index_col=False, low_memory=False)
# 1. 건수 확인
# 1-1  전체 데이터 건수
print("전체 데이터 : {} 건".format(len(train_df)))
# 학습데이터 : 4,261,9063건
# 1-2. 주문 업체 수
print("업체 수 : {}".format(len(train_df["shop_no"].unique())))
# 업체 수 : 14,588
# purch_method_cd : 13개 케이스 존재
# 배민페이,신용/체크카드, 휴대폰결제, 네이버페이, 카카오페이, 토스, 만나서 카드결제, 만나서 현금결제 외 6개 존재
train_df.groupby("purch_method_cd").count()


# 정규표현식으로 datetime의 mili second 제거, review_created_tm, ord_tm
train_df.ord_tm = train_df.ord_tm.str.replace("[.][\d]+","",regex=True)
train_df.review_created_tm = train_df.review_created_tm.str.replace("[.][\d]+","",regex=True)
# time_grouping을 위한 년월일, 시분초 분할
time_df = train_df.ord_tm.str.split(" ",expand=True)
time_df.columns = ["month","time"]
train_df["order_tm_hms"] = time_df["time"]
train_df = train_df.sort_values(["shop_no","ord_tm"])
train_df = train_df.reset_index()
train_df.drop("index",axis=1, inplace=True)
train_df.to_csv(PATH + "/data/prep/train_data_rep_1.csv", index= False, encoding="utf-8")
# 시간단위 데이터 전처리
del train_df
del time_df
###########################################################################################################


train_mn_df = pd.read_csv(PATH + "/data/prep/train_data_rep_1.csv", index_col=False, low_memory=False )
train_mn_df.groupby("review_yn").size().reset_index(name = "count")
'''
   review_yn    count
0          0  3563691
1          1   698212
'''
#문자열 컬럼 제거
train_mn_df = train_mn_df.drop(["ord_msg","item_name"],axis =1)
#del train_df
train_mn_df.columns
'''
Index(['ord_no', 'ci_seq', 'mem_no', 'dvc_id', 'shop_no', 'shop_owner_no',
       'rgn1_cd', 'rgn2_cd', 'rgn3_cd', 'ord_tm', 'item_quantity',
       'cpn_use_cnt', 'ord_price', 'purch_method_cd', 'review_yn', 'rating',
       'review_created_tm', 'image_review_yn', 'delivery_yn', 'ord_prog_cd',
       'ord_date', 'ord_dt', 'order_tm_hms']
'''
# 정렬
#train_mn_df = train_mn_df.sort_values(["shop_no","ord_tm"])
train_mn_df.head()
# reindex 는 정렬결과를 가져오는것이아니라 index의 순으로 정리한다.
#train_mn_df = train_mn_df.reindex(range(0,len(train_mn_df)))

#train_mn_df.drop("index",inplace=True,axis=1)

# -------------문자열 컬럼 데잍터 정합성 체크----------------------#
# Find Data no Matched in columns
import re
def find_not_match(x):
    if re.match("[가-힣]",x):
        print(x)
train_mn_df['ord_no'].apply(find_not_match)

miss_columns_data  = []
for idx, str in enumerate(train_mn_df["ord_no"]):
    if re.match("[가-힣]",str):
        print("{0}은 {1}로 잘못된 데이터 입니다.".format(idx,str))
        miss_columns_data.append(idx)

#train_mn_df.drop(miss_columns_data,inplace=True)
#train_mn_df = train_mn_df.reindex(range(len(train_mn_df)))

# columns check
columns_check = ['ord_no', 'ci_seq', 'mem_no', 'dvc_id', 'shop_no', 'shop_owner_no',
       'rgn1_cd', 'rgn2_cd', 'rgn3_cd']
for i in columns_check:
    print(i)

    for idx, str in enumerate(train_mn_df[i]):
        try:
            if re.match("[가-힣]",str):
                print("{0}컬럼의 {1}은 {2}로 잘못된 데이터 입니다.".format(i,idx,str))
        except:
            print("Error_columns : {0}, index : {1}".format(i,idx))


# ord no 이외의 컬럼의 이상한 값 없음
# 컬럼확인
train_mn_df["ord_prog_cd"].unique()
# array(['주문완료', '주문취소'], dtype=object)
train_mn_df["delivery_yn"].unique()
#array(['배달', '배민오더'], dtype=object)

# 컬럼 값 정수 변환
# 1) delivery_yn
train_mn_df.loc[train_mn_df.delivery_yn == "배달", "delivery_yn"] = 1
train_mn_df.loc[train_mn_df.delivery_yn == "배민오더", "delivery_yn"] = 2
# 1: 배달 , 2: 배민오더
# 2) ord_prog_cd
train_mn_df.loc[train_mn_df.ord_prog_cd == "주문완료", "ord_prog_cd"] = 1
train_mn_df.loc[train_mn_df.ord_prog_cd == "주문취소", "ord_prog_cd"] = 0
# 1: 주문완료, 0 : 주문취소
train_mn_df["ord_prog_cd"].sum()
# 주문완료 건수 : 4,080,755  주문취소 건수 : 4,261,905 - 4,080,755

# 지역은 개인 주문 고객 별로 각각의 고유값으로 부여(지역으로 묶이지 않음)
len(train_mn_df["rgn1_cd"].unique()) # 지역 1 : 15 도/특별시/광역시
len(train_mn_df["rgn2_cd"].unique()) # 지역 2 : 141 시/군/구
len(train_mn_df["rgn3_cd"].unique()) # 지역 3 : 1844  읍/면/동

# 지역별 업체수 수, 지역별 업체 주문수 동단위까지 분할?(생각)

# 중복된 주문 번호 확인
ord_no_ck = train_mn_df.groupby("ord_no").count()["ci_seq"]
ord_no_ck = ord_no_ck.reset_index()
ord_no_ck.loc[ord_no_ck["ci_seq"] == 2]
dup_data = train_mn_df.loc[train_mn_df["ord_no"] == "E5A78396E525C7EB4B75F7306D80C49B"]
dup_data.iloc[0, :]
# 1806975,1806976 중복 데이터
train_mn_df.drop(index=1806976,axis=1, inplace=True)
train_mn_df = train_mn_df.reset_index()
train_mn_df.drop("index",axis=1,inplace=True)

# 주문자정보_한회원번호에 여러가지 기기번호를 가지고 있는 사용자 여부 파악
len(train_mn_df["ord_no"].unique()) # 주문 건수 : 4,261,902 건
len(train_mn_df["ci_seq"].unique()) # 주문자의 ci 순번 : 2,549,241
len(train_mn_df["mem_no"].unique()) # 주문자의 회원번호 : 2,563,836
len(train_mn_df["dvc_id"].unique()) # 주문자의 div 번호 : 2,631,584`
train_mn_df.groupby("mem_no").count()["dvc_id"].to_csv(PATH + "data/dev_val/dvc_id.csv", encoding="cp949")

# 같은날 한 회원이 동일한 가게에서 다른 기종으로 주문한 경우 Case 건수 확인 변수화


# 전체 리뷰 작성 건수(주문완료건에 비해 생각 보다 작성된 리뷰는 적음)
train_mn_df["review_yn"].sum() # 698,211 건
# 이미지 포함 리뷰 작성 건수
train_mn_df["image_review_yn"].sum() # 600,313 건
# 698211
train_mn_df[["review_yn", "review_created_tm","ord_tm"]].loc[train_mn_df["review_yn"] == 1]
# 3563690
train_mn_df[["review_yn", "review_created_tm","ord_tm"]].loc[train_mn_df["review_created_tm"] == "\\N"]
# 3,563,690건
train_mn_df[["review_yn", "review_created_tm","ord_tm"]].loc[train_mn_df["review_yn"] == 0]
print(train_mn_df.groupby("review_yn").size().reset_index(name = 'count'))
'''
   review_yn    count
0        0.0  3563690
1        1.0   698211
'''
train_mn_df.to_csv(PATH + "data/prep/train_prep_01.csv", index=False, encoding="utf-8")
##############################################################################################################

# 주문시간 - 리뷰시간 리뷰작성까지 걸리는 시간 컬럼생성
import datetime
import math

def _time_delta(x,y):
    try:
        t1 = datetime.datetime.strptime(x , '%Y-%m-%d %H:%M:%S')
        t2 = datetime.datetime.strptime(y, '%Y-%m-%d %H:%M:%S')
        return ((t1-t2).days *24 + (math.ceil((t1 - t2).seconds / 3600)))
    except:
        print("함수에러")


def _print(x):
    print(x)

train_mn_df.tail()
# 698,212 건

train_mn_df[["review_yn","review_created_tm","ord_tm"]][1:2]

# NaN으로 값이 모두 처리되 어있는 CASE
np.where(train_mn_df.review_yn.isnull())
train_mn_df.iloc[1806976]
train_mn_df.drop(1806976,inplace=True)
train_mn_df.review_yn = train_mn_df.review_yn.apply(int)
train_mn_df = train_mn_df.reset_index()
train_mn_df.drop("index",axis=1, inplace=True)

#print(train_mn_df["review_created_tm"][9999], train_mn_df["ord_tm"][9999])
len(train_mn_df.loc[train_mn_df.ord_review_delta_tm == 0])
np.where(train_mn_df.loc[train_mn_df.ord_review_delta_tm == 0])
[2351405,4261899,4261900]
train_mn_df.iloc[1806976]
train_mn_df[["review_yn","review_created_tm","ord_tm","ord_review_delta_tm"]].iloc[2351405]
# 4261899,4261900
train_mn_df[["review_yn","ord_review_delta_tm"]].iloc[2351406]
train_mn_df[["review_yn","ord_review_delta_tm"]].iloc[4261900]
train_mn_df[["ord_tm","review_created_tm"]].iloc[4261900]
train_mn_df["ord_review_delta_tm"][2351405] = _time_delta(train_mn_df["review_created_tm"][2351405], train_mn_df["ord_tm"][2351405])

t1 = datetime.datetime.strptime(train_mn_df["review_created_tm"][2351405], '%Y-%m-%d %H:%M:%S')
t2 = datetime.datetime.strptime(train_mn_df["ord_tm"][2351405], '%Y-%m-%d %H:%M:%S')
print((t1-t2).days *24 + (math.ceil((t1 - t2).seconds /60)))
train_mn_df["ord_review_delta_tm"].isnull().sum()



#-----------------------------------------------#
train_mn_df["ord_review_delta_tm"] = 0
loof_count = 0
for i in range(0,len(train_mn_df)):
    try:
        if train_mn_df["review_yn"][i] == 0:
            train_mn_df["ord_review_delta_tm"][i] = 9999
        else:
            #train_mn_df["ord_review_delta_tm"][i] = _time_delta(train_mn_df["review_created_tm"][i], train_mn_df["ord_tm"][i])

            try:
                # print(train_mn_df["review_created_tm"][i],  train_mn_df["ord_tm"][i])
                train_mn_df["ord_review_delta_tm"][i] = _time_delta(train_mn_df["review_created_tm"][i], train_mn_df["ord_tm"][i])
            except:
                # 주문 시간이 없는데 리뷰 작성시간은 존재
                print(train_mn_df["review_created_tm"][i],train_mn_df["ord_tm"][i])
    except:
        print("error index : {}".format(i))
    loof_count += 1
    if loof_count % 10000 == 0:
        print("--------------------",loof_count,"--------------------")

train_mn_df.to_csv(PATH + "data/dev_val/train_prep_2.csv", index=False, encoding="utf-8")
#-------------------------------------------------------#


for i in range(0, len(train_mn_df)):
    if re.match("\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}", train_mn_df["review_created_tm"][i]):
        pass;
    else:
        print(train_mn_df["review_created_tm"][i])






# 2. 변수 그룹핑

# 2.1 업체별 일별 주문 금액, 구문 메뉴수
train_mn_df[["delivery_yn","ord_prog_cd"]]
train_mn_df.groupby(["shop_no","ord_dt","ord_prog_cd"]).count()["ord_no"]


# Total Group Sum
# 1. 업체별,
train_tot_sum = train_mn_df.groupby(["shop_no","ord_dt"]).sum()[["ord_price", "item_quantity", "cpn_use_cnt", "review_yn","image_review_yn", "ord_prog_cd"]]
train_tot_count = train_mn_df.groupby[["shop_no","ord_dt"]].count()["ord_prog_yn", "review_yn","image_review_yn", ]
train_tot_sum.to_csv(PATH + "data/dev_val/tot_sum.csv",index= True, encoding="utf-8")

train_mn_df.head()
def _split_tm(x):
    return x.split(":")[0]

train_mn_df["ord_tmz"] = train_mn_df.order_tm_hms.apply(_split_tm)
train_mn_df["ord_tmz"] = train_mn_df.ord_tmz.apply(int)
test_pivot = pd.pivot_table(train_mn_df, index = ["shop_no","ord_dt"] ,
               values = ["item_quantity","ord_price","cpn_use_cnt", "review_yn"],
               columns = "ord_tmz",
               aggfunc=['sum', 'max','min','mean','count', "std"])

test_pivot = test_pivot.reset_index()
test_pivot = test_pivot.fillna(0)
test_pivot = test_pivot.astype(int)

test_pivot = test_pivot.reset_index()
test_pivot = test_pivot.set_index(["shop_no", "ord_dt"])

test_pivot.xs(["sum", "item_quantity"], level=0, axis=1)
test_pivot.xs(["max", "item_quantity"], level=0, axis=1)
test_pivot.xs(["min", "item_quantity"], level=0, axis=1)
test_pivot.xs(["mean", "item_quantity"], level=0, axis=1)
test_pivot.xs(["count", "item_quantity"], level=0, axis=1)


test_pivot.xs("ord_price", level=0, axis =1)
test_pivot.xs("cpn_use_cnt", level=0, axis =1)
test_pivot.xs("review_yn", level=0, axis =1)
train_mn_df.head()

test = train_mn_df.sort_values(["shop_no","ord_dt"])

test[:30000].to_csv(PATH + "data/dev_val/pivot_test.csv",index=False, encoding= "utf-8")