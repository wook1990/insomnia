import pandas as pd


def line_logging(*messages):
    import datetime
    import sys
    today = datetime.datetime.today()
    log_time = today.strftime('[%Y/%m/%d %H:%M:%S]')
    log = []
    for message in messages:
        log.append(str(message))
    print(log_time + ':[' + ', '.join(log) + ']')
    sys.stdout.flush()

# data Load
DATA_PATH = "/workspace/data/"
SAVE_PATH = "/workspace/data/gr_dev_var/"


#DATA_PATH = "E:/"
#SAVE_PATH = "E:/test/"

line_logging("Data Load")
df = pd.read_csv(DATA_PATH + "train_data_prep_3.csv", index_col=False, low_memory=False)
#df.loc[df.ord_review_delta_tm == 9999, "ord_review_delta_tm"] = 0
#df.to_csv(DATA_PATH + "train_data_prep_3.csv",  index=False, encoding="utf-8")
#df = df.loc[df.ord_dt >= "2020-08-15"]
#####################################################################################################
line_logging("Data split")
# data slit
# 1) 주문완료
line_logging("주문완료")
ord_ok_df = df.loc[df["ord_prog_cd"] == 1]
# 1-1) 배달주문완료
line_logging("배달주문완료")
ord_ok_del_df = df.loc[(df["ord_prog_cd"] == 1)&(df["delivery_yn"]==1)]
# 1-2) 배민오더주문완료
line_logging("배민오더주문완료")
ord_ok_baemin_df = df.loc[(df["ord_prog_cd"] == 1)&(df["delivery_yn"]==2)]

# 2) 주문취소
line_logging("주문취소")
ord_cc_df = df.loc[df["ord_prog_cd"] == 0]
# 2-1) 배달주문취소
line_logging("배달주문취소")
ord_cc_del_df = df.loc[(df["ord_prog_cd"] == 0)&(df["delivery_yn"]==1)]
# 2-2) 배민오더주문취소
line_logging("배민오더주문취소")
ord_cc_baemin_df = df.loc[(df["ord_prog_cd"] == 0)&(df["delivery_yn"]==2)]

######################################################################################################
######################################################################################################
# group by (전체) 주문수량, 금액, 쿠폰사용

line_logging("전체 그룹 파생 변수 생성")
df_gr = df[["shop_no","ord_dt","item_quantity","ord_price","cpn_use_cnt"]].groupby(["shop_no","ord_dt"]).agg(["count","sum","min","max","std","mean"])
line_logging("테이블 {}의 NA 값 처리".format("df_gr"))
df_gr = df_gr.fillna(0)
line_logging("테이블 {} 데이터 저장".format("df_gr"))
df_gr.to_csv(SAVE_PATH + "{}.csv".format("df_gr"), index=True, encoding="utf-8")
line_logging("테이블 {} 작업완료 데이터 삭제".format("df_gr"))
del df_gr

# 1) 주문완료
line_logging("주문 완료 그룹 파생 변수 생성")
ord_ok_df_gr = ord_ok_df[["shop_no","ord_dt","item_quantity","ord_price","cpn_use_cnt"]].groupby(["shop_no","ord_dt"]).agg(["count","sum","min","max","std","mean"])
line_logging("테이블 {}의 NA 값 처리".format("ord_ok_df_gr"))
ord_ok_df_gr = ord_ok_df_gr.fillna(0)
line_logging("테이블 {} 데이터 저장".format("ord_ok_df_gr"))
ord_ok_df_gr.to_csv(SAVE_PATH + "{}.csv".format("ord_ok_df_gr"), index=True, encoding="utf-8")
line_logging("테이블 {} 작업완료 데이터 삭제".format("ord_ok_df_gr"))
del ord_ok_df_gr

# 1-1) 배달주문완료
line_logging("배달 주문 완료 파생 변수 생성")
ord_ok_del_df_gr = ord_ok_del_df[["shop_no","ord_dt","item_quantity","ord_price","cpn_use_cnt"]].groupby(["shop_no","ord_dt"]).agg(["count","sum","min","max","std","mean"])

line_logging("테이블 {}의 NA 값 처리".format("ord_ok_df_gr"))
ord_ok_del_df_gr = ord_ok_del_df_gr.fillna(0)
line_logging("테이블 {} 데이터 저장".format("ord_ok_del_df_gr"))
ord_ok_del_df_gr.to_csv(SAVE_PATH + "{}.csv".format("ord_ok_del_df_gr"), index=True, encoding="utf-8")
line_logging("테이블 {} 작업완료 데이터 삭제".format("ord_ok_del_df_gr"))
del ord_ok_del_df_gr

# 1-2) 배민오더주문완료
line_logging("배민오더 주문 완료 파생 변수 생성")
ord_ok_baemin_df_gr = ord_ok_baemin_df[["shop_no","ord_dt","item_quantity","ord_price","cpn_use_cnt"]].groupby(["shop_no","ord_dt"]).agg(["count","sum","min","max","std","mean"])

line_logging("테이블 {}의 NA 값 처리".format("ord_ok_baemin_df_gr"))
ord_ok_baemin_df_gr = ord_ok_baemin_df_gr.fillna(0)
line_logging("테이블 {} 데이터 저장".format("ord_ok_baemin_df_gr"))
ord_ok_baemin_df_gr.to_csv(SAVE_PATH + "{}.csv".format("ord_ok_baemin_df_gr"), index=True, encoding="utf-8")
line_logging("테이블 {} 작업완료 데이터 삭제".format("ord_ok_baemin_df_gr"))
del ord_ok_baemin_df_gr

# 2) 주문취소
line_logging("주문 취소 파생 변수 생성")
ord_cc_df_gr = ord_cc_df[["shop_no","ord_dt","item_quantity","ord_price","cpn_use_cnt"]].groupby(["shop_no","ord_dt"]).agg(["count","sum","min","max","std","mean"])

line_logging("테이블 {}의 NA 값 처리".format("ord_cc_df_gr"))
ord_cc_df_gr = ord_cc_df_gr.fillna(0)
line_logging("테이블 {} 데이터 저장".format("ord_cc_df_gr"))
ord_cc_df_gr.to_csv(SAVE_PATH + "{}.csv".format("ord_cc_df_gr"), index=True, encoding="utf-8")
line_logging("테이블 {} 작업완료 데이터 삭제".format("ord_cc_df_gr"))
del ord_cc_df_gr

# 2-1) 배달주문취소
line_logging("배달 주문 취소 파생 변수 생성")
ord_cc_del_df_gr = ord_cc_del_df[["shop_no","ord_dt","item_quantity","ord_price","cpn_use_cnt"]].groupby(["shop_no","ord_dt"]).agg(["count","sum","min","max","std","mean"])

line_logging("테이블 {}의 NA 값 처리".format("ord_cc_del_df_gr"))
ord_cc_del_df_gr = ord_cc_del_df_gr.fillna(0)
line_logging("테이블 {} 데이터 저장".format("ord_cc_del_df_gr"))
ord_cc_del_df_gr.to_csv(SAVE_PATH + "{}.csv".format("ord_cc_del_df_gr"), index=True, encoding="utf-8")
line_logging("테이블 {} 작업완료 데이터 삭제".format("ord_cc_del_df_gr"))
del ord_cc_del_df_gr


# 2-2) 배민오더주문취소
line_logging("배민오더 주문 취소 파생 변수 생성")
ord_cc_baemin_df_gr = ord_cc_baemin_df[["shop_no","ord_dt","item_quantity","ord_price","cpn_use_cnt"]].groupby(["shop_no","ord_dt"]).agg(["count","sum","min","max","std","mean"])

line_logging("테이블 {}의 NA 값 처리".format("ord_cc_baemin_df_gr"))
ord_cc_baemin_df_gr = ord_cc_baemin_df_gr.fillna(0)
line_logging("테이블 {} 데이터 저장".format("ord_cc_baemin_df_gr"))
ord_cc_baemin_df_gr.to_csv(SAVE_PATH + "{}.csv".format("ord_cc_baemin_df_gr"), index=True, encoding="utf-8")
line_logging("테이블 {} 작업완료 데이터 삭제".format("ord_cc_baemin_df_gr"))
del ord_cc_baemin_df_gr

######################################################################################################
######################################################################################################
# 리뷰, 이미지리뷰, 작성건수( 1 의 합)
# 1) 주문완료
line_logging("주문완료 리뷰 그룹 파생 변수 생성")
ord_ok_review_df = ord_ok_df[["shop_no","ord_dt","review_yn","image_review_yn"]].groupby(["shop_no","ord_dt"]).sum()

line_logging("테이블 {}의 NA 값 처리".format("ord_ok_review_df"))
ord_ok_review_df = ord_ok_review_df.fillna(0)
line_logging("테이블 {} 데이터 저장".format("ord_ok_review_df"))
ord_ok_review_df.to_csv(SAVE_PATH + "{}.csv".format("ord_ok_review_df"), index=True, encoding="utf-8")
line_logging("테이블 {} 작업완료 데이터 삭제".format("ord_ok_review_df"))
del ord_ok_review_df

# 1-1) 배달주문완료
line_logging("배달주문완료 리뷰 그룹 파생 변수 생성")
ord_ok_del_review_df = ord_ok_del_df[["shop_no","ord_dt","review_yn","image_review_yn"]].groupby(["shop_no","ord_dt"]).sum()

line_logging("테이블 {}의 NA 값 처리".format("ord_ok_del_review_df"))
ord_ok_del_review_df = ord_ok_del_review_df.fillna(0)
line_logging("테이블 {} 데이터 저장".format("ord_ok_del_review_df"))
ord_ok_del_review_df.to_csv(SAVE_PATH + "{}.csv".format("ord_ok_del_review_df"), index=True, encoding="utf-8")
line_logging("테이블 {} 작업완료 데이터 삭제".format("ord_ok_del_review_df"))
del ord_ok_del_review_df

# 1-2) 배민오더주문완료
line_logging("배민오더 주문완료 리뷰 그룹 파생 변수 생성")
ord_ok_baemin_review_df = ord_ok_baemin_df[["shop_no","ord_dt","review_yn","image_review_yn"]].groupby(["shop_no","ord_dt"]).sum()

line_logging("테이블 {}의 NA 값 처리".format("ord_ok_baemin_review_df"))
ord_ok_baemin_review_df = ord_ok_baemin_review_df.fillna(0)
line_logging("테이블 {} 데이터 저장".format("ord_ok_baemin_review_df"))
ord_ok_baemin_review_df.to_csv(SAVE_PATH + "{}.csv".format("ord_ok_baemin_review_df"), index=True, encoding="utf-8")
line_logging("테이블 {} 작업완료 데이터 삭제".format("ord_ok_baemin_review_df"))
del ord_ok_baemin_review_df

# 2) 주문취소
line_logging("주문취소 리뷰 그룹 파생 변수 생성")
ord_cc_reivew_df = ord_cc_df[["shop_no","ord_dt","review_yn","image_review_yn"]].groupby(["shop_no","ord_dt"]).sum()

line_logging("테이블 {}의 NA 값 처리".format("ord_cc_reivew_df"))
ord_cc_reivew_df = ord_cc_reivew_df.fillna(0)
line_logging("테이블 {} 데이터 저장".format("ord_cc_reivew_df"))
ord_cc_reivew_df.to_csv(SAVE_PATH + "{}.csv".format("ord_cc_reivew_df"), index=True, encoding="utf-8")
line_logging("테이블 {} 작업완료 데이터 삭제".format("ord_cc_reivew_df"))
del ord_cc_reivew_df

# 2-1) 배달주문취소
line_logging("배달주문취소 리뷰 그룹 파생 변수 생성")
ord_cc_del_review_df = ord_cc_del_df[["shop_no","ord_dt","review_yn","image_review_yn"]].groupby(["shop_no","ord_dt"]).sum()

line_logging("테이블 {}의 NA 값 처리".format("ord_cc_del_review_df"))
ord_cc_del_review_df = ord_cc_del_review_df.fillna(0)
line_logging("테이블 {} 데이터 저장".format("ord_cc_del_review_df"))
ord_cc_del_review_df.to_csv(SAVE_PATH + "{}.csv".format("ord_cc_del_review_df"), index=True, encoding="utf-8")
line_logging("테이블 {} 작업완료 데이터 삭제".format("ord_cc_del_review_df"))
del ord_cc_del_review_df


# 2-2) 배민오더주문취소
line_logging("배민오더주문취소 리뷰 그룹 파생 변수 생성")
ord_cc_baemin_review_df = ord_cc_baemin_df[["shop_no","ord_dt","review_yn","image_review_yn"]].groupby(["shop_no","ord_dt"]).sum()

line_logging("테이블 {}의 NA 값 처리".format("ord_cc_baemin_review_df"))
ord_cc_baemin_review_df = ord_cc_baemin_review_df.fillna(0)
line_logging("테이블 {} 데이터 저장".format("ord_cc_baemin_review_df"))
ord_cc_baemin_review_df.to_csv(SAVE_PATH + "{}.csv".format("ord_cc_baemin_review_df"), index=True, encoding="utf-8")
line_logging("테이블 {} 작업완료 데이터 삭제".format("ord_cc_baemin_review_df"))
del ord_cc_baemin_review_df

######################################################################################################
######################################################################################################
# Rating & 리뷰작성까지 시간
# 1) 주문완료
line_logging("주문완료 평점 작성시간 그룹 파생 변수 생성")
ord_ok_rating_tm_df = ord_ok_df[["shop_no","ord_dt","rating","ord_review_delta_tm"]].groupby(["shop_no","ord_dt"]).agg(["sum","min","max","mean","std"])

line_logging("테이블 {}의 NA 값 처리".format("ord_ok_rating_tm_df"))
ord_ok_rating_tm_df = ord_ok_rating_tm_df.fillna(0)
line_logging("테이블 {} 데이터 저장".format("ord_ok_rating_tm_df"))
ord_ok_rating_tm_df.to_csv(SAVE_PATH + "{}.csv".format("ord_ok_rating_tm_df"), index=True, encoding="utf-8")
line_logging("테이블 {} 작업완료 데이터 삭제".format("ord_ok_rating_tm_df"))
del ord_ok_rating_tm_df


# 1-1) 배달주문완료
line_logging("배달주문완료 평점 작성시간 그룹 파생 변수 생성")
ord_ok_del_rating_tm_df = ord_ok_del_df[["shop_no","ord_dt","rating","ord_review_delta_tm"]].groupby(["shop_no","ord_dt"]).agg(["sum","min","max","mean","std"])

line_logging("테이블 {}의 NA 값 처리".format("ord_ok_del_rating_tm_df"))
ord_ok_del_rating_tm_df = ord_ok_del_rating_tm_df.fillna(0)
line_logging("테이블 {} 데이터 저장".format("ord_ok_del_rating_tm_df"))
ord_ok_del_rating_tm_df.to_csv(SAVE_PATH + "{}.csv".format("ord_ok_del_rating_tm_df"), index=True, encoding="utf-8")
line_logging("테이블 {} 작업완료 데이터 삭제".format("ord_ok_del_rating_tm_df"))
del ord_ok_del_rating_tm_df

# 1-2) 배민오더주문완료
line_logging("배민오더주문완료 평점 작성시간 그룹 파생 변수 생성")
ord_ok_baemin_rating_tm_df = ord_ok_baemin_df[["shop_no","ord_dt","rating","ord_review_delta_tm"]].groupby(["shop_no","ord_dt"]).agg(["sum","min","max","mean","std"])

line_logging("테이블 {}의 NA 값 처리".format("ord_ok_baemin_rating_tm_df"))
ord_ok_baemin_rating_tm_df = ord_ok_baemin_rating_tm_df.fillna(0)
line_logging("테이블 {} 데이터 저장".format("ord_ok_baemin_rating_tm_df"))
ord_ok_baemin_rating_tm_df.to_csv(SAVE_PATH + "{}.csv".format("ord_ok_baemin_rating_tm_df"), index=True, encoding="utf-8")
line_logging("테이블 {} 작업완료 데이터 삭제".format("ord_ok_baemin_rating_tm_df"))
del ord_ok_baemin_rating_tm_df

# 2) 주문취소
line_logging("주문취소 평점 작성시간 그룹 파생 변수 생성")
ord_cc_rating_tm_df = ord_cc_df[["shop_no","ord_dt","rating","ord_review_delta_tm"]].groupby(["shop_no","ord_dt"]).agg(["sum","min","max","mean","std"])

line_logging("테이블 {}의 NA 값 처리".format("ord_cc_rating_tm_df"))
ord_cc_rating_tm_df = ord_cc_rating_tm_df.fillna(0)
line_logging("테이블 {} 데이터 저장".format("ord_cc_rating_tm_df"))
ord_cc_rating_tm_df.to_csv(SAVE_PATH + "{}.csv".format("ord_cc_rating_tm_df"), index=True, encoding="utf-8")
line_logging("테이블 {} 작업완료 데이터 삭제".format("ord_cc_rating_tm_df"))
del ord_cc_rating_tm_df

# 2-1) 배달주문취소
line_logging("배달 주문취소 평점 작성시간 그룹 파생 변수 생성")
ord_cc_del_rating_tm_df = ord_cc_del_df[["shop_no","ord_dt","rating","ord_review_delta_tm"]].groupby(["shop_no","ord_dt"]).agg(["sum","min","max","mean","std"])

line_logging("테이블 {}의 NA 값 처리".format("ord_cc_del_rating_tm_df"))
ord_cc_del_rating_tm_df = ord_cc_del_rating_tm_df.fillna(0)
line_logging("테이블 {} 데이터 저장".format("ord_cc_del_rating_tm_df"))
ord_cc_del_rating_tm_df.to_csv(SAVE_PATH + "{}.csv".format("ord_cc_del_rating_tm_df"), index=True, encoding="utf-8")
line_logging("테이블 {} 작업완료 데이터 삭제".format("ord_cc_del_rating_tm_df"))
del ord_cc_del_rating_tm_df

# 2-2) 배민오더주문취소
line_logging("배민오더 주문취소 평점 작성시간 그룹 파생 변수 생성")
ord_cc_baemin_rating_tm_df = ord_cc_baemin_df[["shop_no","ord_dt","rating","ord_review_delta_tm"]].groupby(["shop_no","ord_dt"]).agg(["sum","min","max","mean","std"])

line_logging("테이블 {}의 NA 값 처리".format("ord_cc_baemin_rating_tm_df"))
ord_cc_baemin_rating_tm_df = ord_cc_baemin_rating_tm_df.fillna(0)
line_logging("테이블 {} 데이터 저장".format("ord_cc_baemin_rating_tm_df"))
ord_cc_baemin_rating_tm_df.to_csv(SAVE_PATH + "{}.csv".format("ord_cc_baemin_rating_tm_df"), index=True, encoding="utf-8")
line_logging("테이블 {} 작업완료 데이터 삭제".format("ord_cc_baemin_rating_tm_df"))
del ord_cc_baemin_rating_tm_df

######################################################################################################
######################################################################################################
# 시간대별
# 1. 한시간 단위
# 1) 주문완료
line_logging("시간단위 주문완료 그룹 파생 변수 생성")
tm_ord_ok_df = pd.pivot_table(ord_ok_df,index=["shop_no","ord_dt"],
                        values = ["item_quantity","ord_price","cpn_use_cnt"],
                        columns = "ord_tmz",
                        aggfunc = ["count","sum","min","max","std","mean"])


line_logging("테이블 {}의 NA 값 처리".format("tm_ord_ok_df"))
tm_ord_ok_df = tm_ord_ok_df.fillna(0)
line_logging("테이블 {} 데이터 저장".format("tm_ord_ok_df"))
tm_ord_ok_df.to_csv(SAVE_PATH + "{}.csv".format("tm_ord_ok_df"), index=True, encoding="utf-8")
line_logging("테이블 {} 작업완료 데이터 삭제".format("tm_ord_ok_df"))
del tm_ord_ok_df

# 1-1) 배달주문완료
line_logging("시간단위 배달주문완료 그룹 파생 변수 생성")
tm_ord_ok_del_df = pd.pivot_table(ord_ok_del_df,index=["shop_no","ord_dt"],
                        values = ["item_quantity","ord_price","cpn_use_cnt"],
                        columns = "ord_tmz",
                        aggfunc = ["count","sum","min","max","std","mean"])

line_logging("테이블 {}의 NA 값 처리".format("tm_ord_ok_del_df"))
tm_ord_ok_del_df = tm_ord_ok_del_df.fillna(0)
line_logging("테이블 {} 데이터 저장".format("tm_ord_ok_del_df"))
tm_ord_ok_del_df.to_csv(SAVE_PATH + "{}.csv".format("tm_ord_ok_del_df"), index=True, encoding="utf-8")
line_logging("테이블 {} 작업완료 데이터 삭제".format("tm_ord_ok_del_df"))
del tm_ord_ok_del_df

# 1-2) 배민오더주문완료
line_logging("시간단위 배민오더주문완료 그룹 파생 변수 생성")
tm_ord_ok_baemin_df = pd.pivot_table(ord_ok_baemin_df,index=["shop_no","ord_dt"],
                        values = ["item_quantity","ord_price","cpn_use_cnt"],
                        columns = "ord_tmz",
                        aggfunc = ["count","sum","min","max","std","mean"])

line_logging("테이블 {}의 NA 값 처리".format("tm_ord_ok_baemin_df"))
tm_ord_ok_baemin_df = tm_ord_ok_baemin_df.fillna(0)
line_logging("테이블 {} 데이터 저장".format("tm_ord_ok_baemin_df"))
tm_ord_ok_baemin_df.to_csv(SAVE_PATH + "{}.csv".format("tm_ord_ok_baemin_df"), index=True, encoding="utf-8")
line_logging("테이블 {} 작업완료 데이터 삭제".format("tm_ord_ok_baemin_df"))
del tm_ord_ok_baemin_df


# 2) 주문취소
line_logging("시간단위 주문 취소 그룹 파생 변수 생성")
tm_ord_cc_df  = pd.pivot_table(ord_cc_df,index=["shop_no","ord_dt"],
                        values = ["item_quantity","ord_price","cpn_use_cnt"],
                        columns = "ord_tmz",
                        aggfunc = ["count","sum","min","max","std","mean"])

line_logging("테이블 {}의 NA 값 처리".format("tm_ord_cc_df"))
tm_ord_cc_df = tm_ord_cc_df.fillna(0)
line_logging("테이블 {} 데이터 저장".format("tm_ord_cc_df"))
tm_ord_cc_df.to_csv(SAVE_PATH + "{}.csv".format("tm_ord_cc_df"), index=True, encoding="utf-8")
line_logging("테이블 {} 작업완료 데이터 삭제".format("tm_ord_cc_df"))
del tm_ord_cc_df

# 2-1) 배달주문취소
line_logging("시간단위 배달 주문 취소 그룹 파생 변수 생성")
tm_ord_cc_del_df = pd.pivot_table(ord_cc_del_df,index=["shop_no","ord_dt"],
                        values = ["item_quantity","ord_price","cpn_use_cnt"],
                        columns = "ord_tmz",
                        aggfunc = ["count","sum","min","max","std","mean"])

line_logging("테이블 {}의 NA 값 처리".format("tm_ord_cc_del_df"))
tm_ord_cc_del_df = tm_ord_cc_del_df.fillna(0)
line_logging("테이블 {} 데이터 저장".format("tm_ord_cc_del_df"))
tm_ord_cc_del_df.to_csv(SAVE_PATH + "{}.csv".format("tm_ord_cc_del_df"), index=True, encoding="utf-8")
line_logging("테이블 {} 작업완료 데이터 삭제".format("tm_ord_cc_del_df"))
del tm_ord_cc_del_df

# 2-2) 배민오더주문취소
line_logging("시간단위 배민 오더 주문 취소 그룹 파생 변수 생성")
tm_ord_cc_baemin_df = pd.pivot_table(ord_cc_baemin_df,index=["shop_no","ord_dt"],
                        values = ["item_quantity","ord_price","cpn_use_cnt"],
                        columns = "ord_tmz",
                        aggfunc = ["count","sum","min","max","std","mean"])

line_logging("테이블 {}의 NA 값 처리".format("tm_ord_cc_baemin_df"))
tm_ord_cc_baemin_df = tm_ord_cc_baemin_df.fillna(0)
line_logging("테이블 {} 데이터 저장".format("tm_ord_cc_baemin_df"))
tm_ord_cc_baemin_df.to_csv(SAVE_PATH + "{}.csv".format("tm_ord_cc_baemin_df"), index=True, encoding="utf-8")
line_logging("테이블 {} 작업완료 데이터 삭제".format("tm_ord_cc_baemin_df"))
del tm_ord_cc_baemin_df



#####################################################################################################
#####################################################################################################

# 2. 4시간 단위
# 1) 주문완료
line_logging("4시간단위 주문 완료 그룹 파생 변수 생성")
tm4_ord_ok_df = pd.pivot_table(ord_ok_df,index=["shop_no","ord_dt"],
                        values = ["item_quantity","ord_price","cpn_use_cnt"],
                        columns = "ord_tmz_4h",
                        aggfunc = ["count","sum","min","max","std","mean"])

line_logging("테이블 {}의 NA 값 처리".format("tm4_ord_ok_df"))
tm4_ord_ok_df = tm4_ord_ok_df.fillna(0)
line_logging("테이블 {} 데이터 저장".format("tm4_ord_ok_df"))
tm4_ord_ok_df.to_csv(SAVE_PATH + "{}.csv".format("tm4_ord_ok_df"), index=True, encoding="utf-8")
line_logging("테이블 {} 작업완료 데이터 삭제".format("tm4_ord_ok_df"))
del tm4_ord_ok_df



# 1-1) 배달주문완료
line_logging("4시간단위 배달 주문 완료 그룹 파생 변수 생성")
tm4_ord_ok_del_df = pd.pivot_table(ord_ok_del_df,index=["shop_no","ord_dt"],
                        values = ["item_quantity","ord_price","cpn_use_cnt"],
                        columns = "ord_tmz_4h",
                        aggfunc = ["count","sum","min","max","std","mean"])

line_logging("테이블 {}의 NA 값 처리".format("tm4_ord_ok_del_df"))
tm4_ord_ok_del_df = tm4_ord_ok_del_df.fillna(0)
line_logging("테이블 {} 데이터 저장".format("tm4_ord_ok_del_df"))
tm4_ord_ok_del_df.to_csv(SAVE_PATH + "{}.csv".format("tm4_ord_ok_del_df"), index=True, encoding="utf-8")
line_logging("테이블 {} 작업완료 데이터 삭제".format("tm4_ord_ok_del_df"))
del tm4_ord_ok_del_df

# 1-2) 배민오더주문완료
line_logging("4시간단위 배민오더 주문 완료 그룹 파생 변수 생성")
tm4_ord_ok_baemin_df = pd.pivot_table(ord_ok_baemin_df,index=["shop_no","ord_dt"],
                        values = ["item_quantity","ord_price","cpn_use_cnt"],
                        columns = "ord_tmz_4h",
                        aggfunc = ["count","sum","min","max","std","mean"])

line_logging("테이블 {}의 NA 값 처리".format("tm4_ord_ok_baemin_df"))
tm4_ord_ok_baemin_df = tm4_ord_ok_baemin_df.fillna(0)
line_logging("테이블 {} 데이터 저장".format("tm4_ord_ok_baemin_df"))
tm4_ord_ok_baemin_df.to_csv(SAVE_PATH + "{}.csv".format("tm4_ord_ok_baemin_df"), index=True, encoding="utf-8")
line_logging("테이블 {} 작업완료 데이터 삭제".format("tm4_ord_ok_baemin_df"))
del tm4_ord_ok_baemin_df

# 2) 주문취소
line_logging("4시간단위 주문 취소 그룹 파생 변수 생성")
tm4_ord_cc_df  = pd.pivot_table(ord_cc_df,index=["shop_no","ord_dt"],
                        values = ["item_quantity","ord_price","cpn_use_cnt"],
                        columns = "ord_tmz_4h",
                        aggfunc = ["count","sum","min","max","std","mean"])

line_logging("테이블 {}의 NA 값 처리".format("tm4_ord_cc_df"))
tm4_ord_cc_df = tm4_ord_cc_df.fillna(0)
line_logging("테이블 {} 데이터 저장".format("tm4_ord_cc_df"))
tm4_ord_cc_df.to_csv(SAVE_PATH + "{}.csv".format("tm4_ord_cc_df"), index=True, encoding="utf-8")
line_logging("테이블 {} 작업완료 데이터 삭제".format("tm4_ord_cc_df"))
del tm4_ord_cc_df

# 2-1) 배달주문취소
line_logging("4시간단위 배달 주문 취소 그룹 파생 변수 생성")
tm4_ord_cc_del_df = pd.pivot_table(ord_cc_del_df,index=["shop_no","ord_dt"],
                        values = ["item_quantity","ord_price","cpn_use_cnt"],
                        columns = "ord_tmz_4h",
                        aggfunc = ["count","sum","min","max","std","mean"])

line_logging("테이블 {}의 NA 값 처리".format("tm4_ord_cc_del_df"))
tm4_ord_cc_del_df = tm4_ord_cc_del_df.fillna(0)
line_logging("테이블 {} 데이터 저장".format("tm4_ord_cc_del_df"))
tm4_ord_cc_del_df.to_csv(SAVE_PATH + "{}.csv".format("tm4_ord_cc_del_df"), index=True, encoding="utf-8")
line_logging("테이블 {} 작업완료 데이터 삭제".format("tm4_ord_cc_del_df"))
del tm4_ord_cc_del_df

# 2-2) 배민오더주문취소
line_logging("4시간단위 배민오더 주문 취소 그룹 파생 변수 생성")
tm4_ord_cc_baemin_df = pd.pivot_table(ord_cc_baemin_df,index=["shop_no","ord_dt"],
                        values = ["item_quantity","ord_price","cpn_use_cnt"],
                        columns = "ord_tmz_4h",
                        aggfunc = ["count","sum","min","max","std","mean"])

line_logging("테이블 {}의 NA 값 처리".format("tm4_ord_cc_baemin_df"))
tm4_ord_cc_baemin_df = tm4_ord_cc_baemin_df.fillna(0)
line_logging("테이블 {} 데이터 저장".format("tm4_ord_cc_baemin_df"))
tm4_ord_cc_baemin_df.to_csv(SAVE_PATH + "{}.csv".format("tm4_ord_cc_baemin_df"), index=True, encoding="utf-8")
line_logging("테이블 {} 작업완료 데이터 삭제".format("tm4_ord_cc_baemin_df"))
del tm4_ord_cc_baemin_df

#####################################################################################################
#####################################################################################################

# 2. 6시간 단위
# 1) 주문완료
line_logging("6시간단위 주문 완료 그룹 파생 변수 생성")
tm6_ord_ok_df = pd.pivot_table(ord_ok_df,index=["shop_no","ord_dt"],
                        values = ["item_quantity","ord_price","cpn_use_cnt"],
                        columns = "ord_tmz_6h",
                        aggfunc = ["count","sum","min","max","std","mean"])

line_logging("테이블 {}의 NA 값 처리".format("tm6_ord_ok_df"))
tm6_ord_ok_df = tm6_ord_ok_df.fillna(0)
line_logging("테이블 {} 데이터 저장".format("tm6_ord_ok_df"))
tm6_ord_ok_df.to_csv(SAVE_PATH + "{}.csv".format("tm6_ord_ok_df"), index=True, encoding="utf-8")
line_logging("테이블 {} 작업완료 데이터 삭제".format("tm6_ord_ok_df"))
del tm6_ord_ok_df

# 1-1) 배달주문완료
line_logging("6시간단위 배달 주문 완료 그룹 파생 변수 생성")
tm6_ord_ok_del_df = pd.pivot_table(ord_ok_del_df,index=["shop_no","ord_dt"],
                        values = ["item_quantity","ord_price","cpn_use_cnt"],
                        columns = "ord_tmz_6h",
                        aggfunc = ["count","sum","min","max","std","mean"])

line_logging("테이블 {}의 NA 값 처리".format("tm6_ord_ok_del_df"))
tm6_ord_ok_del_df = tm6_ord_ok_del_df.fillna(0)
line_logging("테이블 {} 데이터 저장".format("tm6_ord_ok_del_df"))
tm6_ord_ok_del_df.to_csv(SAVE_PATH + "{}.csv".format("tm6_ord_ok_del_df"), index=True, encoding="utf-8")
line_logging("테이블 {} 작업완료 데이터 삭제".format("tm6_ord_ok_del_df"))
del tm6_ord_ok_del_df


# 1-2) 배민오더주문완료
line_logging("6시간단위 배민오더 주문 완료 그룹 파생 변수 생성")
tm6_ord_ok_baemin_df = pd.pivot_table(ord_ok_baemin_df,index=["shop_no","ord_dt"],
                        values = ["item_quantity","ord_price","cpn_use_cnt"],
                        columns = "ord_tmz_6h",
                        aggfunc = ["count","sum","min","max","std","mean"])

line_logging("테이블 {}의 NA 값 처리".format("tm6_ord_ok_baemin_df"))
tm6_ord_ok_baemin_df = tm6_ord_ok_baemin_df.fillna(0)
line_logging("테이블 {} 데이터 저장".format("tm6_ord_ok_baemin_df"))
tm6_ord_ok_baemin_df.to_csv(SAVE_PATH + "{}.csv".format("tm6_ord_ok_baemin_df"), index=True, encoding="utf-8")
line_logging("테이블 {} 작업완료 데이터 삭제".format("tm6_ord_ok_baemin_df"))
del tm6_ord_ok_baemin_df


# 2) 주문취소
line_logging("6시간단위  주문 취소 그룹 파생 변수 생성")
tm6_ord_cc_df  = pd.pivot_table(ord_cc_df,index=["shop_no","ord_dt"],
                        values = ["item_quantity","ord_price","cpn_use_cnt"],
                        columns = "ord_tmz_6h",
                        aggfunc = ["count","sum","min","max","std","mean"])

line_logging("테이블 {}의 NA 값 처리".format("tm6_ord_cc_df"))
tm6_ord_cc_df = tm6_ord_cc_df.fillna(0)
line_logging("테이블 {} 데이터 저장".format("tm6_ord_cc_df"))
tm6_ord_cc_df.to_csv(SAVE_PATH + "{}.csv".format("tm6_ord_cc_df"), index=True, encoding="utf-8")
line_logging("테이블 {} 작업완료 데이터 삭제".format("tm6_ord_cc_df"))
del tm6_ord_cc_df


# 2-1) 배달주문취소
line_logging("6시간단위 배달  주문 취소 그룹 파생 변수 생성")
tm6_ord_cc_del_df = pd.pivot_table(ord_cc_del_df,index=["shop_no","ord_dt"],
                        values = ["item_quantity","ord_price","cpn_use_cnt"],
                        columns = "ord_tmz_6h",
                        aggfunc = ["count","sum","min","max","std","mean"])


line_logging("테이블 {}의 NA 값 처리".format("tm6_ord_cc_del_df"))
tm6_ord_cc_del_df = tm6_ord_cc_del_df.fillna(0)
line_logging("테이블 {} 데이터 저장".format("tm6_ord_cc_del_df"))
tm6_ord_cc_del_df.to_csv(SAVE_PATH + "{}.csv".format("tm6_ord_cc_del_df"), index=True, encoding="utf-8")
line_logging("테이블 {} 작업완료 데이터 삭제".format("tm6_ord_cc_del_df"))
del tm6_ord_cc_del_df


# 2-2) 배민오더주문취소
line_logging("6시간단위 배민오더  주문 취소 그룹 파생 변수 생성")
tm6_ord_cc_baemin_df = pd.pivot_table(ord_cc_baemin_df,index=["shop_no","ord_dt"],
                        values = ["item_quantity","ord_price","cpn_use_cnt"],
                        columns = "ord_tmz_6h",
                        aggfunc = ["count","sum","min","max","std","mean"])

line_logging("테이블 {}의 NA 값 처리".format("tm6_ord_cc_baemin_df"))
tm6_ord_cc_baemin_df = tm6_ord_cc_baemin_df.fillna(0)
line_logging("테이블 {} 데이터 저장".format("tm6_ord_cc_baemin_df"))
tm6_ord_cc_baemin_df.to_csv(SAVE_PATH + "{}.csv".format("tm6_ord_cc_baemin_df"), index=True, encoding="utf-8")
line_logging("테이블 {} 작업완료 데이터 삭제".format("tm6_ord_cc_baemin_df"))
del tm6_ord_cc_baemin_df

#####################################################################################################
#####################################################################################################

# 2. 8시간 단위
# 1) 주문완료
line_logging("8시간 단위 주문 완료 그룹 파생 변수 생성")
tm8_ord_ok_df = pd.pivot_table(ord_ok_df,index=["shop_no","ord_dt"],
                        values = ["item_quantity","ord_price","cpn_use_cnt"],
                        columns = "ord_tmz_8h",
                        aggfunc = ["count","sum","min","max","std","mean"])

line_logging("테이블 {}의 NA 값 처리".format("tm8_ord_ok_df"))
tm8_ord_ok_df = tm8_ord_ok_df.fillna(0)
line_logging("테이블 {} 데이터 저장".format("tm8_ord_ok_df"))
tm8_ord_ok_df.to_csv(SAVE_PATH + "{}.csv".format("tm8_ord_ok_df"), index=True, encoding="utf-8")
line_logging("테이블 {} 작업완료 데이터 삭제".format("tm8_ord_ok_df"))
del tm8_ord_ok_df

# 1-1) 배달주문완료
line_logging("8시간 단위 배달 주문 완료 그룹 파생 변수 생성")
tm8_ord_ok_del_df = pd.pivot_table(ord_ok_del_df,index=["shop_no","ord_dt"],
                        values = ["item_quantity","ord_price","cpn_use_cnt"],
                        columns = "ord_tmz_8h",
                        aggfunc = ["count","sum","min","max","std","mean"])

line_logging("테이블 {}의 NA 값 처리".format("tm8_ord_ok_del_df"))
tm8_ord_ok_del_df = tm8_ord_ok_del_df.fillna(0)
line_logging("테이블 {} 데이터 저장".format("tm8_ord_ok_del_df"))
tm8_ord_ok_del_df.to_csv(SAVE_PATH + "{}.csv".format("tm8_ord_ok_del_df"), index=True, encoding="utf-8")
line_logging("테이블 {} 작업완료 데이터 삭제".format("tm8_ord_ok_del_df"))
del tm8_ord_ok_del_df

# 1-2) 배민오더주문완료
line_logging("8시간 단위 배민오더 주문 완료 그룹 파생 변수 생성")
tm8_ord_ok_baemin_df = pd.pivot_table(ord_ok_baemin_df,index=["shop_no","ord_dt"],
                        values = ["item_quantity","ord_price","cpn_use_cnt"],
                        columns = "ord_tmz_8h",
                        aggfunc = ["count","sum","min","max","std","mean"])

line_logging("테이블 {}의 NA 값 처리".format("tm8_ord_ok_baemin_df"))
tm8_ord_ok_baemin_df = tm8_ord_ok_baemin_df.fillna(0)
line_logging("테이블 {} 데이터 저장".format("tm8_ord_ok_baemin_df"))
tm8_ord_ok_baemin_df.to_csv(SAVE_PATH + "{}.csv".format("tm8_ord_ok_baemin_df"), index=True, encoding="utf-8")
line_logging("테이블 {} 작업완료 데이터 삭제".format("tm8_ord_ok_baemin_df"))
del tm8_ord_ok_baemin_df



# 2) 주문취소
line_logging("8시간 단위 주문취소 그룹 파생 변수 생성")
tm8_ord_cc_df = pd.pivot_table(ord_cc_df,index=["shop_no","ord_dt"],
                        values = ["item_quantity","ord_price","cpn_use_cnt"],
                        columns = "ord_tmz_8h",
                        aggfunc = ["count","sum","min","max","std","mean"])

line_logging("테이블 {}의 NA 값 처리".format("tm8_ord_cc_df"))
tm8_ord_cc_df = tm8_ord_cc_df.fillna(0)
line_logging("테이블 {} 데이터 저장".format("tm8_ord_cc_df"))
tm8_ord_cc_df.to_csv(SAVE_PATH + "{}.csv".format("tm8_ord_cc_df"), index=True, encoding="utf-8")
line_logging("테이블 {} 작업완료 데이터 삭제".format("tm8_ord_cc_df"))
del tm8_ord_cc_df

# 2-1) 배달주문취소
line_logging("8시간 단위 배달주문취소 그룹 파생 변수 생성")
tm8_ord_cc_del_df = pd.pivot_table(ord_cc_del_df,index=["shop_no","ord_dt"],
                        values = ["item_quantity","ord_price","cpn_use_cnt"],
                        columns = "ord_tmz_8h",
                        aggfunc = ["count","sum","min","max","std","mean"])

line_logging("테이블 {}의 NA 값 처리".format("tm8_ord_cc_del_df"))
tm8_ord_cc_del_df = tm8_ord_cc_del_df.fillna(0)
line_logging("테이블 {} 데이터 저장".format("tm8_ord_cc_del_df"))
tm8_ord_cc_del_df.to_csv(SAVE_PATH + "{}.csv".format("tm8_ord_cc_del_df"), index=True, encoding="utf-8")
line_logging("테이블 {} 작업완료 데이터 삭제".format("tm8_ord_cc_del_df"))
del tm8_ord_cc_del_df

# 2-2) 배민오더주문취소
line_logging("8시간 단위 배민오더 주문취소 그룹 파생 변수 생성")
tm8_ord_cc_baemin_df = pd.pivot_table(ord_cc_baemin_df,index=["shop_no","ord_dt"],
                        values = ["item_quantity","ord_price","cpn_use_cnt"],
                        columns = "ord_tmz_8h",
                        aggfunc = ["count","sum","min","max","std","mean"])

line_logging("테이블 {}의 NA 값 처리".format("tm8_ord_cc_baemin_df"))
tm8_ord_cc_baemin_df = tm8_ord_cc_baemin_df.fillna(0)
line_logging("테이블 {} 데이터 저장".format("tm8_ord_cc_baemin_df"))
tm8_ord_cc_baemin_df.to_csv(SAVE_PATH + "{}.csv".format("tm8_ord_cc_baemin_df"), index=True, encoding="utf-8")
line_logging("테이블 {} 작업완료 데이터 삭제".format("tm8_ord_cc_baemin_df"))
del tm8_ord_cc_baemin_df
#####################################################################################################
#####################################################################################################

# 2. 12시간 단위
# 1) 주문완료
line_logging("12시간 단위 주문 완료 그룹 파생 변수 생성")
tm12_ord_ok_df = pd.pivot_table(ord_ok_df,index=["shop_no","ord_dt"],
                        values = ["item_quantity","ord_price","cpn_use_cnt"],
                        columns = "ord_tmz_12h",
                        aggfunc = ["count","sum","min","max","std","mean"])

line_logging("테이블 {}의 NA 값 처리".format("tm12_ord_ok_df"))
tm12_ord_ok_df = tm12_ord_ok_df.fillna(0)
line_logging("테이블 {} 데이터 저장".format("tm12_ord_ok_df"))
tm12_ord_ok_df.to_csv(SAVE_PATH + "{}.csv".format("tm12_ord_ok_df"), index=True, encoding="utf-8")
line_logging("테이블 {} 작업완료 데이터 삭제".format("tm12_ord_ok_df"))
del tm12_ord_ok_df

# 1-1) 배달주문완료
line_logging("12시간 단위 배달 주문 완료 그룹 파생 변수 생성")
tm12_ord_ok_del_df = pd.pivot_table(ord_ok_del_df,index=["shop_no", "ord_dt"],
                        values=["item_quantity", "ord_price", "cpn_use_cnt"],
                        columns="ord_tmz_12h",
                        aggfunc=["count", "sum", "min", "max", "std", "mean"])

line_logging("테이블 {}의 NA 값 처리".format("tm12_ord_ok_del_df"))
tm12_ord_ok_del_df = tm12_ord_ok_del_df.fillna(0)
line_logging("테이블 {} 데이터 저장".format("tm12_ord_ok_del_df"))
tm12_ord_ok_del_df.to_csv(SAVE_PATH + "{}.csv".format("tm12_ord_ok_del_df"), index=True, encoding="utf-8")
line_logging("테이블 {} 작업완료 데이터 삭제".format("tm12_ord_ok_del_df"))
del tm12_ord_ok_del_df


# 1-2) 배민오더주문완료
line_logging("12시간 단위 배민오더 주문 완료 그룹 파생 변수 생성")
tm12_ord_ok_baemin_df = pd.pivot_table(ord_ok_baemin_df,index=["shop_no", "ord_dt"],
                        values=["item_quantity","ord_price","cpn_use_cnt"],
                        columns="ord_tmz_12h",
                        aggfunc=["count","sum","min","max","std","mean"])

line_logging("테이블 {}의 NA 값 처리".format("tm12_ord_ok_baemin_df"))
tm12_ord_ok_baemin_df = tm12_ord_ok_baemin_df.fillna(0)
line_logging("테이블 {} 데이터 저장".format("tm12_ord_ok_baemin_df"))
tm12_ord_ok_baemin_df.to_csv(SAVE_PATH + "{}.csv".format("tm12_ord_ok_baemin_df"), index=True, encoding="utf-8")
line_logging("테이블 {} 작업완료 데이터 삭제".format("tm12_ord_ok_baemin_df"))
del tm12_ord_ok_baemin_df


# 2) 주문취소
line_logging("12시간 단위 주문 취소 그룹 파생 변수 생성")
tm12_ord_cc_df  = pd.pivot_table(ord_cc_df,index=["shop_no","ord_dt"],
                        values = ["item_quantity","ord_price","cpn_use_cnt"],
                        columns = "ord_tmz_12h",
                        aggfunc = ["count","sum","min","max","std","mean"])

line_logging("테이블 {}의 NA 값 처리".format("tm12_ord_cc_df"))
tm12_ord_cc_df = tm12_ord_cc_df.fillna(0)
line_logging("테이블 {} 데이터 저장".format("tm12_ord_cc_df"))
tm12_ord_cc_df.to_csv(SAVE_PATH + "{}.csv".format("tm12_ord_cc_df"), index=True, encoding="utf-8")
line_logging("테이블 {} 작업완료 데이터 삭제".format("tm12_ord_cc_df"))
del tm12_ord_cc_df

# 2-1) 배달주문취소
line_logging("12시간 단위 배달 주문 취소 그룹 파생 변수 생성")
tm12_ord_cc_del_df = pd.pivot_table(ord_cc_del_df,index=["shop_no","ord_dt"],
                        values = ["item_quantity","ord_price","cpn_use_cnt"],
                        columns = "ord_tmz_12h",
                        aggfunc = ["count","sum","min","max","std","mean"])


line_logging("테이블 {}의 NA 값 처리".format("tm12_ord_cc_del_df"))
tm12_ord_cc_del_df = tm12_ord_cc_del_df.fillna(0)
line_logging("테이블 {} 데이터 저장".format("tm12_ord_cc_del_df"))
tm12_ord_cc_del_df.to_csv(SAVE_PATH + "{}.csv".format("tm12_ord_cc_del_df"), index=True, encoding="utf-8")
line_logging("테이블 {} 작업완료 데이터 삭제".format("tm12_ord_cc_del_df"))
del tm12_ord_cc_del_df

# 2-2) 배민오더주문취소
line_logging("12시간 단위 배민오더 주문 취소 그룹 파생 변수 생성")
tm12_ord_cc_baemin_df = pd.pivot_table(ord_cc_baemin_df,index=["shop_no","ord_dt"],
                        values = ["item_quantity","ord_price","cpn_use_cnt"],
                        columns = "ord_tmz_12h",
                        aggfunc = ["count","sum","min","max","std","mean"])

line_logging("테이블 {}의 NA 값 처리".format("tm12_ord_cc_baemin_df"))
tm12_ord_cc_baemin_df = tm12_ord_cc_baemin_df.fillna(0)
line_logging("테이블 {} 데이터 저장".format("tm12_ord_cc_baemin_df"))
tm12_ord_cc_baemin_df.to_csv(SAVE_PATH + "{}.csv".format("tm12_ord_cc_baemin_df"), index=True, encoding="utf-8")
line_logging("테이블 {} 작업완료 데이터 삭제".format("tm12_ord_cc_baemin_df"))
del tm12_ord_cc_baemin_df

#####################################################################################################
#####################################################################################################

# 구매방법
# 구매수량, 금액, 쿠폰사용 (min max count, std, mean)
# 1) 주문완료
line_logging("구매방법 주문완료 그룹 파생 변수 생성")
purch_ord_ok_df = pd.pivot_table(ord_ok_df,index=["shop_no","ord_dt"],
                        values = ["item_quantity","ord_price","cpn_use_cnt"],
                        columns = "purch_method_cd_label",
                        aggfunc = ["count","sum","min","max","std","mean"])

line_logging("테이블 {}의 NA 값 처리".format("purch_ord_ok_df"))
purch_ord_ok_df = purch_ord_ok_df.fillna(0)
line_logging("테이블 {} 데이터 저장".format("purch_ord_ok_df"))
purch_ord_ok_df.to_csv(SAVE_PATH + "{}.csv".format("purch_ord_ok_df"), index=True, encoding="utf-8")
line_logging("테이블 {} 작업완료 데이터 삭제".format("purch_ord_ok_df"))
del purch_ord_ok_df

# 1-1) 배달주문완료
line_logging("구매방법 배달주문완료 그룹 파생 변수 생성")
purch_ord_ok_del_df = pd.pivot_table(ord_ok_del_df,index=["shop_no","ord_dt"],
                        values = ["item_quantity","ord_price","cpn_use_cnt"],
                        columns = "purch_method_cd_label",
                        aggfunc = ["count","sum","min","max","std","mean"])

line_logging("테이블 {}의 NA 값 처리".format("purch_ord_ok_del_df"))
purch_ord_ok_del_df = purch_ord_ok_del_df.fillna(0)
line_logging("테이블 {} 데이터 저장".format("purch_ord_ok_del_df"))
purch_ord_ok_del_df.to_csv(SAVE_PATH + "{}.csv".format("purch_ord_ok_del_df"), index=True, encoding="utf-8")
line_logging("테이블 {} 작업완료 데이터 삭제".format("purch_ord_ok_del_df"))
del purch_ord_ok_del_df

# 1-2) 배민오더주문완료
line_logging("구매방법 배민오더주문완료 그룹 파생 변수 생성")
purch_ord_ok_baemin_df = pd.pivot_table(ord_ok_baemin_df,index=["shop_no","ord_dt"],
                        values = ["item_quantity","ord_price","cpn_use_cnt"],
                        columns = "purch_method_cd_label",
                        aggfunc = ["count","sum","min","max","std","mean"])

line_logging("테이블 {}의 NA 값 처리".format("purch_ord_ok_baemin_df"))
purch_ord_ok_baemin_df = purch_ord_ok_baemin_df.fillna(0)
line_logging("테이블 {} 데이터 저장".format("purch_ord_ok_baemin_df"))
purch_ord_ok_baemin_df.to_csv(SAVE_PATH + "{}.csv".format("purch_ord_ok_baemin_df"), index=True, encoding="utf-8")
line_logging("테이블 {} 작업완료 데이터 삭제".format("purch_ord_ok_baemin_df"))
del purch_ord_ok_baemin_df

# 2) 주문취소
line_logging("구매방법 주문취소 그룹 파생 변수 생성")
purch_ord_cc_df  = pd.pivot_table(ord_cc_df,index=["shop_no","ord_dt"],
                        values = ["item_quantity","ord_price","cpn_use_cnt"],
                        columns = "purch_method_cd_label",
                        aggfunc = ["count","sum","min","max","std","mean"])

line_logging("테이블 {}의 NA 값 처리".format("purch_ord_cc_df"))
purch_ord_cc_df = purch_ord_cc_df.fillna(0)
line_logging("테이블 {} 데이터 저장".format("purch_ord_cc_df"))
purch_ord_cc_df.to_csv(SAVE_PATH + "{}.csv".format("purch_ord_cc_df"), index=True, encoding="utf-8")
line_logging("테이블 {} 작업완료 데이터 삭제".format("purch_ord_cc_df"))
del purch_ord_cc_df

# 2-1) 배달주문취소
line_logging("구매방법 배달주문취소 그룹 파생 변수 생성")
purch_ord_cc_del_df = pd.pivot_table(ord_cc_del_df,index=["shop_no","ord_dt"],
                        values = ["item_quantity","ord_price","cpn_use_cnt"],
                        columns = "purch_method_cd_label",
                        aggfunc = ["count","sum","min","max","std","mean"])

line_logging("테이블 {}의 NA 값 처리".format("purch_ord_cc_del_df"))
purch_ord_cc_del_df = purch_ord_cc_del_df.fillna(0)
line_logging("테이블 {} 데이터 저장".format("purch_ord_cc_del_df"))
purch_ord_cc_del_df.to_csv(SAVE_PATH + "{}.csv".format("purch_ord_cc_del_df"), index=True, encoding="utf-8")
line_logging("테이블 {} 작업완료 데이터 삭제".format("purch_ord_cc_del_df"))
del purch_ord_cc_del_df

# 2-2) 배민오더주문취소
line_logging("구매방법 배민오더주문취소 그룹 파생 변수 생성")
purch_ord_cc_baemin_df = pd.pivot_table(ord_cc_baemin_df,index=["shop_no","ord_dt"],
                        values = ["item_quantity","ord_price","cpn_use_cnt"],
                        columns = "purch_method_cd_label",
                        aggfunc = ["count","sum","min","max","std","mean"])

line_logging("테이블 {}의 NA 값 처리".format("purch_ord_cc_baemin_df"))
purch_ord_cc_baemin_df = purch_ord_cc_baemin_df.fillna(0)
line_logging("테이블 {} 데이터 저장".format("purch_ord_cc_baemin_df"))
purch_ord_cc_baemin_df.to_csv(SAVE_PATH + "{}.csv".format("purch_ord_cc_baemin_df"), index=True, encoding="utf-8")
line_logging("테이블 {} 작업완료 데이터 삭제".format("purch_ord_cc_baemin_df"))
del purch_ord_cc_baemin_df

#####################################################################################################
#####################################################################################################


# 지역별(rgn1_cd,. rgn2_Cd) 까지만 그룹핑(그하위 단위는 너무 세분위)
# 구매수량, 금액, 쿠폰사용 (min max count, std, mean)
# 1. rgn1
# 1) 주문완료
line_logging("지역1 주문완료 그룹 파생 변수 생성")
rgn1_ord_ok_df = pd.pivot_table(ord_ok_df,index=["shop_no","ord_dt"],
                        values = ["item_quantity","ord_price","cpn_use_cnt"],
                        columns = "rgn1_cd_label",
                        aggfunc = ["count","sum","min","max","std","mean"])

line_logging("테이블 {}의 NA 값 처리".format("rgn1_ord_ok_df"))
rgn1_ord_ok_df = rgn1_ord_ok_df.fillna(0)
line_logging("테이블 {} 데이터 저장".format("rgn1_ord_ok_df"))
rgn1_ord_ok_df.to_csv(SAVE_PATH + "{}.csv".format("rgn1_ord_ok_df"), index=True, encoding="utf-8")
line_logging("테이블 {} 작업완료 데이터 삭제".format("rgn1_ord_ok_df"))
del rgn1_ord_ok_df

# 1-1) 배달주문완료
line_logging("지역1 배달주문완료 그룹 파생 변수 생성")
rgn1_ord_ok_del_df = pd.pivot_table(ord_ok_del_df,index=["shop_no","ord_dt"],
                        values = ["item_quantity","ord_price","cpn_use_cnt"],
                        columns = "rgn1_cd_label",
                        aggfunc = ["count","sum","min","max","std","mean"])

line_logging("테이블 {}의 NA 값 처리".format("rgn1_ord_ok_del_df"))
rgn1_ord_ok_del_df = rgn1_ord_ok_del_df.fillna(0)
line_logging("테이블 {} 데이터 저장".format("rgn1_ord_ok_del_df"))
rgn1_ord_ok_del_df.to_csv(SAVE_PATH + "{}.csv".format("rgn1_ord_ok_del_df"), index=True, encoding="utf-8")
line_logging("테이블 {} 작업완료 데이터 삭제".format("rgn1_ord_ok_del_df"))
del rgn1_ord_ok_del_df


# 1-2) 배민오더주문완료
line_logging("지역1 배민오더 주문완료 그룹 파생 변수 생성")
rgn1_ord_ok_baemin_df = pd.pivot_table(ord_ok_baemin_df,index=["shop_no","ord_dt"],
                        values = ["item_quantity","ord_price","cpn_use_cnt"],
                        columns = "rgn1_cd_label",
                        aggfunc = ["count","sum","min","max","std","mean"])

line_logging("테이블 {}의 NA 값 처리".format("rgn1_ord_ok_baemin_df"))
rgn1_ord_ok_baemin_df = rgn1_ord_ok_baemin_df.fillna(0)
line_logging("테이블 {} 데이터 저장".format("rgn1_ord_ok_baemin_df"))
rgn1_ord_ok_baemin_df.to_csv(SAVE_PATH + "{}.csv".format("rgn1_ord_ok_baemin_df"), index=True, encoding="utf-8")
line_logging("테이블 {} 작업완료 데이터 삭제".format("rgn1_ord_ok_baemin_df"))
del rgn1_ord_ok_baemin_df


# 2) 주문취소
line_logging("지역1 주문취소 그룹 파생 변수 생성")
rgn1_ord_cc_df  = pd.pivot_table(ord_cc_df,index=["shop_no","ord_dt"],
                        values = ["item_quantity","ord_price","cpn_use_cnt"],
                        columns = "rgn1_cd_label",
                        aggfunc = ["count","sum","min","max","std","mean"])

line_logging("테이블 {}의 NA 값 처리".format("rgn1_ord_cc_df"))
rgn1_ord_cc_df = rgn1_ord_cc_df.fillna(0)
line_logging("테이블 {} 데이터 저장".format("rgn1_ord_cc_df"))
rgn1_ord_cc_df.to_csv(SAVE_PATH + "{}.csv".format("rgn1_ord_cc_df"), index=True, encoding="utf-8")
line_logging("테이블 {} 작업완료 데이터 삭제".format("rgn1_ord_cc_df"))
del rgn1_ord_cc_df

# 2-1) 배달주문취소
line_logging("지역1 배달 주문취소 그룹 파생 변수 생성")
rgn1_ord_cc_del_df = pd.pivot_table(ord_cc_del_df,index=["shop_no","ord_dt"],
                        values = ["item_quantity","ord_price","cpn_use_cnt"],
                        columns = "rgn1_cd_label",
                        aggfunc = ["count","sum","min","max","std","mean"])

line_logging("테이블 {}의 NA 값 처리".format("rgn1_ord_cc_del_df"))
rgn1_ord_cc_del_df = rgn1_ord_cc_del_df.fillna(0)
line_logging("테이블 {} 데이터 저장".format("rgn1_ord_cc_del_df"))
rgn1_ord_cc_del_df.to_csv(SAVE_PATH + "{}.csv".format("rgn1_ord_cc_del_df"), index=True, encoding="utf-8")
line_logging("테이블 {} 작업완료 데이터 삭제".format("rgn1_ord_cc_del_df"))
del rgn1_ord_cc_del_df

# 2-2) 배민오더주문취소
line_logging("지역1 배민오더 주문취소 그룹 파생 변수 생성")
rgn1_ord_cc_baemin_df = pd.pivot_table(ord_cc_baemin_df,index=["shop_no","ord_dt"],
                        values = ["item_quantity","ord_price","cpn_use_cnt"],
                        columns = "rgn1_cd_label",
                        aggfunc = ["count","sum","min","max","std","mean"])

line_logging("테이블 {}의 NA 값 처리".format("rgn1_ord_cc_baemin_df"))
rgn1_ord_cc_baemin_df = rgn1_ord_cc_baemin_df.fillna(0)
line_logging("테이블 {} 데이터 저장".format("rgn1_ord_cc_baemin_df"))
rgn1_ord_cc_baemin_df.to_csv(SAVE_PATH + "{}.csv".format("rgn1_ord_cc_baemin_df"), index=True, encoding="utf-8")
line_logging("테이블 {} 작업완료 데이터 삭제".format("rgn1_ord_cc_baemin_df"))
del rgn1_ord_cc_baemin_df

#####################################################################################################
#####################################################################################################
# 2. rgn2
# 1) 주문완료
line_logging("지역2 주문완료 그룹 파생 변수 생성")
rgn2_ord_ok_df = pd.pivot_table(ord_ok_df,index=["shop_no","ord_dt"],
                        values = ["item_quantity","ord_price","cpn_use_cnt"],
                        columns = "rgn2_cd_label",
                        aggfunc = ["count","sum","min","max","std","mean"])

line_logging("테이블 {}의 NA 값 처리".format("rgn2_ord_ok_df"))
rgn2_ord_ok_df = rgn2_ord_ok_df.fillna(0)
line_logging("테이블 {} 데이터 저장".format("rgn2_ord_ok_df"))
rgn2_ord_ok_df.to_csv(SAVE_PATH + "{}.csv".format("rgn2_ord_ok_df"), index=True, encoding="utf-8")
line_logging("테이블 {} 작업완료 데이터 삭제".format("rgn2_ord_ok_df"))
del rgn2_ord_ok_df

# 1-1) 배달주문완료
line_logging("지역2 배달 주문완료 그룹 파생 변수 생성")
rgn2_ord_ok_del_df = pd.pivot_table(ord_ok_del_df,index=["shop_no","ord_dt"],
                        values = ["item_quantity","ord_price","cpn_use_cnt"],
                        columns = "rgn2_cd_label",
                        aggfunc = ["count","sum","min","max","std","mean"])

line_logging("테이블 {}의 NA 값 처리".format("rgn2_ord_ok_del_df"))
rgn2_ord_ok_del_df = rgn2_ord_ok_del_df.fillna(0)
line_logging("테이블 {} 데이터 저장".format("rgn2_ord_ok_del_df"))
rgn2_ord_ok_del_df.to_csv(SAVE_PATH + "{}.csv".format("rgn2_ord_ok_del_df"), index=True, encoding="utf-8")
line_logging("테이블 {} 작업완료 데이터 삭제".format("rgn2_ord_ok_del_df"))
del rgn2_ord_ok_del_df

# 1-2) 배민오더주문완료
line_logging("지역2 배민오더 주문완료 그룹 파생 변수 생성")
rgn2_ord_ok_baemin_df = pd.pivot_table(ord_ok_baemin_df,index=["shop_no","ord_dt"],
                        values = ["item_quantity","ord_price","cpn_use_cnt"],
                        columns = "rgn2_cd_label",
                        aggfunc = ["count","sum","min","max","std","mean"])

line_logging("테이블 {}의 NA 값 처리".format("rgn2_ord_ok_baemin_df"))
rgn2_ord_ok_baemin_df = rgn2_ord_ok_baemin_df.fillna(0)
line_logging("테이블 {} 데이터 저장".format("rgn2_ord_ok_baemin_df"))
rgn2_ord_ok_baemin_df.to_csv(SAVE_PATH + "{}.csv".format("rgn2_ord_ok_baemin_df"), index=True, encoding="utf-8")
line_logging("테이블 {} 작업완료 데이터 삭제".format("rgn2_ord_ok_baemin_df"))
del rgn2_ord_ok_baemin_df


# 2) 주문취소
line_logging("지역2 주문취소 그룹 파생 변수 생성")
rgn2_ord_cc_df  = pd.pivot_table(ord_cc_df,index=["shop_no","ord_dt"],
                        values = ["item_quantity","ord_price","cpn_use_cnt"],
                        columns = "rgn2_cd_label",
                        aggfunc = ["count","sum","min","max","std","mean"])

line_logging("테이블 {}의 NA 값 처리".format("rgn2_ord_cc_df"))
rgn2_ord_cc_df = rgn2_ord_cc_df.fillna(0)
line_logging("테이블 {} 데이터 저장".format("rgn2_ord_cc_df"))
rgn2_ord_cc_df.to_csv(SAVE_PATH + "{}.csv".format("rgn2_ord_cc_df"), index=True, encoding="utf-8")
line_logging("테이블 {} 작업완료 데이터 삭제".format("rgn2_ord_cc_df"))
del rgn2_ord_cc_df

# 2-1) 배달주문취소
line_logging("지역2 배달주문취소 그룹 파생 변수 생성")
rgn2_ord_cc_del_df = pd.pivot_table(ord_cc_del_df,index=["shop_no","ord_dt"],
                        values = ["item_quantity","ord_price","cpn_use_cnt"],
                        columns = "rgn2_cd_label",
                        aggfunc = ["count","sum","min","max","std","mean"])

line_logging("테이블 {}의 NA 값 처리".format("rgn2_ord_cc_del_df"))
rgn2_ord_cc_del_df = rgn2_ord_cc_del_df.fillna(0)
line_logging("테이블 {} 데이터 저장".format("rgn2_ord_cc_del_df"))
rgn2_ord_cc_del_df.to_csv(SAVE_PATH + "{}.csv".format("rgn2_ord_cc_del_df"), index=True, encoding="utf-8")
line_logging("테이블 {} 작업완료 데이터 삭제".format("rgn2_ord_cc_del_df"))
del rgn2_ord_cc_del_df


# 2-2) 배민오더주문취소
line_logging("지역2 배민오더주문취소 그룹 파생 변수 생성")
rgn2_ord_cc_baemin_df = pd.pivot_table(ord_cc_baemin_df,index=["shop_no","ord_dt"],
                        values = ["item_quantity","ord_price","cpn_use_cnt"],
                        columns = "rgn2_cd_label",
                        aggfunc = ["count","sum","min","max","std","mean"])

line_logging("테이블 {}의 NA 값 처리".format("rgn2_ord_cc_baemin_df"))
rgn2_ord_cc_baemin_df = rgn2_ord_cc_baemin_df.fillna(0)
line_logging("테이블 {} 데이터 저장".format("rgn2_ord_cc_baemin_df"))
rgn2_ord_cc_baemin_df.to_csv(SAVE_PATH + "{}.csv".format("rgn2_ord_cc_baemin_df"), index=True, encoding="utf-8")
line_logging("테이블 {} 작업완료 데이터 삭제".format("rgn2_ord_cc_baemin_df"))
del rgn2_ord_cc_baemin_df

#####################################################################################################
#####################################################################################################
'''
# 3. rgn3
# 1) 주문완료
line_logging("지역3 주문완료 그룹 파생 변수 생성")
rgn3_ord_ok_df = pd.pivot_table(ord_ok_df,index=["shop_no","ord_dt"],
                        values = ["item_quantity","ord_price","cpn_use_cnt"],
                        columns = "rgn3_cd_label",
                        aggfunc = ["count","sum","min","max","std","mean"])

line_logging("테이블 {}의 NA 값 처리".format("rgn3_ord_ok_df"))
rgn3_ord_ok_df = rgn3_ord_ok_df.fillna(0)
line_logging("테이블 {} 데이터 저장".format("rgn3_ord_ok_df"))
rgn3_ord_ok_df.to_csv(SAVE_PATH + "{}.csv".format("rgn3_ord_ok_df"), index=True, encoding="utf-8")
line_logging("테이블 {} 작업완료 데이터 삭제".format("rgn3_ord_ok_df"))
del rgn3_ord_ok_df

# 1-1) 배달주문완료
line_logging("지역3 배달 주문완료 그룹 파생 변수 생성")
rgn3_ord_ok_del_df = pd.pivot_table(ord_ok_del_df,index=["shop_no","ord_dt"],
                        values = ["item_quantity","ord_price","cpn_use_cnt"],
                        columns = "rgn3_cd_label",
                        aggfunc = ["count","sum","min","max","std","mean"])

line_logging("테이블 {}의 NA 값 처리".format("rgn3_ord_ok_del_df"))
rgn3_ord_ok_del_df = rgn3_ord_ok_del_df.fillna(0)
line_logging("테이블 {} 데이터 저장".format("rgn3_ord_ok_del_df"))
rgn3_ord_ok_del_df.to_csv(SAVE_PATH + "{}.csv".format("rgn3_ord_ok_del_df"), index=True, encoding="utf-8")
line_logging("테이블 {} 작업완료 데이터 삭제".format("rgn3_ord_ok_del_df"))
del rgn3_ord_ok_del_df

# 1-2) 배민오더주문완료
line_logging("지역3 배민오더 주문완료 그룹 파생 변수 생성")
rgn3_ord_ok_baemin_df = pd.pivot_table(ord_ok_baemin_df,index=["shop_no","ord_dt"],
                      
                        values = ["item_quantity","ord_price","cpn_use_cnt"],
                        columns = "rgn3_cd_label",
                        aggfunc = ["count","sum","min","max","std","mean"])

line_logging("테이블 {}의 NA 값 처리".format("rgn3_ord_ok_baemin_df"))
rgn3_ord_ok_baemin_df = rgn3_ord_ok_baemin_df.fillna(0)
line_logging("테이블 {} 데이터 저장".format("rgn3_ord_ok_baemin_df"))
rgn3_ord_ok_baemin_df.to_csv(SAVE_PATH + "{}.csv".format("rgn3_ord_ok_baemin_df"), index=True, encoding="utf-8")
line_logging("테이블 {} 작업완료 데이터 삭제".format("rgn3_ord_ok_baemin_df"))
del rgn3_ord_ok_baemin_df

# 2) 주문취소
line_logging("지역3 주문취소 그룹 파생 변수 생성")
rgn3_ord_cc_df  = pd.pivot_table(ord_cc_df,index=["shop_no","ord_dt"],
                        values = ["item_quantity","ord_price","cpn_use_cnt"],
                        columns = "rgn3_cd_label",
                        aggfunc = ["count","sum","min","max","std","mean"])

line_logging("테이블 {}의 NA 값 처리".format("rgn3_ord_cc_df"))
rgn3_ord_cc_df = rgn3_ord_cc_df.fillna(0)
line_logging("테이블 {} 데이터 저장".format("rgn3_ord_cc_df"))
rgn3_ord_cc_df.to_csv(SAVE_PATH + "{}.csv".format("rgn3_ord_cc_df"), index=True, encoding="utf-8")
line_logging("테이블 {} 작업완료 데이터 삭제".format("rgn3_ord_cc_df"))
del rgn3_ord_cc_df

# 2-1) 배달주문취소
line_logging("지역3 배달 주문취소 그룹 파생 변수 생성")
rgn3_ord_cc_del_df = pd.pivot_table(ord_cc_del_df,index=["shop_no","ord_dt"],
                        values = ["item_quantity","ord_price","cpn_use_cnt"],
                        columns = "rgn3_cd_label",
                        aggfunc = ["count","sum","min","max","std","mean"])

line_logging("테이블 {}의 NA 값 처리".format("rgn3_ord_cc_del_df"))
rgn3_ord_cc_del_df = rgn3_ord_cc_del_df.fillna(0)
line_logging("테이블 {} 데이터 저장".format("rgn3_ord_cc_del_df"))
rgn3_ord_cc_del_df.to_csv(SAVE_PATH + "{}.csv".format("rgn3_ord_cc_del_df"), index=True, encoding="utf-8")
line_logging("테이블 {} 작업완료 데이터 삭제".format("rgn3_ord_cc_del_df"))
del rgn3_ord_cc_del_df

# 2-2) 배민오더주문취소
line_logging("지역3 배민오더 주문취소 그룹 파생 변수 생성")
rgn3_ord_cc_baemin_df = pd.pivot_table(ord_cc_baemin_df,index=["shop_no","ord_dt"],
                        values = ["item_quantity","ord_price","cpn_use_cnt"],
                        columns = "rgn3_cd_label",
                        aggfunc = ["count","sum","min","max","std","mean"])

line_logging("테이블 {}의 NA 값 처리".format("rgn3_ord_cc_baemin_df"))
rgn3_ord_cc_baemin_df = rgn3_ord_cc_baemin_df.fillna(0)
line_logging("테이블 {} 데이터 저장".format("rgn3_ord_cc_baemin_df"))
rgn3_ord_cc_baemin_df.to_csv(SAVE_PATH + "{}.csv".format("rgn3_ord_cc_baemin_df"), index=True, encoding="utf-8")
line_logging("테이블 {} 작업완료 데이터 삭제".format("rgn3_ord_cc_baemin_df"))
del rgn3_ord_cc_baemin_df

##### -------- local memory error -------- ###########
'''

#####################################################################################################
#####################################################################################################
'''
line_logging("파이썬 전역 변수 객체 선언")
global_list = globals().keys()
line_logging("선언한 데이터프레임만 추출")
global_list = [word for word in global_list if "df" in word]
global_list.remove("df")
# 각 데이터프레임 값 처리
for i in global_list:

    line_logging("테이블 {}의 NA 값 처리".format(i))
    globals()[i] = eval(i).fillna(0)
    line_logging("테이블 {} 데이터 저장".format(i))
    globals()[i].to_csv("e:/test/{}.csv".format(i),index=True, encoding="utf-8")
    line_logging("테이블 {} 작업완료 데이터 삭제".format(i))
'''