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
#DATA_PATH = "/workspace/data/"
#SAVE_PATH = "/workspace/data/gr_dev_var/"


DATA_PATH = "E:/"
SAVE_PATH = "E:/test/"

line_logging("Data Load")
df = pd.read_csv(DATA_PATH + "train_data_prep_3.csv", index_col=False, low_memory=False)
#df.loc[df.ord_review_delta_tm == 9999, "ord_review_delta_tm"] = 0
#df.to_csv(DATA_PATH + "train_data_prep_3.csv",  index=False, encoding="utf-8")
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



import glob

file_list = glob.glob(SAVE_PATH + "*.csv")
except_list = ["ord_cc_baemin_review_df", "ord_cc_del_review_df","ord_cc_reivew_df",
               "ord_ok_baemin_review_df","ord_ok_del_review_df","ord_ok_review_df"]
for i in range(0,len(file_list)):
    cv_colname = []
    try:

        df = pd.read_csv(file_list[i], header=[0, 1, 2], index_col=[0, 1])
        print("피벗팅")
        column_list = df.columns
        df_name = file_list[i].split("\\")[1].replace(".csv", "")
        print(df_name)
        if df_name in except_list:
            pass
        else:
            for j in range(0, len(column_list)):
                cv_colname.append(df_name + "_" + column_list[j][1] + "_label_" + column_list[j][2] + "_" + column_list[j][0])
            df.columns = cv_colname
            df.to_csv(file_list[i], index=True, encoding="utf-8")
    except:
        df = pd.read_csv(file_list[i], header=[0, 1], index_col=[0, 1])
        print("그룹핑")
        column_list = df.columns
        df_name = file_list[i].split("\\")[1].replace(".csv", "")
        print(df_name)
        if df_name in except_list:
            pass
        else:
            for j in range(0, len(column_list)):
                cv_colname.append(df_name + "_" + column_list[j][1] + "_" + column_list[j][0])
            df.columns = cv_colname
            df.to_csv(file_list[i],index=True, encoding="utf-8")


##########################################################