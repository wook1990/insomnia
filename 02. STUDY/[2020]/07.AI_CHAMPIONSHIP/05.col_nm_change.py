##################################################################################################################
import pandas as pd
import dask.dataframe as dd

PATH = "/workspace/"
SAVE_PATH = "/workspace/"

# 1) column에 count 가 들어가잇는 컬럼 추출
df = dd.read_csv(PATH + "model_train_data_fn.csv",index_col=False, low_memory=False)
#df = predict_df
####################
collist = df.columns

ck_collist = []
gr_collist = []
pv_collist = []
group_df_list = ["df_gr","ord_ok_df_gr","ord_ok_del_df_gr","ord_ok_baemin_df_gr","ord_cc_df_gr","ord_cc_del_df_gr",
                 "ord_cc_baemin_df_gr","ord_ok_review_df","ord_ok_del_review_df","ord_ok_baemin_review_df","ord_cc_reivew_df",
                 "ord_cc_del_review_df","ord_cc_baemin_review_df","ord_ok_rating_tm_df","ord_ok_del_rating_tm_df",
                 "ord_ok_baemin_rating_tm_df","ord_cc_rating_tm_df","ord_cc_del_rating_tm_df","ord_cc_baemin_rating_tm_df"]

for idx, i in enumerate(collist):
    if "count" in i:
        ck_collist.append(i)
# 1,212 의 count 변수

# 2) grpingr으로 만들어진 컬럼
for idx, i in enumerate(ck_collist):
    for j in group_df_list:
        if j in i:
            gr_collist.append(i)

gr_cnt_df = df[gr_collist].fillna(0)
rm_col_name = gr_cnt_df.columns
gr_cnt_df = gr_cnt_df.T
df_name = []
for i in rm_col_name:
    df_name.append(i.split("_count")[0])

gr_cnt_df["df_name"] = df_name
gr_cnt_df = gr_cnt_df.groupby("df_name").max()
gr_cnt_df = gr_cnt_df.T
gr_cnt_df.columns = [word+"_count" for word in gr_cnt_df.columns]

# 3) pivoting으로 만들어진 라벨이 있는 컬럼
# 1191개
import re


pv_collist = list(set(ck_collist) - set(gr_collist))
pv_cnt_df = df[pv_collist].fillna(0)
rm_col_name = pv_cnt_df.columns
pv_cnt_df = pv_cnt_df.T
df_name = []
for i in rm_col_name:
    #print(i.split("_df")[0]+"_"+ re.sub("[_a-z]","", i.split("_df")[1].split("_label_")[1]))
    df_name.append(i.split("_df")[0]+"_"+ re.sub("[_a-z]","", i.split("_df")[1].split("_label_")[1]))
pv_cnt_df["df_name"] = df_name
pv_cnt_df = pv_cnt_df.groupby("df_name").max()
pv_cnt_df = pv_cnt_df.T
pv_cnt_df.columns = [word+"_count" for word in pv_cnt_df.columns]

# 최종 데이터 프레임
df.drop(ck_collist, axis=1, inplace=True)
df= pd.concat([df,gr_cnt_df, pv_cnt_df],axis=1)
df.to_csv("e:/AIchamp/predict_data.csv")
