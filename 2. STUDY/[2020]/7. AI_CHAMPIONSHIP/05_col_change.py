import pandas as pd
import glob

#DATA_PATH = "/workspace/data/"
# SAVE_PATH = "/workspace/data/gr_dev_var/"
SAVE_PATH = "e:/test/"

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



file_list = glob.glob(SAVE_PATH + "*.csv")
except_list = ["ord_cc_baemin_review_df", "ord_cc_del_review_df","ord_cc_reivew_df","ord_ok_baemin_review_df","ord_ok_del_review_df","ord_ok_review_df"]

for i in range(0,len(file_list)):
    cv_colname = []
    try:

        df = pd.read_csv(file_list[i], header=[0, 1, 2], index_col=[0, 1])
        line_logging("피벗팅")
        column_list = df.columns
        df_name = file_list[i].split("\\")[1].replace(".csv", "")
        line_logging(df_name)
        if df_name in except_list:
            pass
        else:
            for j in range(0, len(column_list)):
                cv_colname.append(df_name + "_" + column_list[j][1] + "_label_" + column_list[j][2] + "_" + column_list[j][0])
            df.columns = cv_colname
            df.to_csv(file_list[i], index=True, encoding="utf-8")
    except:
        df = pd.read_csv(file_list[i], header=[0, 1], index_col=[0, 1])
        line_logging("그룹핑")
        column_list = df.columns
        df_name = file_list[i].split("\\")[1].replace(".csv", "")
        line_logging(df_name)
        if df_name in except_list:
            pass
        else:
            for j in range(0, len(column_list)):
                cv_colname.append(df_name + "_" + column_list[j][0] + "_" + column_list[j][1])
            df.columns = cv_colname
            df.to_csv(file_list[i],index=True, encoding="utf-8")


#####################################################


file_list = glob.glob(SAVE_PATH + "*.csv")
tot_df = pd.DataFrame()
pop_list = []
for idx , string in enumerate(file_list):
    if "rgn2_" in string:
        pop_list.append(idx)

del file_list[pop_list[0]:pop_list[5]+1]

for i in range(0,len(file_list)):
    if i == 0:
        tot_df = pd.read_csv(file_list[i], low_memory = False)
    else:
        line_logging(file_list[i])
        df = pd.read_csv(file_list[i], low_memory = False)
        tot_df = pd.merge(tot_df, df, on = ["shop_no","ord_dt"], how="left")
        line_logging("병합완료")
tot_df.to_csv("e:/train_data_fn.csv",index=False, encoding="utf-8")


tot_df_col_list = tot_df.columns
tot_df_col_list
cnt_col_list = []
for idx, _str in enumerate(tot_df_col_list):
    if "count" in _str:
        cnt_col_list.append(idx)

len(cnt_col_list)
len(tot_df)
tot_df.ord_dt.min()


