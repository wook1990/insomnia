import pandas as pd
import time
import warnings
import glob
import datetime
import operator
import math


warnings.filterwarnings(action="ignore")

def conv_time(x):
     return datetime.datetime.strptime(x, "%Y-%m-%d %H:%M")

def conv_time_to_str(x):
    return x.strftime("%Y-%m-%d %H:%M")

def _cal_min(x):
    return  math.ceil(x.total_seconds()/60)

def _total_train_set(df):


    df = df.loc[df["work_group"] != 0]
    columns = list(df.columns)
    colname = [x for x in columns if operator.contains(x,"토출온도")]

    head = df.groupby("work_group").head(1)
    tail = df.groupby("work_group").tail(1)
    start_time = head["min_time"]
    start_time = start_time.apply(conv_time)
    delta_f = lambda x : x + datetime.timedelta(hours=4)
    delta_t = start_time.apply(delta_f)
    find_t = delta_t.apply(conv_time_to_str)
    # 네시간 이후의 값이 존재하는 그룹만 df 선언
    dis_temp_df = df[df["min_time"].isin(find_t.tolist())][[colname[0],"work_group"]]
    dis_temp_df = dis_temp_df.set_index("work_group")
    dis_temp_df.columns = ["dis_temp_4hour_left"]


    # 평균, 최대, 최소
    df_mean = df.groupby("work_group").mean().add_suffix("_mean")
    #df_max = df.groupby("work_group").max().add_suffix("_max")
    #df_min = df.groupby("work_group").min().add_suffix("_min")

    # 작업시간 구하기
    head = head.set_index("work_group")
    tail = tail.set_index("work_group")
    head.columns = ["start_" + x for x in head.columns.tolist()]
    tail.columns = ["end_" + x for x in tail.columns.tolist()]
    work_gr_df = pd.merge(head, tail, left_index=True, right_index=True, how="left")
    work_gr_df["working_time"] = work_gr_df["end_min_time"].apply(conv_time)- work_gr_df["start_min_time"].apply(conv_time)
    work_gr_df["working_time"] = work_gr_df["working_time"].apply(_cal_min)


    # 전체 데이터셋 합치기
    total_df = pd.merge(work_gr_df, dis_temp_df, right_index=True, left_index=True, how="right")
    total_df = pd.merge(total_df, df_mean, right_index=True, left_index=True, how="left")
    #total_df = pd.merge(total_df, df_max, right_index=True, left_index=True, how="left")
    #total_df = pd.merge(total_df, df_mean, right_index=True, left_index=True, how="left")
    #total_df = pd.merge(total_df, dis_temp_df, right_index=True, left_index=True, how="right")
    target_f = lambda x : 1 if x > 0 else 0
    total_df["target"] = total_df.dis_temp_4hour_left.apply(target_f)

    #print(total_df.columns)
    #print(total_df.info())
    total_df.columns = ['start_min_time', 'start_temp', 'start_hum', 'start_cell_A_rpm', 'start_cell_B_rpm', 'start_unit_dist_temp', 'start_unit_lng_temp', 'start_unit_inlet_valve', 'start_group_vapor_flow', 'end_min_time', 'end_temp', 'end_hum', 'end_cell_A_rpm', 'end_cell_B_rpm', 'end_unit_dist_temp', 'end_unit_lng_temp', 'end_unit_inlet_valve', 'end_group_vapor_flow','working_time', 'dis_temp_4hour_left','mean_temp', 'mean_hum', 'mean_cell_A_rpm', 'mean_cell_B_rpm', 'mean_unit_dist_temp', 'mean_unit_lng_temp', 'mean_unit_inlet_valve', 'mean_group_vapor_flow','target']

    total_df["working_hour_gr"] = total_df["start_min_time"].apply(lambda x: x.split(" ")[1]).str.replace(":\d\d","",regex=True)
    total_df["fan_rpm"] = (total_df["start_cell_A_rpm"] + total_df["start_cell_B_rpm"])/2
    for i in range(0,len(total_df)):
        if total_df["fan_rpm"].iloc[i]%50 < 25:
            total_df["fan_rpm"].iloc[i] = (total_df["fan_rpm"].iloc[i]//50) * 50
        else:
            total_df["fan_rpm"].iloc[i] = (total_df["fan_rpm"].iloc[i]//50 + 1) *50
    total_df = total_df[["fan_rpm", "working_hour_gr", "working_time","start_temp", "start_hum","mean_group_vapor_flow","mean_unit_lng_temp", "target"]]
    total_df.columns = ["fan_rpm", "working_hour_gr", "working_time","start_temp", "start_hum","group_vapor_flow","unit_lng_temp", "target"]

    return total_df

if __name__ == "__main__":
    file_list = glob.glob("/home/data/prep/final/*_work_*.csv")
    total_df = pd.DataFrame()
    for i in file_list:

        print(i)
        df = pd.read_csv(i,index_col=False)
        final_df = _total_train_set(df)
        total_df = pd.concat([total_df, final_df])
    total_df = total_df.loc[total_df["fan_rpm"] != 0]
    total_df = total_df.round(2)
    total_df.reset_index(drop=True, inplace=True)
    total_df.to_csv("/home/data/prep/final/train_set.csv",index=False)