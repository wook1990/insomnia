import pandas as pd
import sys
import os
import warnings
import glob

warnings.filterwarnings(action="ignore")


folder_list=['19년 12월', '20년 1월', '20년 2월', '20년 3월', '20년 4월', '20년 5월', '20년 6월', '20년 7월', '20년 8월', '20년 9월','20년 10월']

print(folder_list)
df_total =  pd.DataFrame()

for i in folder_list:
    print(i, flush=True)
    # raw_data merge
    #file_list = glob.glob("/home/data/raw/{0}/*.csv".format(i))
    # group_data merge
    file_list = glob.glob("/home/data/prep/{0}/*.csv".format(i))
    for j in range(0, len(file_list)):
        if j == 0:
            print(file_list[j], flush=True)
            df = pd.read_csv(file_list[j], index_col=False)
            df_merge = df
        else:
            print(file_list[j], flush=True)
            df2 = pd.read_csv(file_list[j], index_col =False)
            df_merge = pd.merge(df_merge, df2, left_on = "min_time", right_on = "min_time", how = "outer")
    df_merge.to_csv("/home/data/prep/total/"+ i + "_merged.csv", index=False)
    df_total = pd.concat([df_total, df_merge])

print(len(df_total))
df_total.to_csv("/home/data/prep/total/total_data.csv", index=False)

print("DONE")
'''
print("그룹별 평균으로 결측치 대체")
fill_na_gr = lambda g:g.fillna(g.mean())
print("그룹핑 변수 생성")
df_total["min_gr"] = df["min_time"].str.replace(":\d+","",regex=True)
print("결측치 처리")
df_total = df_total.groupby("min_gr").apply(fill_na_gr)
df_total.reset_index(drop=True, inplace=True)
df_total.drop("min_gr", axis=1,inplace=True)
print("최종 병합 파일 로드")
print(len(df_total))
df_total.to_csv("/home/data/공기식기화기/total/total_data_missing_treat.csv", index=False)
'''
