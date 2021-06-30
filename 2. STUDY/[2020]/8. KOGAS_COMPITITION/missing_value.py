import pandas as pd
import datetime
import math

def conv_time(x):
     return datetime.datetime.strptime(x, "%Y-%m-%d %H")

def _delta_time(x):
    x_tm = datetime.datetime.strptime(x, "%Y-%m-%d %H")
    delta_tm = x_tm - datetime.timedelta(hours=24)
    x_tm_str = delta_tm.strftime("%Y-%m-%d %H")
    return x_tm_str

def conv_time_to_str(x):
    return x.strftime("%Y-%m-%d %H")


if __name__ == "__main__":

    # 전처리 1 단계 파일 로드
    df = pd.read_csv("/home/data/prep/final/total_prep_1.csv",index_col=False)

    # null 값이 존재하는 컬럼 추출
    df_sub = df[["min_time","대기온도","습도"]]

    # 시간 그룹 변수 생성
    df_sub["hr"] = df["min_time"].str.replace(":\d+$", "", regex=True)

    # 시간 그룹 변수로 시간대별 평균값 데이터 생성
    df_sub_gr = df_sub[["hr","대기온도","습도"]].groupby("hr", dropna=False).mean()
    df_sub_gr.reset_index(inplace=True)
    df_sub_gr.columns = ["hr","평균대기온도","평균습도"]
    df_sub_gr = df_sub_gr.round(2)

    # 결측이 있는 시간대만 추출
    df_missing = df_sub_gr.loc[(df_sub_gr["평균대기온도"].isnull()) | (df_sub_gr["평균습도"].isnull())]

    # 결측이 있는 시간대 이전의 날짜 컬럼 생성
    df_missing["day_ago"] = df_missing["hr"].apply(_delta_time)

    # 이전날자와 join하여 결측치 보정 table생성
    merge_missing = pd.merge(df_missing, df_sub_gr, left_on="day_ago", right_on="hr", how="left")
    merge_missing = merge_missing[["hr_x", "평균대기온도_y","평균습도_y"]]
    merge_missing.columns  = ["hr","평균대기온도","평균습도"]
    merge_missing = merge_missing.fillna(method="bfill")

    # Null 이존재하는 전체 데이터 조인
    treate_df = pd.merge(df_sub, merge_missing, left_on="hr", right_on = "hr", how="left")
    print(treate_df)
    for i in range(0,len(df)):
        if math.isnan(treate_df["대기온도"].iloc[i]):
            bef = treate_df["대기온도"].iloc[i]
            treate_df["대기온도"].iloc[i] = treate_df["평균대기온도"].iloc[i]
            aft = treate_df["대기온도"].iloc[i]
            print("이전값 : {0} , 변환값 : {1} ".format(bef,aft))
        else:
            pass

        if math.isnan(treate_df["습도"].iloc[i]):
            bef = treate_df["습도"].iloc[i]
            treate_df["습도"].iloc[i] = treate_df["평균습도"].iloc[i]
            aft = treate_df["습도"].iloc[i]
            print("이전값 : {0} , 변환값 : {1} ".format(bef, aft))

        df["대기온도"] = treate_df["대기온도"]
        df["습도"] = treate_df["습도"]

        print(df.isnull().sum())
        df.to_csv("/home/data/prep/final/total_prep_2.csv", index=False)

