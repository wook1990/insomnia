import pandas as pd
import time
import warnings
import glob
import datetime
import operator

warnings.filterwarnings(action="ignore")

def conv_time(x):
     return datetime.datetime.strptime(x, "%Y-%m-%d %H")


def _find_dis_temp(x):
    start_time = x.head(1)["min_time"]
    start_time = datetime.strptime(start_time, "%Y-%m-%d %H:%M")
    delta_t = start_time + datetime.timedelta(hours=4)
    find_t = delta_t.strftime("%Y=%m-%d %H:%M")
    dis_temp = df.loc[df["min_time"] == find_t]


def _delta_time(x):
    x_tm = datetime.datetime.strptime(x, "%Y-%m-%d %H")
    delta_tm = x_tm - datetime.timedelta(hours=24)
    x_tm_str = delta_tm.strftime("%Y-%m-%d %H")
    return x_tm_str

def conv_time_to_str(x):
    return x.strftime("%Y-%m-%d %H")

def _work_group(F_PATH):

    # 작업그룹을 지정할 번호 선언
    work_group_num = 1
    # 전체 작업시간을 체크할 변수 선언
    cnt = 1

    # 데이터 로드
    df = pd.read_csv(F_PATH, index_col=False)
    # work group 컬럼 초기화 선언
    df["work_group"] = 0
    columns = list(df.columns)
    colname = [x for x in columns if operator.contains(x,"인입밸브")]
    print(colname)
    # work group 부여 (인입밸브 100이상인 작업에 대하여)
    for i in range(0, len(df)):
        if (df[colname[0]].iloc[i]) >= 100:
            df["work_group"].iloc[i] = work_group_num
            cnt+=1
            print("작업시간 : {0} , 인입밸브 : {1} , 작업그룹부여 : {2}".format(df["min_time"].iloc[i],df[colname[0]].iloc[i],df["work_group"].iloc[i]))
            try:
                if df[colname[0]].iloc[i+1] < 100:
                    work_group_num += 1
                    cnt = 1
            except:
                pass;
        else:
            df["work_group"].iloc[i] = 0
    file_name = F_PATH.split("/home/data/prep/final/")[1].replace(".csv","")
    df = df.iloc[df["work_group"] != 0 ]
    df.to_csv(PATH + file_name + "_work_group.csv", index=False)

if __name__ == "__main__":

    PATH = "/home/data/prep/final/"
    file_list = glob.glob(PATH + "unit_*.csv")

    for i in file_list:
        print(i)
        _work_group(i)

