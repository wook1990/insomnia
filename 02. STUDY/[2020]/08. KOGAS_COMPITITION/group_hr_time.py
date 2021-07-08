import pandas as pd
import glob
import warnings
import re
import sys
import os


warnings.filterwarnings(action="ignore")

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

def _group_by_min(PATH):

    name = PATH.split("/")[-1].replace(".csv","")
    line_logging(name)
    df = pd.read_csv(PATH, index_col=False)

    if name in ["대기온도","습도"]:
        df["min_time"] = df.TIMESTAMP.replace("(:\d+$)", "", regex=True)
        df_gr_min = df[["DATA","min_time"]].groupby("min_time").mean()
        df_gr_min.reset_index(inplace=True)
        df_gr_min.columns = ["min_time", name]
        if os.path.isdir(SAVE_PATH + duration + "/"):
            df_gr_min.to_csv(SAVE_PATH + duration+ "/"  + "분당"+ name + ".csv", index=False)
        else:
            os.mkdir(SAVE_PATH + duration + "/")
            df_gr_min.to_csv(SAVE_PATH + duration + "/" + "분당"+ name + ".csv", index=False)

    else:
        eq_name = name.split("_")[0]
        file_name = name.split("_")[1]
        df["min_time"] = df.TIMESTAMP.replace("(:\d+$)", "", regex=True)

        if file_name == "팬회전수":
            df_gr_min = df[["DATA","min_time"]].groupby("min_time").max()
        else:
            df_gr_min = df[["DATA","min_time"]].groupby("min_time").mean()

        df_gr_min.reset_index(inplace=True)
        df_gr_min.columns = ["min_time", eq_name+"_"+file_name]

        if os.path.isdir(SAVE_PATH + duration + "/"):
            df_gr_min.to_csv(SAVE_PATH + duration+ "/"  + eq_name +"_분당"+ file_name + ".csv", index=False)
        else:
            os.mkdir(SAVE_PATH + duration + "/")
            df_gr_min.to_csv(SAVE_PATH + duration + "/" + eq_name +"_분당"+ file_name + ".csv", index=False)

if __name__ == "__main__":

    duration = sys.argv[1]
    PATH = "/home/data/raw/{0}/".format(duration)
    SAVE_PATH = "/home/data/prep/"
    file_list = glob.glob(PATH + "*.csv")

    for i in file_list:
        line_logging(i)
        _group_by_min(i)
