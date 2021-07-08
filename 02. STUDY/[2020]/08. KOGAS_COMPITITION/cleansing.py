import pandas as pd
import glob
import warnings
import os
import sys
import re
import datetime

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

def _convert_time(x):
    dt_x = datetime.datetime.strptime(x, "%Y-%m-%d %H:%M:%S")
    time_str = dt_x.strftime('%Y-%m-%d %H:%M:%S')
    return time_str
def _cleansing_data(PATH,duration, key):

    df = pd.read_csv(PATH)
    df["TIMESTAMP"] = df["TIMESTAMP"].str.lstrip()
    df["TIMESTAMP"] = df["TIMESTAMP"].apply(_convert_time)


# 팬 회 전 수
def _fan_speed(duration):


    if duration =="20년 5월":
        PATH = "/production/2차_ 200910/2. 공기식기화기_팬회전수/20년 5월/"
    elif duration == "20년 3월":
        PATH = "/production/생산파트_추가데이터_200925/1. 데이터 합치기/공기식기화기/3월/"
    elif duration == "20년 7월":
        PATH = "/production/생산파트_추가데이터_200925/1. 데이터 합치기/공기식기화기/7월/"
    elif duration == "20년 8월":
        PATH = "/production/7차_ 201109/공기식기화기_추가데이터/8월/"
    elif duration == "20년 9월":
        PATH = "/production/7차_ 201109/공기식기화기_추가데이터/9월/"
    elif duration == "20년 10월":
        PATH = "/production/7차_ 201109/공기식기화기_추가데이터/10월/"
    else:
        PATH = "/production/1. 생산/{0}/공기식기화기/".format(duration)

    file_list = glob.glob(PATH + "*_팬회전수.csv")
    for i in file_list:

        line_logging(i)
        pattern = "(V-\d+\D\D)"
        eq_name = re.findall(pattern,i)[0]
        line_logging(eq_name)
        df = pd.read_csv(i)
        line_logging("원본데이터건수 : ", len(df))
        check_duration = duration_dict[duration]
        df["TIMESTAMP"] = df["TIMESTAMP"].str.lstrip()
        df.columns = ["TIMESTAMP","DATA"]
        df = df[df["TIMESTAMP"].str.contains(check_duration)]
        line_logging("수정후데이터건수 : ", len(df))
        line_logging("시간수정")
        line_logging(df["TIMESTAMP"][1])
        df["TIMESTAMP"] = df["TIMESTAMP"].apply(_convert_time)
        line_logging(df["TIMESTAMP"][1])
        if os.path.isdir(SAVE_PATH + duration + "/"):
            df.to_csv(SAVE_PATH + duration + "/" + "{0}_팬회전수.csv".format(eq_name), index=False)
        else:
            os.mkdir(SAVE_PATH + duration +"/")
            df.to_csv(SAVE_PATH + duration + "/" + "{0}_팬회전수.csv".format(eq_name), index=False)

def _temp_hum(duration):

    if duration == "20년 8월":
        PATH = "/production/7차_ 201109/공기식기화기_추가데이터/8월/"
    elif duration == "20년 9월":
        PATH = "/production/7차_ 201109/공기식기화기_추가데이터/9월/"
    elif duration == "20년 10월":
        PATH = "/production/7차_ 201109/공기식기화기_추가데이터/10월/"
    else:
        PATH = "/production/1. 생산/{0}/공기식기화기/".format(duration)

    file_list_1 = glob.glob(PATH + "*대기온도.csv")
    file_list_2 = glob.glob(PATH + "*습도.csv")

    line_logging("-----대기온도-----")
    df = pd.read_csv(file_list_1[0])
    line_logging("원본데이터건수 : ", len(df))
    check_duration = duration_dict[duration]
    df["TIMESTAMP"] = df["TIMESTAMP"].str.lstrip()
    df.columns = ["TIMESTAMP","DATA"]
    df = df[df["TIMESTAMP"].str.contains(check_duration)]
    line_logging("수정후데이터건수 : ", len(df))
    line_logging("시간수정")
    line_logging(df["TIMESTAMP"][1])
    df["TIMESTAMP"] = df["TIMESTAMP"].apply(_convert_time)
    line_logging(df["TIMESTAMP"][1])

    if os.path.isdir(SAVE_PATH + duration + "/"):
        df.to_csv(SAVE_PATH + duration + "/" + "대기온도.csv", index=False)
    else:
        os.mkdir(SAVE_PATH + duration +"/")
        df.to_csv(SAVE_PATH + duration + "/" + "대기온도.csv", index=False)

    line_logging("------습도------")
    df = pd.read_csv(file_list_2[0])
    line_logging("원본데이터건수 : ", len(df))
    check_duration = duration_dict[duration]
    df["TIMESTAMP"] = df["TIMESTAMP"].str.lstrip()
    df.columns = ["TIMESTAMP","DATA"]
    df = df[df["TIMESTAMP"].str.contains(check_duration)]
    line_logging("수정후데이터건수 : ", len(df))
    line_logging("시간수정")
    line_logging(df["TIMESTAMP"][1])
    df["TIMESTAMP"] = df["TIMESTAMP"].apply(_convert_time)
    line_logging(df["TIMESTAMP"][1])
    if os.path.isdir(SAVE_PATH + duration + "/"):
        df.to_csv(SAVE_PATH + duration + "/" + "습도.csv", index=False)
    else:
        os.mkdir(SAVE_PATH + duration +"/")
        df.to_csv(SAVE_PATH + duration + "/" + "습도.csv", index=False)

def _lng_temp(duration):

    if duration == "20년 8월":
        PATH = "/production/7차_ 201109/공기식기화기_추가데이터/8월/"
    elif duration == "20년 9월":
        PATH = "/production/7차_ 201109/공기식기화기_추가데이터/9월/"
    elif duration == "20년 10월":
        PATH = "/production/7차_ 201109/공기식기화기_추가데이터/10월/"
    else:
        PATH = "/production/1. 생산/{0}/공기식기화기/".format(duration)

    file_list = glob.glob(PATH + "*_lng온도.csv")
    for i in file_list:

        line_logging(i)
        pattern = "(V-\d+\D\D\D\D)"
        eq_name = re.findall(pattern,i)[0]
        line_logging(eq_name)
        df = pd.read_csv(i)
        line_logging("원본데이터건수 : ", len(df))
        check_duration = duration_dict[duration]
        df["TIMESTAMP"] = df["TIMESTAMP"].str.lstrip()
        df.columns = ["TIMESTAMP","DATA"]
        df = df[df["TIMESTAMP"].str.contains(check_duration)]
        line_logging("수정후데이터건수 : ", len(df))
        line_logging("시간수정")
        line_logging(df["TIMESTAMP"][1])
        df["TIMESTAMP"] = df["TIMESTAMP"].apply(_convert_time)
        line_logging(df["TIMESTAMP"][1])
        if os.path.isdir(SAVE_PATH + duration + "/"):
            df.to_csv(SAVE_PATH + duration + "/" + "{0}_lng온도.csv".format(eq_name), index=False)
        else:
            os.mkdir(SAVE_PATH + duration +"/")
            df.to_csv(SAVE_PATH + duration + "/" + "{0}_lng온도.csv".format(eq_name), index=False)

def _inlet_valve(duration):

    if duration == "20년 8월":
        PATH = "/production/7차_ 201109/공기식기화기_추가데이터/8월/"
    elif duration == "20년 9월":
        PATH = "/production/7차_ 201109/공기식기화기_추가데이터/9월/"
    elif duration == "20년 10월":
        PATH = "/production/7차_ 201109/공기식기화기_추가데이터/10월/"
    else:
        PATH = "/production/생산파트_추가데이터_200925/3. 공기식기화기_인입밸브/{0}/".format(duration)

    file_list = glob.glob(PATH + "*_인입밸브.csv")
    for i in file_list:

        line_logging(i)
        pattern = "(V-\d+\D\D\D\D)"
        eq_name = re.findall(pattern,i)[0]
        line_logging(eq_name)
        df = pd.read_csv(i)
        line_logging("원본데이터건수 : ", len(df))
        check_duration = duration_dict[duration]
        df["TIMESTAMP"] = df["TIMESTAMP"].str.lstrip()
        df.columns = ["TIMESTAMP","DATA"]
        df = df[df["TIMESTAMP"].str.contains(check_duration)]
        line_logging("수정후데이터건수 : ", len(df))
        line_logging("시간수정")
        line_logging(df["TIMESTAMP"][1])
        df["TIMESTAMP"] = df["TIMESTAMP"].apply(_convert_time)
        line_logging(df["TIMESTAMP"][1])
        if os.path.isdir(SAVE_PATH + duration + "/"):
            df.to_csv(SAVE_PATH + duration + "/" + "{0}_인입밸브.csv".format(eq_name), index=False)
        else:
            os.mkdir(SAVE_PATH + duration +"/")
            df.to_csv(SAVE_PATH + duration + "/" + "{0}_인입밸브.csv".format(eq_name), index=False)

def _discharge_temp(duration):

    if duration == "20년 8월":
        PATH = "/production/7차_ 201109/공기식기화기_추가데이터/8월/"
    elif duration == "20년 9월":
        PATH = "/production/7차_ 201109/공기식기화기_추가데이터/9월/"
    elif duration == "20년 10월":
        PATH = "/production/7차_ 201109/공기식기화기_추가데이터/10월/"
    else:
        PATH = "/production/1. 생산/{0}/공기식기화기/".format(duration)

    file_list = glob.glob(PATH + "*_토출온도.csv")

    for i in file_list:

        pattern = "(V-\d+\D\D\D\D)"
        eq_name = re.findall(pattern, i)[0]
        line_logging(eq_name)
        try:
            if eq_name[5:9] == "AAAD":
                line_logging("error check_1")
                eq_name = eq_name[0:5] + "AAAB"
                line_logging("Modifiy eq_name : " + eq_name)
            elif eq_name[5:9] == "ABAC":
                line_logging("error_check_2")
                eq_name = eq_name[0:5]+ "ACAD"
                line_logging("Modifiy eq_name : " + eq_name)
            elif eq_name[5:9] == "AAAD":
                line_logging("error_check_3")
                eq_name = eq_name[0:5] + "AAAB"
                line_logging("Modifiy eq_name : " + eq_name)
            elif eq_name[5:9] == "AC_토":
                line_logging("error_check_4")
                eq_name = "V-202ACAD"
                line_logging("Modifiy eq_name : " + eq_name)
        except:
            pass

        df = pd.read_csv(i)
        line_logging("원본데이터건수 : ", len(df))
        check_duration = duration_dict[duration]
        df["TIMESTAMP"] = df["TIMESTAMP"].str.lstrip()
        df.columns = ["TIMESTAMP","DATA"]
        df = df[df["TIMESTAMP"].str.contains(check_duration)]
        line_logging("수정후데이터건수 : ", len(df))
        line_logging("시간수정")
        line_logging(df["TIMESTAMP"][1])
        df["TIMESTAMP"] = df["TIMESTAMP"].apply(_convert_time)
        line_logging(df["TIMESTAMP"][1])
        if os.path.isdir(SAVE_PATH + duration + "/"):
            df.to_csv(SAVE_PATH + duration + "/" + "{0}_토출온도.csv".format(eq_name), index=False)
        else:
            os.mkdir(SAVE_PATH + duration +"/")
            df.to_csv(SAVE_PATH + duration + "/" + "{0}_토출온도.csv".format(eq_name), index=False)

def _vapor_flow(duration):

    if duration == "20년 8월":
        PATH = "/production/7차_ 201109/공기식기화기_추가데이터/8월/"
    elif duration == "20년 9월":
        PATH = "/production/7차_ 201109/공기식기화기_추가데이터/9월/"
    elif duration == "20년 10월":
        PATH = "/production/7차_ 201109/공기식기화기_추가데이터/10월/"
    else:
        PATH = "/production/1. 생산/{0}/공기식기화기/".format(duration)

    file_list = glob.glob(PATH + "*_기화유량.csv")

    for i in file_list:

        pattern = "(V-\d+\D\D~\D\D)"
        eq_name = re.findall(pattern,i)[0]
        if eq_name[5:10] == "AB~AD":
            eq_name = eq_name[0:5]+"AA~AD"
        line_logging(eq_name)

        df = pd.read_csv(i)
        line_logging("원본데이터건수 : ", len(df))
        check_duration = duration_dict[duration]
        df["TIMESTAMP"] = df["TIMESTAMP"].str.lstrip()
        df.columns = ["TIMESTAMP","DATA"]
        df = df[df["TIMESTAMP"].str.contains(check_duration)]
        line_logging("수정후데이터건수 : ", len(df))
        line_logging("시간수정")
        line_logging(df["TIMESTAMP"][1])
        df["TIMESTAMP"] = df["TIMESTAMP"].apply(_convert_time)
        line_logging(df["TIMESTAMP"][1])
        if os.path.isdir(SAVE_PATH + duration + "/"):
            df.to_csv(SAVE_PATH + duration + "/" + "{0}_기화유량.csv".format(eq_name), index=False)
        else:
            os.mkdir(SAVE_PATH + duration +"/")
            df.to_csv(SAVE_PATH + duration + "/" + "{0}_기화유량.csv".format(eq_name), index=False)


if __name__ == "__main__":

    SAVE_PATH = "/home/data/raw/"
    duration = sys.argv[1]

    duration_dict = {"19년 12월":"2019-12", "20년 1월":"2020-01", "20년 2월":"2020-02", "20년 3월":"2020-03",
                     "20년 4월":"2020-04", "20년 5월":"2020-05", "20년 6월":"2020-06", "20년 7월":"2020-07",
                     "20년 8월":"2020-08", "20년 9월":"2020-09", "20년 10월":"2020-10"}
    line_logging("------------FAN SPEED--------------")
    _fan_speed(duration)
    line_logging("-------------TEMP HUM--------------")
    _temp_hum(duration)
    line_logging("------------INLET VALVE------------")
    _inlet_valve(duration)
    line_logging("-------------LNG TEMP--------------")
    _lng_temp(duration)
    line_logging("----------DISCHARGE TEMP-----------")
    _discharge_temp(duration)
    line_logging("------------VAPOR FLOW-------------")
    _vapor_flow(duration)
    line_logging("--------",duration," FINISHED-------")
