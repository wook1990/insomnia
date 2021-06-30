import pandas as pd
import numpy as np

SAVE_PATH = "e:/WAI/data/KBO/"


y = np.arange(2008,2021,1)
yearly_game_score = pd.DataFrame()

for year in y:
    if year in [2009, 2010, 2012, 2020]:
        month = ['04','05','06','07','08','09','10']
        for m in month:

            url =  "https://sports.news.naver.com/kbaseball/schedule/index.nhn?month={0}&year={1}".format(m, year)
            df= pd.read_html(url)
            t_df = pd.DataFrame()
            for i in range(1, len(df) - 1):
                t_df = pd.concat([t_df, df[i]])
            t_df['년도'] = year
            t_df.columns = ['날짜', '시간', '경기', '중계/기록', '중계방송사', '구장', '알림받기', "년도"]
            t_df = t_df[["년도", "날짜", "시간", "경기", "구장"]]
            t_df = t_df.reset_index().drop("index", axis=1)

            t_df = t_df[t_df["경기"] != "프로야구 경기가 없습니다."]
            yearly_game_score = pd.concat([yearly_game_score, t_df])
            yearly_game_score = yearly_game_score.reset_index().drop("index", axis=1)

            for i in range(0, len(yearly_game_score)):
                yearly_game_score.loc[i, "경기"] = str(yearly_game_score.loc[i, "경기"]).replace("1   ", "")
                yearly_game_score.loc[i, "경기"] = str(yearly_game_score.loc[i, "경기"]).replace("  ", " ")
                yearly_game_score.loc[i, "경기"] = str(yearly_game_score.loc[i, "경기"]).strip()
                print(str(yearly_game_score.loc[i, "경기"]))

                yearly_game_score.loc[i,"원정팀"] =  yearly_game_score.loc[i,"경기"].split(" ")[0]
                yearly_game_score.loc[i,"홈팀"] = yearly_game_score.loc[i, "경기"].split(" ")[2]

                if yearly_game_score.loc[i, "경기"].split(" ")[1] == "VS":
                    yearly_game_score.loc[i, "원정팀점수"] = "경기취소"
                    yearly_game_score.loc[i, "홈팀점수"] = "경기취소"
                else:
                    yearly_game_score.loc[i,"원정팀점수"] = yearly_game_score.loc[i, "경기"].split(" ")[1].split(":")[0]
                    yearly_game_score.loc[i, "홈팀점수"] = yearly_game_score.loc[i, "경기"].split(" ")[1].split(":")[1]

                if yearly_game_score.loc[i,"원정팀점수"] > yearly_game_score.loc[i, "홈팀점수"]:
                    yearly_game_score.loc[i, "승자"] = yearly_game_score.loc[i,"원정팀"]
                elif yearly_game_score.loc[i,"원정팀점수"] < yearly_game_score.loc[i, "홈팀점수"]:
                    yearly_game_score.loc[i, "승자"] = yearly_game_score.loc[i, "홈팀"]
                else:
                    yearly_game_score.loc[i, "승자"] = "무승부"

        yearly_game_score.drop("경기", axis=1)
        yearly_game_score.to_csv("e:/WAI/data/KBO/{0}_game_result(cp949).csv".format(year), index=False,
                                     encoding = "cp949")
    else:
        month = ['03','04','05','06','07','08','09','10']

        for m in month:
            url = "https://sports.news.naver.com/kbaseball/schedule/index.nhn?month={0}&year={1}".format(m, year)
            df = pd.read_html(url)
            t_df = pd.DataFrame()
            for i in range(1, len(df) - 1):
                t_df = pd.concat([t_df, df[i]])
            t_df['년도'] = year
            t_df.columns = ['날짜', '시간', '경기', '중계/기록', '중계방송사', '구장', '알림받기', "년도"]
            t_df = t_df[["년도", "날짜", "시간", "경기", "구장"]]
            t_df = t_df.reset_index().drop("index", axis=1)

            t_df = t_df[t_df["경기"] != "프로야구 경기가 없습니다."]
            yearly_game_score = pd.concat([yearly_game_score, t_df])
            yearly_game_score = yearly_game_score.reset_index().drop("index", axis=1)

        for i in range(0, len(yearly_game_score)):

            yearly_game_score.loc[i, "경기"] = str(yearly_game_score.loc[i, "경기"]).replace("1   ", "")
            yearly_game_score.loc[i, "경기"] = str(yearly_game_score.loc[i, "경기"]).replace("  ", " ")
            yearly_game_score.loc[i, "경기"] = str(yearly_game_score.loc[i, "경기"]).strip()
            print(str(yearly_game_score.loc[i, "경기"]))
            yearly_game_score.loc[i, "원정팀"] = yearly_game_score.loc[i, "경기"].split(" ")[0]
            yearly_game_score.loc[i, "홈팀"] = yearly_game_score.loc[i, "경기"].split(" ")[2]

            if yearly_game_score.loc[i, "경기"].split(" ")[1] == "VS":
                yearly_game_score.loc[i, "원정팀점수"] = "경기취소"
                yearly_game_score.loc[i, "홈팀점수"] = "경기취소"
            else:
                yearly_game_score.loc[i, "원정팀점수"] = yearly_game_score.loc[i, "경기"].split(" ")[1].split(":")[0]
                yearly_game_score.loc[i, "홈팀점수"] = yearly_game_score.loc[i, "경기"].split(" ")[1].split(":")[1]

            if yearly_game_score.loc[i, "원정팀점수"] > yearly_game_score.loc[i, "홈팀점수"]:
                yearly_game_score.loc[i, "승자"] = yearly_game_score.loc[i, "원정팀"]
            elif yearly_game_score.loc[i, "원정팀점수"] < yearly_game_score.loc[i, "홈팀점수"]:
                yearly_game_score.loc[i, "승자"] = yearly_game_score.loc[i, "홈팀"]
            else:
                yearly_game_score.loc[i, "승자"] = "무승부"

        yearly_game_score.drop("경기",axis=1)
        yearly_game_score.to_csv("e:/WAI/data/KBO/{0}_game_result(cp949).csv".format(year), index=False,
                                 encoding="cp949")