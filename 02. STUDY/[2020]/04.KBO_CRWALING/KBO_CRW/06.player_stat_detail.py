from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from bs4 import BeautifulSoup
import pandas as pd
import time
import os
import sys

CHROM_DRIVER_DIR = "E:/Download/chromedriver.exe"
SAVE_PATH = "e:/WAI/data/KBO/"

'''
Using Selenium  
- 전역 변수  
  CHROM_DRIVER_DIR : 크롬 드라이버 설치 경로
  SAVE_PATH : 데이터 저장경로
- 외부입력파라미터
   year : 년도별 데이터 크롤링
   (초기 적재시 year = np.arange(2001, 2021, 1)의 리스트를 받아 전체 데이터 적재)
   제공된 코드는 1년씩 파라미터를 받아서 데이터 수집
'''


# 로깅 함수
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


def session_open(driver_dir, open_url):

    options = webdriver.ChromeOptions()
    options.add_argument('headless')
    options.add_argument('window-size=1920x1080')
    options.add_argument("disable-gpu")
    options.add_argument("user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/61.0.3163.100 Safari/537.36")

    driver = webdriver.Chrome(driver_dir, options= options)
    driver.get(open_url)
    return driver

def get_player_info(year, file_name):

    year_field = driver.find_element_by_xpath(
        "//select[@name='ctl00$ctl00$ctl00$cphContents$cphContents$cphContents$ddlSeason$ddlSeason']")
    driver.implicitly_wait(2);
    time.sleep(1)
    year_field.send_keys(str(year))
    driver.implicitly_wait(1)
    time.sleep(1)
    # 정규 시즌 데이터 팀별
    team_field = driver.find_element_by_xpath(
        "//select[@name='ctl00$ctl00$ctl00$cphContents$cphContents$cphContents$ddlTeam$ddlTeam']")
    team_name = team_field.text.split("\n ")[1:]

    try:
        team_name.remove('')
        line_logging(team_name)
    except:
        line_logging(team_name)

    # situation 1
    situation_field_basic = driver.find_element_by_xpath(
        "//select[@name='ctl00$ctl00$ctl00$cphContents$cphContents$cphContents$ddlSituation$ddlSituation']"
    )
    driver.implicitly_wait(1)
    time.sleep(2)

    situation_basic_name = situation_field_basic.text.split("\n")[1:]
    try:
        situation_basic_name.remove('')
        line_logging(situation_basic_name)
    except:
        line_logging(situation_basic_name)

    player_detail_record_df = pd.DataFrame()
    line_logging("START DATA CRAWLING")
    count = 1

    for i in team_name:
        line_logging("팀 이름 : " + i)
        team_field = driver.find_element_by_xpath(
            "//select[@name='ctl00$ctl00$ctl00$cphContents$cphContents$cphContents$ddlTeam$ddlTeam']")
        time.sleep(1)
        line_logging("TEAM SELECTION")

        team_field.send_keys(i)
        driver.implicitly_wait(2);
        time.sleep(1)
        situation_df = pd.DataFrame()
        for j in situation_basic_name:
            line_logging("경기상황1 : " + j)
            situation_field_basic = driver.find_element_by_xpath(
                "//select[@name='ctl00$ctl00$ctl00$cphContents$cphContents$cphContents$ddlSituation$ddlSituation']"
            )
            situation_field_basic.send_keys(j)
            driver.implicitly_wait(1)
            time.sleep(1)
            situation_field_basic.send_keys(Keys.RETURN)
            driver.implicitly_wait(2);
            time.sleep(2);
            detail_situation = driver.find_element_by_xpath(
                "//select[@name='ctl00$ctl00$ctl00$cphContents$cphContents$cphContents$ddlSituationDetail$ddlSituationDetail']"
            )
            detail_situation_name = detail_situation.text.split("\n")[1:]
            line_logging(detail_situation_name)

            detail_df = pd.DataFrame()

            for k in detail_situation_name:
                line_logging("경기상황2 : " + k)
                detail_situation = driver.find_element_by_xpath(
                    "//select[@name='ctl00$ctl00$ctl00$cphContents$cphContents$cphContents$ddlSituationDetail$ddlSituationDetail']"
                )
                driver.implicitly_wait(1)
                time.sleep(2)

                detail_situation.send_keys(k)
                driver.implicitly_wait(1)
                detail_situation.send_keys(Keys.ENTER)
                driver.implicitly_wait(1)
                time.sleep(2)

                page = driver.page_source
                soup = BeautifulSoup(page, "html.parser")
                table = soup.find('table', {'class': 'tData01 tt'}).find('thead')
                th = table.find_all("th")
                col_list = []
                line_logging("MAKE VAR NAME AND CREATE VARIABLE")
                for columns in th:
                    col_list.append(columns.text)
                col_list[0] = 'RANK'
                col_list[1] = 'PLAYER_NAME'
                col_list[2] = 'TEAM_NAME'
                line_logging(col_list)
                for colname in col_list:
                    if colname == "2B":
                        colname = "TWO_BASE"
                        col_list[col_list.index("2B")] = colname
                        globals()[colname] = []
                    elif colname == "3B":
                        colname = "THREE_BASE"
                        col_list[col_list.index("3B")] = colname
                        globals()[colname] = []
                    elif colname == "PH-BA":
                        colname = "PH_BA"
                        col_list[col_list.index("PH-BA")] = colname
                        globals()[colname] = []
                    elif colname == "CS%":
                        colname = "CS_RATIO"
                        col_list[col_list.index("CS%")] = colname
                        globals()[colname] = []
                    elif colname == "GO/AO":
                        colname = "GO_AO"
                        col_list[col_list.index("GO/AO")] = colname
                        globals()[colname] = []
                    elif colname == "BB/K":
                        colname = "BB_K"
                        col_list[col_list.index("BB/K")] = colname
                        globals()[colname] = []
                    elif colname == "GW RBI":
                        colname = "GW_RBI"
                        col_list[col_list.index("GW RBI")] = colname
                        globals()[colname] = []
                    elif colname == "P/PA":
                        colname = "P_PA"
                        col_list[col_list.index("P/PA")] = colname
                        globals()[colname] = []
                    elif colname == "P/G":
                        colname = "P_G"
                        col_list[col_list.index("P/G")] = colname
                        globals()[colname] = []
                    elif colname == "P/IP":
                        colname = "P_IP"
                        col_list[col_list.index("P/IP")] = colname
                        globals()[colname] = []
                    elif colname == "K/9":
                        colname = "K_9"
                        col_list[col_list.index("K/9")] = colname
                        globals()[colname] = []
                    elif colname == "BB/9":
                        colname = "BB_9"
                        col_list[col_list.index("BB/9")] = colname
                        globals()[colname] = []
                    elif colname == "K/BB":
                        colname = "K_BB"
                        col_list[col_list.index("K/BB")] = colname
                        globals()[colname] = []
                    else:
                        globals()[colname] = []

                table = soup.find('table', {'class': 'tData01 tt'}).find('tbody')
                trs = table.find_all('tr')
                line_logging("GET DATA IN TABLE ROW")

                for tr in trs:
                    tds = tr.find_all('td')
                    for num in range(0, len(tds)):
                        globals()[col_list[num]].append(tds[num].text)

                detail_info_dict = {}

                for num in range(0, len(col_list)):
                    detail_info_dict[col_list[num]] = eval(col_list[num])

                player_detail_df = pd.DataFrame.from_dict(detail_info_dict, orient="columns", dtype=None, columns=None)
                player_detail_df["SITUATION_ 2"] = k.replace(" ","")
                detail_df = pd.concat([detail_df, player_detail_df])
                line_logging("\n" + detail_df.tail(1))

            detail_df["SITUATION_1"] = j.replace(" ","")
            situation_df = pd.concat([situation_df, detail_df])
            line_logging("\n" + situation_df.tail(1))
            line_logging("접근제어 방지를 위해 3초간 작업을 중지하겠습니다.")
            time.sleep(3)

        player_detail_record_df = pd.concat([player_detail_record_df, situation_df])
        if os.path.isdir(SAVE_PATH + "team_each_detail/"):
            situation_df.to_csv(SAVE_PATH + "team_each_detail/{0}_{1}_{2}_detail.csv".format(year, i, file_name), index=False, encoding ="cp949")
        else:
            os.mkdir(SAVE_PATH + "team_each_detail/")

        line_logging("\n" + player_detail_record_df.tail(1))

        count = count + 1
        if count%3 == 0 :
            line_logging("접근제어 방지를 위해 7초간 휴식하겠습니다.")
            time.sleep(7)

    return player_detail_record_df



if __name__ == "__main__" :

    # 상황별 데이터 크롤링
    url_hitter_situation = "https://www.koreabaseball.com/Record/Player/HitterBasic/Situation.aspx"
    url_pitcher_situation = "https://www.koreabaseball.com/Record/Player/PitcherBasic/Situation.aspx"

    year = sys.argv[1]
    data_url = ["url_hitter_situation" , "url_pitcher_situation"]

    for url in data_url:

        line_logging(eval(url) + " 작업을 시작합니다")
        driver = session_open(CHROM_DRIVER_DIR, eval(url))
        time.sleep(2)

        line_logging(str(year)  + "년의 데이터를 크롤링 중입니다."); file_name = url.split("_")[1]
        df = get_player_info(year, file_name)
        df["year"] = year
        df.to_csv(SAVE_PATH + "{0}_{1}(cp949).csv".format(year, url.split("url_")[1]), index=False, encoding="cp949")
        line_logging(str(year) + "년의 데이터를 크롤링 완료하였습니다.")

        driver.quit()
        line_logging(url + "의 작업이 완료되었습니다.")
        line_logging("다음 작업을 위해 10초간 대기 하겠습니다.")
        time.sleep(10)

    line_logging(str(year) +"의 작업이 종료되었습니다.")


