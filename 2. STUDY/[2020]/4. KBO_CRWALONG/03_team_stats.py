from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from bs4 import BeautifulSoup
import pandas as pd
import time
import sys
import numpy as np

# office
    #chromdriver = "D:/99.study/chromedriver.exe"
    # HomePC
    # chromdriver = "E:/Download/chromedriver.exe"
    # MAC
    #chromdriver = "/usr/local/bin/chromedriver"

CHROM_DRIVER_DIR = "E:/Download/chromedriver.exe"
SAVE_PATH = "e:/WAI/data/KBO/"

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
    #options.add_argument('headless')
    options.add_argument('window-size=1920x1080')
    options.add_argument("disable-gpu")
    options.add_argument("user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/61.0.3163.100 Safari/537.36")

    driver = webdriver.Chrome(driver_dir, options= options)
    driver.get(open_url)
    return driver

def get_team_info(year):

    # 년도 입력
    year_field = driver.find_element_by_xpath(
        "//select[@name='ctl00$ctl00$ctl00$cphContents$cphContents$cphContents$ddlSeason$ddlSeason']")

    driver.implicitly_wait(30);
    time.sleep(1)
    year_field.send_keys(str(year))
    time.sleep(1)
    #year_field.send_keys(Keys.RETURN)

    # 정규 시즌 데이터 팀별
    #  선수 성적을 닮을 테이블 초기화
    team_record_df = pd.DataFrame()


    line_logging("read page source")
    page = driver.page_source
    soup = BeautifulSoup(page, "html.parser")
    line_logging("find table data")
    soup.find('table',{})
    table = soup.find('table', {'class': 'tData tt'}).find('thead')
    th = table.find_all("th")
    col_list = []
    line_logging("make var name")
    for columns in th:
        col_list.append(columns.text)
    col_list[0] = 'RANK'
    col_list[1] = 'TEAM_NAME'
    print(col_list)
    for colname in col_list:
        if colname == 'PH-BA':
            colname = "PH_BA"
            col_list[col_list.index("PH-BA")] = colname
            globals()[colname] = []
        elif colname == 'CS%':
            colname = "CS_RATIO"
            col_list[col_list.index("CS%")] = colname
            globals()[colname] = []
        elif colname == 'SB%':
            colname = "SB_RATIO"
            col_list[col_list.index("SB%")] = colname
            globals()[colname] = []
        elif colname == '2B':
            colname = "TWO_BASE"
            col_list[col_list.index("2B")] = colname
            globals()[colname] = []
        elif colname == '3B':
            colname = "TREE_BASE"
            col_list[col_list.index("3B")] = colname
            globals()[colname] = []
        else:
            globals()[colname] = []

    page = driver.page_source
    driver.implicitly_wait(30)
    soup = BeautifulSoup(page, "html.parser")
    table = soup.find('table', {'class': 'tData tt'}).find('tbody')
    trs = table.find_all('tr')
    line_logging("get data in table row")

    for tr in trs:
        tds = tr.find_all('td')
        for i in range(0, len(tds)):
            globals()[col_list[i]].append(tds[i].text)

    team_info_dict = {}

    for i in range(0, len(col_list)):
        # line_logging(i)
        team_info_dict[col_list[i]] = eval(col_list[i])

    team_rc_df = pd.DataFrame.from_dict(team_info_dict, orient="columns", dtype=None, columns=None)
    team_record_df = pd.concat([team_record_df, team_rc_df])

    return team_record_df


if __name__ == "__main__" :


    url_team_hitter_basic = "https://www.koreabaseball.com/Record/Team/Hitter/Basic1.aspx"
    url_team_hitter_basic2 = "https://www.koreabaseball.com/Record/Team/Hitter/Basic2.aspx"
    url_team_pitcher_basic = "https://www.koreabaseball.com/Record/Team/Pitcher/Basic1.aspx"
    url_team_pitcher_basic2 = "https://www.koreabaseball.com/Record/Team/Pitcher/Basic2.aspx"
    url_team_defense_basic = "https://www.koreabaseball.com/Record/Team/Defense/Basic.aspx"
    url_team_runner_basic = "https://www.koreabaseball.com/Record/Team/Runner/Basic.aspx"



    data_url = ["url_team_hitter_basic", "url_team_hitter_basic2", "url_team_pitcher_basic", "url_team_pitcher_basic2",
                "url_team_defense_basic","url_team_runner_basic"]

    for url in data_url:
        line_logging(url + " 작업 시작 되었습니다.")
        driver = session_open(CHROM_DRIVER_DIR ,eval(url))
        year = np.arange(2002, 2021, 1)
        for y in year:
            line_logging(str(y) + "년도 데이터 크롤링 중입니다.")
            df =  get_team_info(y)
            df["year"] = y
            line_logging(str(y) + "년도 데이터 크롤링 완료되었습니다.")
            df.to_csv(SAVE_PATH + "{0}_{1}(cp949).csv".format(y, url.split("url_")[1]), index=False, encoding="cp949")
        driver.quit()
        line_logging(url + "의 작업이 완료되었습니다.")
        time.sleep(2)