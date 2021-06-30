from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from bs4 import BeautifulSoup
import pandas as pd
import time
import sys
import numpy as np

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
    #options.add_argument('headless')
    options.add_argument('window-size=1920x1080')
    options.add_argument("disable-gpu")
    options.add_argument("user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/61.0.3163.100 Safari/537.36")

    driver = webdriver.Chrome(driver_dir, options= options)
    driver.get(open_url)
    return driver

def get_team_record(year):


    year_field = driver.find_element_by_xpath(
        "//select[@name='ctl00$ctl00$ctl00$cphContents$cphContents$cphContents$ddlYear']")

    driver.implicitly_wait(30);
    time.sleep(1)
    year_field.send_keys(str(year))
    time.sleep(1)

    team_record_df = pd.DataFrame()

    line_logging("read page source")
    page = driver.page_source
    soup = BeautifulSoup(page, "html.parser")
    line_logging("find table data")
    soup.find('table', {})
    table = soup.find('table', {'class': 'tData'}).find('thead')
    th = table.find_all("th")
    col_list = []
    line_logging("make var name")
    for columns in th:
        col_list.append(columns.text)
    print(col_list)

    for colname in col_list:
        globals()[colname] = []

    page = driver.page_source
    driver.implicitly_wait(30)
    soup = BeautifulSoup(page, "html.parser")
    table = soup.find('table', {'class': 'tData'}).find('tbody')
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

    return team_rc_df


if __name__ == "__main__" :

    url = "https://www.koreabaseball.com/TeamRank/TeamRank.aspx"
    driver = session_open(CHROM_DRIVER_DIR,url)
    year = sys.argv[1]

    line_logging(str(year) + "년도 팀 성적표 크롤링 시작합니다.")
    df = get_team_record(year)
    print(df.info())
    line_logging(str(year) + "년도 팀 성적표 크롤링 완료되었습니다.")
    df.to_csv(SAVE_PATH + "{0}_team_record(cp949).csv".format(y), index=False, encoding="cp949", date_format = None)




