from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from bs4 import BeautifulSoup
import pandas as pd
import time
import sys
import numpy as np

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
    options.add_argument('headless')
    options.add_argument('window-size=1920x1080')
    options.add_argument("disable-gpu")
    options.add_argument("user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/61.0.3163.100 Safari/537.36")

    driver = webdriver.Chrome(driver_dir, options= options)
    driver.get(open_url)
    return driver

def get_player_info(year):
    # 년도 입력

    year_field = driver.find_element_by_xpath(
        "//select[@name='ctl00$ctl00$ctl00$cphContents$cphContents$cphContents$ddlSeason$ddlSeason']")

    driver.implicitly_wait(30);
    time.sleep(1)
    year_field.send_keys(str(year))
    time.sleep(1)
    #year_field.send_keys(Keys.RETURN)
    # 정규 시즌 데이터 팀별
    team_field = driver.find_element_by_xpath(
        "//select[@name='ctl00$ctl00$ctl00$cphContents$cphContents$cphContents$ddlTeam$ddlTeam']")
    team_name = team_field.text.split("\n ")[1:]
    line_logging(team_name)
    try:
        team_name.remove('')
        line_logging(team_name)
    except:
        pass
        line_logging(team_name)
    # 팀 입력
    #  선수 성적을 닮을 테이블 초기화
    player_record_df = pd.DataFrame()

    for i in team_name:
        line_logging("팀 이름 : " + i)
        team_field = driver.find_element_by_xpath(
            "//select[@name='ctl00$ctl00$ctl00$cphContents$cphContents$cphContents$ddlTeam$ddlTeam']")
        time.sleep(1)
        line_logging("team selection")
        team_field.send_keys(i)
        '''
        try:
            team_field.send_keys(Keys.RETURN)
        except:
            pass
        '''
        driver.implicitly_wait(30);
        time.sleep(2)
        line_logging("page number check")
        paging = driver.find_element_by_class_name("paging")
        page_num = paging.text.split(" ")
        driver.implicitly_wait(30);
        time.sleep(1)

        line_logging("read page source")
        page = driver.page_source
        soup = BeautifulSoup(page, "html.parser")
        line_logging("find table data")
        table = soup.find('table', {'class': 'tData01 tt'}).find('thead')
        th = table.find_all("th")
        col_list = []
        line_logging("make var name")
        for columns in th:
            col_list.append(columns.text)
        col_list[0] = 'RANK'
        col_list[1] = 'PLAYER_NAME'
        col_list[2] = 'TEAM_NAME'
        print(col_list)
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

        if len(page_num) == 1:
            element_id = "cphContents_cphContents_cphContents_ucPager_btnNo1"

            paging.find_element_by_id(element_id).click()
            driver.implicitly_wait(30);time.sleep(3);

            page = driver.page_source
            driver.implicitly_wait(30)
            soup = BeautifulSoup(page, "html.parser")
            table = soup.find('table', {'class': 'tData01 tt'}).find('tbody')
            trs = table.find_all('tr')
            line_logging("get data in table row")

            for tr in trs:
                tds = tr.find_all('td')
                for i in range(0, len(tds)):
                    globals()[col_list[i]].append(tds[i].text)

            # 딕셔너리에 데이터 담기
            player_info_dict = {}

            for i in range(0, len(col_list)):
                # line_logging(i)
                player_info_dict[col_list[i]] = eval(col_list[i])

            player_rc_df = pd.DataFrame.from_dict(player_info_dict, orient="columns", dtype=None, columns=None)
            player_record_df = pd.concat([player_record_df, player_rc_df])


        else:
            for i in page_num:
                line_logging("page number : " + i)
                element_id = "cphContents_cphContents_cphContents_ucPager_btnNo" + i
                paging = driver.find_element_by_class_name("paging")
                driver.implicitly_wait(30)
                paging.find_element_by_id(element_id).click()
                driver.implicitly_wait(30);
                time.sleep(1)

                page = driver.page_source
                driver.implicitly_wait(30)
                soup = BeautifulSoup(page, "html.parser")
                table = soup.find('table', {'class': 'tData01 tt'}).find('tbody')
                trs = table.find_all('tr')
                line_logging("get data in table row")

                for tr in trs:
                    tds = tr.find_all('td')
                    for i in range(0, len(tds)):
                        globals()[col_list[i]].append(tds[i].text)

            paging = driver.find_element_by_class_name("paging")
            element_id = "cphContents_cphContents_cphContents_ucPager_btnNo1"
            paging.find_element_by_id(element_id).click()
            driver.implicitly_wait(30);
            time.sleep(3)

            # 딕셔너리에 데이터 담기
            player_info_dict = {}

            for i in range(0, len(col_list)):
                # line_logging(i)
                player_info_dict[col_list[i]] = eval(col_list[i])

            player_rc_df = pd.DataFrame.from_dict(player_info_dict, orient="columns", dtype=None, columns=None)
            player_record_df = pd.concat([player_record_df, player_rc_df])

    return player_record_df



if __name__ == "__main__" :

    # 타자 URL(Basic1 기본기록_1, Basic2 기본기록_2, Detail 세부기록)
    url_hitter_basic = "https://www.koreabaseball.com/Record/Player/HitterBasic/Basic1.aspx"
    url_hitter2_basic = "https://www.koreabaseball.com/Record/Player/HitterBasic/Basic2.aspx"
    url_hitter_detail = "https://www.koreabaseball.com/Record/Player/HitterBasic/Detail1.aspx"
    url_pitcher_basic = "https://www.koreabaseball.com/Record/Player/PitcherBasic/Basic1.aspx"
    url_pitcher2_basic = 'https://www.koreabaseball.com/Record/Player/PitcherBasic/Basic2.aspx'
    url_pitcher_detail = "https://www.koreabaseball.com/Record/Player/PitcherBasic/Detail1.aspx"
    url_pitcher2_detail = "https://www.koreabaseball.com/Record/Player/PitcherBasic/Detail2.aspx"
    url_defense = "https://www.koreabaseball.com/Record/Player/Defense/Basic.aspx"
    url_runner = "https://www.koreabaseball.com/Record/Player/Runner/Basic.aspx"

    # 주소값
    data_url = ["url_hitter_basic", "url_hitter2_basic", "url_hitter_detail", "url_pitcher_basic", "url_pitcher2_basic",
                "url_pitcher_detail", "url_pitcher2_detail", "url_defense", "url_runner"]


    for url in data_url:

        line_logging(eval(url) + " 작업을 시작합니다")
        year = np.arange(2002, 2021, 1).tolist()
        driver = session_open(CHROM_DRIVER_DIR, eval(url))
        time.sleep(2)
        for i in year:
            line_logging(str(i)  + "년의 데이터를 크롤링 중입니다.")
            df = get_player_info(i)
            df["year"] = i
            df.to_csv(SAVE_PATH + "{0}_{1}(cp949).csv".format(i, url.split("url_")[1]), index=False, encoding="cp949")
            line_logging(str(i) + "년의 데이터를 크롤링 완료하였습니다.")
        driver.quit()
        line_logging(url + "의 작업이 완료되었습니다.")
        time.sleep(2)
