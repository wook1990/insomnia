from selenium import webdriver
from bs4 import BeautifulSoup
import pandas as pd
import time

CHROM_DRIVER_DIR = "E:/Download/chromedriver.exe"
SAVE_PATH = "e:/WAI/data/KBO/"


def session_open(driver_dir, open_url):

    options = webdriver.ChromeOptions()
    options.add_argument('headless')
    options.add_argument('window-size=1920x1080')
    options.add_argument("disable-gpu")
    options.add_argument("user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/61.0.3163.100 Safari/537.36")

    driver = webdriver.Chrome(driver_dir, options= options)
    driver.get(open_url)
    return driver


def get_player_info(driver):
    # 선수단 별 드롭다운 옵션이동
    team_select_field = driver.find_element_by_xpath("//select[@name='ctl00$ctl00$ctl00$cphContents$cphContents$cphContents$ddlTeam']")
    driver.implicitly_wait(30)
    time.sleep(1)
    team_name = team_select_field.text.split("\n ")[1:]
    player_total_df = pd.DataFrame()

    # get player info by team name
    for i in team_name:
        print(i)
        team_select_field = driver.find_element_by_xpath(
            "//select[@name='ctl00$ctl00$ctl00$cphContents$cphContents$cphContents$ddlTeam']")
        driver.implicitly_wait(30)
        time.sleep(2)
        team_select_field.send_keys(i)
        driver.implicitly_wait(60)
        time.sleep(2)
        # 팀명, 등번호, 선수명, 포지션, 생년월일, 체격,  출신교
        # 선수단별 선술 리스트 페이지 번호 추출
        paging = driver.find_element_by_class_name("paging")
        page_num = paging.text.split(" ")
        print(page_num)
        driver.implicitly_wait(30)
        time.sleep(2)

        # player info default list
        player_number = []
        player_name = []
        player_team = []
        player_position = []
        player_birth = []
        player_height = []
        player_weight = []
        player_history = []
        # 페이지 번호 별로 선수 크롤링
        for i in page_num:
            print("page number : " + i )
            element_id = "cphContents_cphContents_cphContents_ucPager_btnNo"+i
            print(element_id)
            paging = driver.find_element_by_class_name("paging")
            driver.implicitly_wait(30)
            paging.find_element_by_id(element_id).click()
            driver.implicitly_wait(30)
            time.sleep(5)
            page = driver.page_source
            driver.implicitly_wait(30)
            soup = BeautifulSoup(page,"html.parser")
            table = soup.find('table',{'class' : 'tEx'}).find('tbody')
            trs = table.find_all('tr')

            for idx, tr in enumerate(trs):
                tds = tr.find_all('td')
                player_number.append(tds[0].text.strip())
                player_name.append(tds[1].text.strip())
                player_team.append(tds[2].text.strip())
                player_position.append(tds[3].text.strip())
                player_birth.append(tds[4].text.strip())
                try:
                    player_height.append(tds[5].text.strip().split(", ")[0])
                except:
                    player_height.append("NULL")
                try:
                    player_weight.append(tds[5].text.strip().split(", ")[1])
                except:
                    player_weight.append("NULL")
                player_history.append(tds[6].text.strip())
                driver.implicitly_wait(30)

                player_info = {"player_number": player_number,
                               "player_name": player_name,
                               "player_team": player_team,
                               "player_position": player_position,
                               "player_birth": player_birth,
                               "player_height": player_height,
                               "player_weight": player_weight,
                               "player_history": player_history
                               }

            player_df = pd.DataFrame.from_dict(player_info, orient='columns', dtype=None, columns=None)
            print(player_df.head())

        player_total_df = pd.concat([player_total_df, player_df])
        print(player_total_df.head())

    return player_total_df



'''    
    # 셀레니움 이용
    #table = driver.find_element_by_xpath("//*[@class='tEx']/tbody")
    # page의 html을 parsing 해서 그안의 테이블태그를 통한 크롤링
    # 셀레니움만을 이용한 크롤링은 답없음
    # 셀레니움은 동적웹페이지의 key를 전달해주고 action을 취해주는 방향으로 사용
    time.sleep(5)
    #table_list.append(driver.find_element_by_xpath("//*[@class='tEx']/tbody"))
    #print(table_list)
    print(table)
    table_row = table.find_elements_by_tag_name("tr")
    time.sleep(5)
    print(table_row)
    for tr in table_row:
        td = tr.find_elements_by_tag_name("td")
        time.sleep(1)
        s = " {0}, {1}, {2}, {3}, {4}, {5}, {6}".format(td[0].text, td[1].text, td[2].text, td[3].text, td[4].text, td[5].text, td[6].text)
        print(s)
        player_number.append(td[0].text)
        player_name.append(td[1].text)
        player_team.append(td[2].text)
        player_position.append(td[3].text)
        player_birth.append(td[4].text)
        player_height.append(str(td[5].text).split(", ")[0])
        player_weight.append(str(td[5].text).split(", ")[1])
        player_history.append(td[6].text)
'''








if __name__ == "__main__":


    url = "https://www.koreabaseball.com/Player/Search.aspx"
    driver = session_open(CHROM_DRIVER_DIR, url)

    player_df = get_player_info(driver)

    player_df.to_csv(SAVE_PATH + "player_info_2020(cp949).csv", index=False, encoding="cp949")

    driver.quit()

