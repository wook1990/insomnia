from selenium import webdriver
from bs4 import BeautifulSoup
import pandas as pd
import time



def session_open(driver_dir, open_url):

    options = webdriver.ChromeOptions()
    #options.add_argument('headless')
    options.add_argument('window-size=1920x1080')
    options.add_argument("disable-gpu")
    options.add_argument("user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/61.0.3163.100 Safari/537.36")

    driver = webdriver.Chrome(driver_dir, options= options)
    driver.get(open_url)
    return driver


if __name__ == "__main__" :
    # office

    chromdriver = "D:/99.study/chromedriver.exe"
    # MAC
    #chromdriver = "/usr/local/bin/chromedriver"


    # 타자 URL(Basic1 기본기록_1, Basic2 기본기록_2, Detail 세부기록)
    url_hitter = "https://www.koreabaseball.com/Record/Player/HitterBasic/Basic1.aspx"
    driver = session_open(chromdriver, url_hitter)

    # 2001  년까진 세부기록 없고 2002년부터 세부기록

    # 년도 입력
    year_field = driver.find_element_by_xpath("//select[@name='ctl00$ctl00$ctl00$cphContents$cphContents$cphContents$ddlSeason$ddlSeason']")

    driver.implicitly_wait(30);time.sleep(1)
    year_field.send_keys('2020')

    # 정규 시즌 데이터 팀별
    team_field = driver.find_element_by_xpath("//select[@name='ctl00$ctl00$ctl00$cphContents$cphContents$cphContents$ddlTeam$ddlTeam']")
    team_name = team_field.text.split("\n ")[1:]
    # 팀 입력

    #  선수 성적을 닮을 테이블 초기화
    player_record_df = pd.DataFrame()

    for i in team_name:
        print("팀 이름 : " + i)
        team_field = driver.find_element_by_xpath(
            "//select[@name='ctl00$ctl00$ctl00$cphContents$cphContents$cphContents$ddlTeam$ddlTeam']")

        driver.implicitly_wait(30);time.sleep(1)
        team_field.send_keys(i)
        driver.implicitly_wait(30);time.sleep(1)
        paging = driver.find_element_by_class_name("paging")
        page_num = paging.text.split(" ")
        driver.implicitly_wait(30);time.sleep(1)

        # 기본기록_1
        # Default value list
        rank = []; name = []; team = []; har_rt = []; game_cn = []; pa_cn = []; ab_cn = []; run_cn = []
        hit_cn = []; h2_cn = []; h3_cn = []; hr_cn = []; tb_cn = []; rbi_cn = []; sh_cn = []; sf_cn = []
        # 페이지 이동
        if len(page_num) == 1:
            element_id = "cphContents_cphContents_cphContents_ucPager_btnNo1"
            paging = driver.find_element_by_class_name("paging")
            driver.implicitly_wait(30)
            paging.find_element_by_id(element_id).click()
            driver.implicitly_wait(30);time.sleep(3)

            page = driver.page_source
            driver.implicitly_wait(30)
            soup = BeautifulSoup(page,"html.parser")
            table = soup.find('table',{'class' : 'tData01 tt'}).find('tbody')
            trs = table.find_all('tr')
            print("get data in table row")

            for tr in trs:

                tds = tr.find_all('td')
                rank.append(tds[0].text.strip())
                name.append(tds[1].text.strip())
                team.append(tds[2].text.strip())
                har_rt.append(tds[3].text.strip())
                game_cn.append(tds[4].text.strip())
                pa_cn.append(tds[5].text.strip())
                ab_cn.append(tds[6].text.strip())
                run_cn.append(tds[7].text.strip())
                hit_cn.append(tds[8].text.strip())
                h2_cn.append(tds[9].text.strip())
                h3_cn.append(tds[10].text.strip())
                hr_cn.append(tds[11].text.strip())
                tb_cn.append(tds[12].text.strip())
                rbi_cn.append(tds[13].text.strip())
                sh_cn.append(tds[14].text.strip())
                sf_cn.append(tds[15].text.strip())

            player_record = { "rank" : rank ,
                              "name" : name ,
                              "team" : team ,
                              "har_rt" : har_rt,
                              "game_cn" : game_cn,
                              "pa_cn" : pa_cn ,
                              "ab_cn" : ab_cn ,
                              "run_cn" : run_cn ,
                              "hit_cn" : hit_cn ,
                              "h2_cn" : h2_cn ,
                              "h3_cn" : h3_cn ,
                              "hr_cn" : hr_cn ,
                              "tb_cn" : tb_cn ,
                              "rbi_cn" : rbi_cn ,
                              "sh_cn" : sh_cn ,
                              "sf_cn" : sf_cn
            }
            player_rc_df = pd.DataFrame.from_dict(player_record, orient="columns", dtype = None, columns = None)
            player_record_df = pd.concat([player_record_df, player_rc_df])

        else:
            for i in page_num:
                print("page number : " + i )
                element_id = "cphContents_cphContents_cphContents_ucPager_btnNo" + i
                paging = driver.find_element_by_class_name("paging")
                driver.implicitly_wait(30)
                paging.find_element_by_id(element_id).click()
                driver.implicitly_wait(30); time.sleep(3)

                page = driver.page_source
                driver.implicitly_wait(30)
                soup = BeautifulSoup(page, "html.parser")
                table = soup.find('table', {'class': 'tData01 tt'}).find('tbody')
                trs = table.find_all('tr')

                for tr in trs:
                    tds = tr.find_all('td')
                    rank.append(tds[0].text.strip())
                    name.append(tds[1].text.strip())
                    team.append(tds[2].text.strip())
                    har_rt.append(tds[3].text.strip())
                    game_cn.append(tds[4].text.strip())
                    pa_cn.append(tds[5].text.strip())
                    ab_cn.append(tds[6].text.strip())
                    run_cn.append(tds[7].text.strip())
                    hit_cn.append(tds[8].text.strip())
                    h2_cn.append(tds[9].text.strip())
                    h3_cn.append(tds[10].text.strip())
                    hr_cn.append(tds[11].text.strip())
                    tb_cn.append(tds[12].text.strip())
                    rbi_cn.append(tds[13].text.strip())
                    sh_cn.append(tds[14].text.strip())
                    sf_cn.append(tds[15].text.strip())

            player_record = {"rank": rank,
                         "name": name,
                         "team": team,
                         "har_rt": har_rt,
                         "game_cn": game_cn,
                         "pa_cn": pa_cn,
                         "ab_cn": ab_cn,
                         "run_cn": run_cn,
                         "hit_cn": hit_cn,
                         "h2_cn": h2_cn,
                         "h3_cn": h3_cn,
                         "hr_cn": hr_cn,
                         "tb_cn": tb_cn,
                         "rbi_cn": rbi_cn,
                         "sh_cn": sh_cn,
                         "sf_cn": sf_cn
                         }
            player_rc_df = pd.DataFrame.from_dict(player_record, orient="columns", dtype=None, columns=None)
            player_record_df = pd.concat([player_record_df, player_rc_df])

    # 기본기록_2
    # 수정 사항 : 기본 타자 투수 수비 주루 모든 태그와 클래스는 동일하게 구성
    # thead의 컬럼만 가져와서 값할당후 스스로 초기화 가능하게 하면 함수 하나로
    # 기록 레코드 크롤링 가능 예상




    # 기본기록_2
    # 페이지 이동
    # 세부기록
    # 페이지 이동

    # 투수 URL
    url_pitcher = "https://www.koreabaseball.com/Record/Player/PitcherBasic/Basic1.aspx"

    # 수비 URL
    url_defense = "https://www.koreabaseball.com/Record/Player/Defense/Basic1.aspx"

    # 주루 URL
    url_runner = "https://www.koreabaseball.com/Record/Player/Runner/Basic1.aspx"




