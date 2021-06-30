# -*- coding: utf-8 -*-
from selenium import webdriver
from bs4 import BeautifulSoup
import pandas as pd
import requests
import datetime
import os

# SMTP접속을 위한 서버, 계정 설정
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 465

# 보내는 메일 계정
SMTP_USER = "wook1990@gmail.com"
SMTP_PASSWORD = 'ljjeufaklztvfabn'

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


# 이메일 유효성 검사 함수
def is_valid(addr):
    import re
    if re.match('(^[a-zA-Z-0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$)', addr):
        return True
    else:
        return False


# 이메일 보내기 함수
def send_mail(addr, subj_layout, cont_layout, attachment=None):
    from email.mime.text import MIMEText
    from email.mime.multipart import MIMEMultipart
    import smtplib

    if not is_valid(addr):
        print("Wrong email: " + addr)
        return

    # 텍스트 파일
    msg = MIMEMultipart("alternative")
    # 첨부파일이 있는 경우 mixed로 multipart 생성
    if attachment:
        msg = MIMEMultipart('mixed')
    msg["From"] = SMTP_USER
    msg["To"] = addr
    msg["Subject"] = subj_layout
    contents = cont_layout
    text = MIMEText(_text=contents, _charset="utf-8")
    msg.attach(text)
    # 첨부파일이 있으면
    if attachment:
        from email.mime.base import MIMEBase
        from email import encoders
        file_data = MIMEBase("application", "octect-stream")
        file_data.set_payload(open(attachment, "rb").read())
        encoders.encode_base64(file_data)
        import os
        filename = os.path.basename(attachment)
        file_data.add_header("Content-Disposition", 'attachment', filename=('UTF-8', '', filename))
        msg.attach(file_data)
    # smtp로 접속할 서버 정보를 가진 클래스변수 생성
    smtp = smtplib.SMTP_SSL(SMTP_SERVER, SMTP_PORT)
    # 해당 서버로 로그인
    smtp.login(SMTP_USER, SMTP_PASSWORD)
    # 메일 발송
    smtp.sendmail(SMTP_USER, addr, msg.as_string())
    # 닫기
    smtp.close()


if __name__ == "__main__":

    # Used Beautiful Soup
    # AI NEWS
    url = "http://www.aitimes.kr/news/articleList.html?page=1&total=924&sc_section_code=S1N2&sc_sub_section_code=&sc_serial_code=&sc_area=&sc_level=&sc_article_type=&sc_view_level=&sc_sdate=&sc_edate=&sc_serial_number=&sc_word="
    req = requests.get(url)
    html = req.content.decode("utf-8",'replace')

    soup = BeautifulSoup(html,"html.parser")
    title_html = soup.find_all('div',{'class':'table-row'})
    category, title, date, url = [], [], [] ,[]
    prefix_url = "http://www.aitimes.kr/news/"

    for i in title_html:
        category.append(i.find("small",{"class":'list-section'}).text.replace("\xa0"," "))
        title.append(i.find("strong").text)
        date.append(i.find("div",{"class":"list-dated table-cell"}).text.split(" | ")[1].split(" ")[0])
        url.append(prefix_url + i.find('a')['href'])

    ai_news_data = pd.DataFrame(columns = ["category","title","date","url"],
                                data={"category":category,"title":title,"date":date,"url":url})

    # BI NEWS
    url = "http://www.bikorea.net/news/articleList.html?page=1&total=27274&sc_section_code=&sc_sub_section_code=&sc_serial_code=&sc_area=&sc_level=&sc_article_type=&sc_view_level=&sc_sdate=&sc_edate=&sc_serial_number=&sc_word=&view_type="
    req = requests.get(url)
    html = req.content.decode('euc-kr', 'replace')

    soup = BeautifulSoup(html, "html.parser", )
    title_table = soup.find_all('td', {'class': 'ArtList_Title'})
    date_table = soup.find_all('td', {'class': 'View_SmFont FontEng'})

    category = []
    title = []
    date = []
    url = []
    prefix_url = "http://www.bikorea.net/news/"

    for i, j in zip(title_table, date_table):
        str = i.text.replace("]", "] |")
        str = str.split(" | ")
        category.append(str[0])
        title.append(str[1])
        date.append(j.text)
        url.append(prefix_url + i.find('a')['href'])

    bi_news_data = pd.DataFrame(columns=["category", "title", "date", "url"],
                                data={"category": category, "title": title, "date": date, "url": url})

    # 오늘날짜 기사만 추출
    today_stmp = datetime.datetime.today()
    yesterday_stmp = datetime.datetime.today() - datetime.timedelta(days=1)
    today_per = today_stmp.strftime("%Y-%m-%d")
    yesterday_per = yesterday_stmp.strftime("%Y-%m-%d")
    today_ai_news_date = ai_news_data.loc[ai_news_data.date == yesterday_per]
    today_bi_news_date = bi_news_data.loc[bi_news_data.date == yesterday_per]
    today = today_stmp.strftime('%Y%m%d')


    file_name = 'D:/WAI/O/AINEWS/{}_article.xlsx'.format(today)

    if len(ai_news_data)!=0 or len(bi_news_data)!=0:

        # 엑셀 파일로 내리기
        # 최초 생성 이후 mode는 append; 새로운 시트를 추가합니다.
        if not os.path.exists(file_name):
            with pd.ExcelWriter(file_name, mode='w', engine='openpyxl') as writer:
                bi_news_data.to_excel(writer, index=False, sheet_name="BI_NEWS")
                ai_news_data.to_excel(writer, index=False, sheet_name="AI_NEWS")
        else:
            with pd.ExcelWriter(file_name, mode='a', engine='openpyxl') as writer:
                ai_news_data.to_excel(writer, index=False, sheet_name="BI_NEWS")
                ai_news_data.to_excel(writer, index=False, sheet_name="AI_NEWS")

        cont = today_per + "\n" + "금일 인공지능신문 및 BI뉴스 입니다."
        file_path = 'D:/WAI/O/AINEWS/{}_article.xlsx'.format(today)

        '''
        email_list = ["raphaellee@waikorea.com", "zelico@waikorea.com", "davidjung@waikorea.com", "edward_oh@waikorea.com",
                      "jameskim@waikorea.com", "saeulee@waikorea.com", "gauri617@waikorea.com"]
        '''
        email_list = ["wook1990@gmail.com"]
        for i in email_list:
            send_mail(i, "[WAI] {0}_기술동향기사".format(today), cont, attachment=file_path)

    else:
        cont = today_per + "\n" + "금일은 신규 기사가 업데이트 되지 않았습니다."

        '''
        email_list = ["raphaellee@waikorea.com", "zelico@waikorea.com", "davidjung@waikorea.com", "edward_oh@waikorea.com",
                      "jameskim@waikorea.com", "saeulee@waikorea.com", "gauri617@waikorea.com"]
        '''
        email_list = ["wook1990@gmail.com"]
        for i in email_list:
            send_mail(i, "[WAI] {0}_기술동향기사".format(today), cont, attachment=None)












