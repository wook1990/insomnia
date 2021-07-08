import sys
import os
import pandas as pd
import time
import datetime
import re
import glob
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.action_chains import *
from bs4 import *
from dateutil.relativedelta import relativedelta

import requests
import urllib
import pandas as pd


sys.stdin.encoding # 기본 한글 인코딩 방식 => utf8

from bs4 import BeautifulSoup, Comment

# 오늘 날짜 변수
def crawler(keyword):

    start_date = datetime.datetime.today() - datetime.timedelta(1) # 한달 전 날짜
    start_date = start_date.strftime('%Y/%m/%d')
    end_date = datetime.datetime.today() # 현재 날짜
    end_date = end_date.strftime('%Y/%m/%d')

    # 한글이 저장되는 방식이 네가지(Unicode, UTF8, EUC-KR, CP949)가 있음

    # url은 ASCII 코드 값만 사용됨 : 그래서 1. 한글 => 2. 인코딩 => 3. ASCII 코드 로 변환 과정이 필요함
    # 한글은 ASCII코드로 표현할 수 가 없음(인코딩을 하는 이유)
    # 인코딩의 뜻 : "특정한 문자 집합 안의 문자들을 컴퓨터 시스템에서 사용할 목적으로 일정한 범위 안의 정수들로 변환하는 방법"

    # 나라장터(charset(인코딩) : EUC-KR) 이고 파이썬(UTF-8) 이기 때문에 다른 ASCII CODE 값으로 검색되어지기 때문에 에러가 발생했다.


    url = 'http://www.g2b.go.kr:8101/ep/tbid/tbidList.do?'
    # 검색어 파라미터
    query = {
        'bidNm' : keyword
    }
    search_param = urllib.parse.urlencode(query, encoding = 'EUC-KR')

    # '%BA%D0%BC%AE'.encode('UTF-8')
    # '분석'.encode('EUC-KR').decode('EUC-KR')


    # string type으로 나머지

    for idx in range(0, 10):
            globals()['var_{}'.format(idx)] = []



    # 하루 기간동안 올라온 공고 10page 씩 크롤링
    for page_no in range(1, 11):
            page_no = str(page_no)
            param = '&taskClCds=' \
                    '&searchDtType=1' \
                    '&fromBidDt=' + start_date + \
                    '&toBidDt=' + end_date + \
                    '&fromOpenBidDt=' \
                    '&toOpenBidDt=' \
                    '&radOrgan=1' \
                    '&instNm=' \
                    '&area=' \
                    '&regYn=Y' \
                    '&bidSearchType=1' \
                    '&searchType=1' \
                    '&currentPageNo=' + page_no

            # url 정보
            all_url = url + search_param + param

            request_result = requests.get(all_url)
            html_co_cel = BeautifulSoup(request_result.text, 'html.parser')

            # 항상 find_all 함수를 사용하여 태그 정보를 가져오면 list 객체로 가져오게 된다.
            table_list = html_co_cel.find_all('table', {'class' : 'table_list_tbidTbl table_list'})

            if table_list[0].find_all('td')[0].text.find('검색된 데이터가 없습니다.') is not -1:
                    print(table_list[0].find_all('td')[0].text)
                    break

            line_list = []
            for product_1 in table_list:
                    # td 태그는 '업무'~'투찰'까지의 정보를 가지고 있다.
                    product_list_2 = product_1.find_all('td')
                    print('-----------')
                    print('페이지 번호 : ', page_no, '공고 개수 : ', int(len(product_list_2)/10))

                    for product_2 in product_list_2:
                            print(product_2.text)
                            line_list.append(product_2.text)



            for idx in range(0, len(line_list)):
                    mod = idx % 10
                    for num in range(0, 10):
                            if mod == num:
                                    globals()['var_{}'.format(mod)].append(line_list[idx])


    df = pd.DataFrame({
            '업무' : var_0,
            '공고번호-차수' : var_1,
            '분류' : var_2,
            '공고명' : var_3,
            '공고기관' : var_4,
            '수요기관' : var_5,
            '계약방법' : var_6,
            '입력일시(입찰마감일시)' : var_7,
            '공동수급' : var_8,
            '투찰' : var_9
    })

    cur_date = datetime.datetime.today() # 현재 날짜
    cur_date = cur_date.strftime('%Y%m%d')

    file_name = 'D:\\public_crawling\\result\\나라장터_공고_{}.xlsx'.format(cur_date)
    # 엑셀 파일로 내리기
    # 최초 생성 이후 mode는 append; 새로운 시트를 추가합니다.
    if not os.path.exists(file_name):
        with pd.ExcelWriter(file_name, mode='w', engine='openpyxl') as writer:
            df.to_excel(writer, index = False, sheet_name=keyword)
    else:
        with pd.ExcelWriter(file_name, mode='a', engine='openpyxl') as writer:
            df.to_excel(writer, index = False, sheet_name=keyword)


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

    list = ["분석","빅데이터"]
    for i in list:
        print(i)
        crawler(i)


    # SMTP접속을 위한 서버, 계정 설정
    SMTP_SERVER = "smtp.gmail.com"
    SMTP_PORT = 465

    # 보내는 메일 계정
    SMTP_USER = "wook1990@gmail.com"
    SMTP_PASSWORD = 'ljjeufaklztvfabn'

    today = datetime.datetime.today()
    today = today.strftime('%Y%m%d')
    cont = today + "\t" + "나라장터 분석, 빅데이터 관련 공모 크롤링 자료 입니다."
    file_path = "D:/public_crawling/result/나라장터_공고_{}.xlsx".format(today)
    email_list = ["raphaellee@waikorea.com", "zelico@waikorea.com", "davidjung@waikorea.com"]
    
    for i in email_list:
        send_mail(i, "{0}_나라장터공고".format(today), cont, attachment=file_path)