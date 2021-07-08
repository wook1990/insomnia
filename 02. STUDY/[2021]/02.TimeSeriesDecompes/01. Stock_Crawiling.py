# 필수 라이브러리 임포트
import numpy
import pandas
import scipy.stats
import statsmodels.api as sm
import multiprocessing
import glob
import json

from sklearn import *
from bs4 import *
from selenium import *

DATA_PATH = "/data/2.TimeSeriesDecompose\\"

# ========================================================================================================================================================================
# 함수 모음
# ========================================================================================================================================================================
# 로깅 함수
def line_logging(*messages):
    import datetime
    import sys
    today = datetime.datetime.today()
    log_time = today.strftime('[%Y/%m/%d %H:%M:%S]')
    log = list()
    for message in messages:
        log.append(str(message))
    print(log_time + ':[' + ' '.join(log) + ']', flush=True)


# crawling function

def get_post(p_url, p_param, p_sleep_time=2, p_flag_view_url=True):
    '''
    크롤링 함수
      - 주어진 사이트 URL에 GET방식으로 데이터를 요청하여
        웹페이지의 HTML 코드를 반환하는 함수

    파라미터
      - p_url <String> : 데이터를 제공하는 사이트 URL 주소를 받는 파라미터
      - p_parm <dictionary> : URL에 GET방식으로 데이터를 요청할 Query String 을 받는 파라미터
      - p_sleep_time <int, float> : URL 요청시 호출을 대기하기 위한 Delay time을 받는 파라미터
      - p_flag_viw_url <boolean> :
    '''

    # 크롤링 라이브러리 임포트
    import time
    import urllib
    import requests

    # Dictionary 형식으로 전달되는 파라미터를 GET방식의 Query String으로 변환
    url_full_path = p_url + '?' + urllib.parse.urlencode(p_param)

    if p_flag_view_url:
        line_logging(url_full_path)

    # 크롤링할 사이트에서 비정상 접근으로 일시적으로 네트워크를 제한하는 상황을 방지하기 위해
    # 유저정보인 User-Agent를 지정하여, 해당 사이트의 접근이 사람에 의한 접근이라는 것을 명시해주는
    # 정보 입력
    headers = {
        'content-type': 'application/json, text/javascript, */*; q=0.01',
        'User-Agent': 'Mozilla/5.0 AppleWebKit/605.1.15 (KHTML, like Gecko) Version/12.0 Safari/605.1.15',
        'referer': 'http://finance.daum.net/domestic/exchange/COMMODITY-%2FCLc1'
    }

    # 선언된 URL을 GET 방식으로 데이터 요청하여 사이트화면상 보여지는 정보의 HTML 코드를 반환
    # URL 요청시 접속 지연으로 서버의 데이터가 갱신이 늦어지는 경우를 예외 케이스로 정의하여
    # 요청된 URL의 정보가 표현이 된후 갱신 될 수 있도록 Delay를 주어 갱신되는 페이지의 HTML 코드 반환
    try:
        results = requests.get(url_full_path, headers=headers)
        time.sleep(p_sleep_time)
        return results
    except:
        time.sleep(p_sleep_time * 2)
        results = requests.get(url_full_path, headers=headers)
        time.sleep(p_sleep_time)
        return results


# 종목별 시세 수집
def collect_equity(p_equity_code, p_page_no):
    '''
    종목별 시세 수집 함수
      - 종목코드, 수집할 데이터의 페이지 수를 정의하여 일별 시세 수집

    파라미터
      - p_equity_code <String> : 종목코드를 선언하는 파라미터
      - p_page_no <int> : 데이터를 조회할 페이지 번호를 선언하는 파라미터

    '''

    # https://finance.naver.com/item/sise_day.nhn?code=005930&page=2

    # 네이버 금융의 일별 시세 URL 선언
    url = "https://finance.naver.com/item/sise_day.nhn"

    # 시세 정보를 담는 리스트 선언
    list_price = list()
    # Query String의 조건을 전달하는 파라미터 선언(Dictionary Type)
    # 종목코드, 수집할 페이지 수
    param = {
        'code': p_equity_code,
        'page': p_page_no
    }

    # 선언된 URL과 파라미터를 Query String으로 변환하여
    # get_post함수를 사용하여 페이지 갱신을 요청하여 표현된 결과의 HTML 코드 반환
    results = get_post(url, param)

    # 텍스트 형식인 HTML 코드로 반환된 페이지 결과를 BeautifulSoup를 사용하여,
    # HTML 코드형식으로 Parsing 하여 구조화시킨후 일별 시세가 포함되어 있는
    # HTML 태그를 찾아 데이터를 수집
    # 일별 시세가 담겨있는 table 태그의 HTML 구조 반환
    price_table = BeautifulSoup(results.content, "html.parser").find_all('table')[0]
    # 반환된 table 태그의 HTML 구조를 파싱하여 테이블내 데이터가 존재하는 각 row를 담는
    # tr(table row) 태그의 전체 HTML 구조를 반환
    price_trs = BeautifulSoup(str(price_table), "html.parser").find_all('tr')

    # 일별 시세의 데이터을 담고 있는 tr태그에서 각각의 데이터를 포함한 td(table data) 태그를
    # 리스트 형식으로 반환하여 각 row 단위로 반복처리하여 데이터 추출
    for price_tr in price_trs:
        row = BeautifulSoup(str(price_tr), "html.parser").find_all('td')
        # table에서 추출한 각 row에는 시세정보외 구분선을 나타내는 tr 태그도 존재하므로
        # tr 태그에 포함된 td의 개수가 3개 초과인 경우에만 td에 포함된 데이터를 추출
        if len(row) > 3:
            # table의 각 row는 날짜, 종가, 전일비, 시가, 고가, 저가, 거래량을 컬럼값으로 구성
            # 추출된 각 row의 데이터의 정보를 Dictionry Type으로 선언하여 list_price 리스트에
            # append하여 데이터를 수집
            # 표현된 데이터는 String형식으로 수집되어 문자열 처리(공백제거, 구분자제거)를 하여
            # int Type으로 변환하여 데이터 추출
            # 선언된 Dictionary Type에 맞지 않는 데이터의 형태인 경우 예외처리하여
            # 종목코드, 페이지 번호, 데이터가 담긴 row를 출력하여 결과 확인
            # 수집된 데이터로 구성된 list_price를 pandas DataFrame으로 변환하여 결과 반환
            try:
                list_price.append({
                    'eod_date': row[0].text.strip().replace('.', ''),
                    'item_code': p_equity_code,
                    'price_close': int(row[1].text.strip().replace(',', '')),
                    'price_open': int(row[3].text.strip().replace(',', '')),
                    'price_high': int(row[4].text.strip().replace(',', '')),
                    'price_low': int(row[5].text.strip().replace(',', '')),
                    'trade_amount': int(row[6].text.strip().replace(',', '')),
                    'diff': float(row[2].text.strip().replace(',', '')),
                })
            except:
                line_logging(p_equity_code, p_page_no, row)

    return pandas.DataFrame(list_price)


# 종목별 수급 수집
def collect_group(p_equity_code, p_page_no):
    '''
    종목별 수급 수집 함수
      - 종목별 기관/외국인의 보유주수, 보유율, 매매량 데이터를 크롤링하는 함수

    파라미터
      - p_equity_code <String> : 종목코드를 선언하는 파라미터
      - p_page_no <int> : 데이터를 조회할 페이지 번호를 선언하는 파라미터

    '''

    # 기관/외국인 투자정보가 표현되는 URL 주소 선언
    url = "https://finance.naver.com/item/frgn.nhn"

    # 결과를 담을 비어있는 리스트를 선언
    list_price = list()
    # URL의 Query String에 전달할 파라미터 선언
    # 종목코드, 페이지 번호
    param = {
        'code': p_equity_code,
        'page': p_page_no
    }

    # 선언된 URL과 파라미터를 Query String으로 변환하여
    # get_post함수를 사용하여 페이지 갱신을 요청하여 표현된 결과의 HTML 코드 반환
    results = get_post(url, param)

    # 텍스트 형식인 HTML 코드로 반환된 페이지 결과를 BeautifulSoup를 사용하여 HTML 코드 형식으로 Parsing
    # 구조화된 HTML 코드에 수집하고자하는 정보가 담겨있는 HTML 태그를 찾아 데이터를 수집
    # 수급 데이터가 표현되는 영역의 table 태그를 찾기 위해
    # table 태그의 속성인 class의 속성값이 type2인 영역을 파싱하여 HTML 형식으로 반환
    price_table = BeautifulSoup(results.content, "html.parser").find_all('table', {"class": "type2"})[1]

    # 반환된 table 영역의 데이터가 포함된 영역인 tr태그의 정보를 파싱하여 HTML 형식으로 반환
    price_trs = BeautifulSoup(str(price_table), "html.parser").find_all('tr')

    # 종목별 수급 데이터을 담고 있는 tr태그에서 각각의 데이터를 포함한 td(table data) 태그를
    # 리스트 형식으로 반환하여 각 row 단위로 반복처리하여 데이터 추출
    for price_tr in price_trs:
        row = BeautifulSoup(str(price_tr), "html.parser").find_all('td')
        # table에서 추출한 각 row에는 날짜, 종가, 전일비, 등락률, 거래량, 기관 순매매량,
        # 외국인 순매매량, 외국인 보유주수, 외국인 보유율의 9개의 컬럼을 포함하고 있어
        # row에 포함된 td 태그의 개수가 9개이 상인경우의 데이터만 추출하여
        # 각 row의 컬럼값을 Dictionary Type으로 데이터를 저장하여
        # 앞에 선언해둔 list_price 리스트에 append 하여 데이터 추출
        # 문자열 처리(공백제거, 문자열 변경)하여 int형식으로 데이터 추출하여
        # pandas DataFrame 형식의 데이터 타입으로 수집된 결과 반환
        if len(row) > 8:
            try:
                list_price.append({
                    'eod_date': row[0].text.strip().replace('.', ''),
                    'item_code': p_equity_code,
                    'foreign_hold_count': int(row[7].text.strip().replace(',', '').replace('%', '').replace('+', '')),
                    'foreign_holds_ratio': float(
                        row[8].text.strip().replace(',', '').replace('%', '').replace('+', '')),
                    'foreign_trade_count': int(row[6].text.strip().replace(',', '').replace('%', '').replace('+', '')),
                    'organization_count': int(row[5].text.strip().replace(',', '').replace('%', '').replace('+', '')),
                })
            except:
                line_logging(p_equity_code, p_page_no, row)
    return pandas.DataFrame(list_price)


# 금리 정보 수집
def collect_bond(p_market, page_no=1, p_sleep_time=2):
    '''
    금리 정보 수집 함수
      -
    '''
    url = "http://finance.daum.net/api/domestic/exchanges/BOND-/KRGUCORP=KQ/days"
    list_price = list()
    param = {
        "symbolCode": p_market,
        "page": page_no,
        "perPage": 30,
        "fieldName": "changeRate",
        "order": "desc",
        "pagination": "true",
    }
    results = get_post(url, param, p_sleep_time=p_sleep_time)
    # {"data":[{"symbolCode":"COMMODITY-/CLc1","name":null,"date":"2020-12-16","country":null,"region":null,"tradePrice":47.82,"tradeDate":"2020-12-16","changeRate":0.004199916,"changePrice":0.2},
    for row in json.loads(results.text)['data']:
        list_price.append({
            'eod_date': str(row['date']).replace('-', ''),
            'item_code': row['symbolCode'],
            'price_close': row['tradePrice'],
            'diff': row['changePrice'],
            'rate': row['changeRate'],
        })
    return pandas.DataFrame(list_price)


# 해외 지수 수집
def collect_global(p_market, page_no=1, p_sleep_time=2):
    '''
    해외 지수 수집 함수
      - 네이버 금융에서 제공하는 해외 지수 데이터를 수집하는 함수
        Page내부에 해외지수정보를 데이터를 JSON형식으로 전달하여 비동기식처리가되어 갱신되는
        페이지의 정보 수집(동적웹페이지)

    파라미터
      - p_market <String> : 해외지수 종목코드을 선언하는 파라미터
      - page_no <int> : 조회할 페이지 번호를 선언하는 파라미터
      - p_sleep_time <int> : 페이지 갱신시 서버요청을 대기하는 Delay를 선언하는 파라미터
    '''

    # 네이버 금융의 해외 주가 페이지 내에서 비동기식 처리모델로 데이터가 갱신되도록 구성
    # JSON형식으로 데이터가 전달되어 일별 주가 정보 영역만 데이터가 갱신되는 방식으로 구성되어 있어
    # 페이지가 갱신되는 영역의 URL을 추출하여 선언

    url = "http://finance.naver.com/world/worldDayListJson.nhn"

    # 결과를 담을 비어있는 리스트를 선언
    list_price = list()

    # 조회 조건의 Query String에 전달할 파라미터 선언
    # 해외 지수 종목 코드, 페이지 번호
    param = {
        "symbol": p_market,
        "fdtc": "0",
        "page": page_no
    }

    # 선언된 URL과 파라미터를 Query String으로 변환하여
    # get_post함수를 사용하여 페이지 갱신을 요청하여 표현된 결과의 HTML 코드 반환
    results = get_post(url, param, p_sleep_time=p_sleep_time)

    # 반환된 결과를 JSON으로 변환하여 각 row에 해당하는 데이터를
    # Dictionary 객체에 담아 list_price에 추가하여
    # pandas DataFrame 객체로 최종 결과를 반환
    for row in json.loads(results.text):
        '''
        {
        "xymd":"20171212", # 거래일
        "symb":"DJI@DJI",
        "open":24452.96, # 시가
        "high":24552.97, # 고가
        "low":24443.83, # 저가
        "clos":24504.8, # 종가
        "gvol":342223357, # 거래량
        "diff":118.77, # 증감
        "rate":0.49, # 등락율
        }
        '''
        list_price.append({
            'eod_date': row['xymd'],
            'item_code': row['symb'],
            'price_open': row['open'],
            'price_high': row['high'],
            'price_low': row['low'],
            'price_close': row['clos'],
            'trade_amount': row['gvol'],
            'diff': row['diff'],
            'rate': row['rate'],
        })
    return pandas.DataFrame(list_price)


def collect_MarketCap(rank, option):
    '''
    시가총액 순위별 데이터 추출 함수
      - 시가총액 상위 그룹에 해당하는 전체 종목의 시세, 수급의 정보를 수집하는 함수
        option 조건에 따라 코스피, 코스닥에 해당하는 종목의 정보를 수집

    파라미터
      - rank <int> : 시가총액 순위를 선언하는 파라미터
      - option <int> : 코스피, 코스닥 시가총액을 정의하는 파라미터
                       코스피 : 0, 코스닥 : 1 로 선언

    '''

    # collect_MarketCap내 필요 라이브러리 임포트
    from bs4 import BeautifulSoup
    import requests
    import datetime
    import math

    # 조회 날짜
    today = datetime.datetime.today().date()

    # 시가총액 데이터가 있는 URL 선언
    # option 값을 지정하여 코스피, 코스닥 탭을 구분하여 조회하도록 URL 구분
    url = "https://finance.naver.com/sise/sise_market_sum.nhn?sosok={0}&page=".format(option)

    # 페이지당 50개의 종목이 표현되며
    # 조회하고싶은 순위의 전체 종목의 페이지를 선언
    page_num = math.ceil(rank / 50) + 1

    # 결과를 담을 DataFrame 선언
    marketCap_df = pandas.DataFrame()

    # 페이지 별로 갱신되는 종목 순위별 시가총액 정보 수집
    for i in range(1, page_num):

        # Query String 선언
        url_path = url + str(i)
        line_logging(url_path)

        # 이상 접근 방지를 위한 User-Agent 선언
        headers = {
            'content-type': 'application/json, text/javascript, */*; q=0.01',
            'User-Agent': 'Mozilla/5.0 AppleWebKit/605.1.15 (KHTML, like Gecko) Version/12.0 Safari/605.1.15',
            'referer': 'http://finance.daum.net/domestic/exchange/COMMODITY-%2FCLc1'
        }

        # Query String으로 완성된 URL의 웹페이지의 HTML 코드를 반환
        req = requests.get(url_path, headers=headers)
        # 반환된 HTML코드를 BeautifulSoup를 사용하여 파싱해 HTML 구조화
        soup = BeautifulSoup(req.text, "html.parser")

        # 구조화된 HTML 코드에서 종목별 시가총액관련 정보가 표현되는 영역 선택
        html_obj = soup.select_one("#contentarea > div.box_type_l")

        # 표시된 영역의 table의 header 와 data 영역으로 구분하여 데이터 추출
        header = [item.get_text().strip() for item in html_obj.select('thead th')][1:-1]
        data = [item.get_text().strip() for item in html_obj.find_all(
            lambda x: (x.name == 'a' and 'tltle' in x.get('class', [])) or (
                        x.name == 'td' and 'number' in x.get('class', [])))]

        # 종목명의 a 태그안의 종목코드별 시세 정보 사이트 주소를 파싱하고 분해하여 종목코드 추출
        code_href = [values["href"].split("=") for values in
                     html_obj.find_all(lambda x: x.name == 'a' and 'tltle' in x.get('class', 'tltle'))][1:-12]

        # 추출된 종목코드 리스트 생성
        code_ex = [x[1] for x in code_href]

        code = list()
        for x in code_ex:
            if x not in code:
                code.append(x)

        # 리스트형식으로 추출된 data를  numpy array로 변환
        data = numpy.array(data)

        # 변환된 numpy array형식의 data를 종목수, 컬럼수로 사이즈 재정의
        data.resize(len(code), len(header))

        # 시가 총액 데이터 프레임 생성
        marketCap = pandas.DataFrame(data=data, columns=header)
        marketCap["종목코드"] = code
        marketCap["수집일자"] = today
        marketCap = marketCap[['수집일자', '종목코드', '종목명', '현재가', '전일비', '등락률', '액면가', '시가총액', '상장주식수', '외국인비율', '거래량',
                               'PER', 'ROE']]
        marketCap = marketCap[['수집일자', '종목코드', '종목명', '현재가', '전일비', '등락률', '액면가', '시가총액', '상장주식수', '외국인비율', '거래량',
                               'PER', 'ROE']]

        # 초기 선언된 결과 DataFrame에 페이별로 생성되는 종목별 시가총액 DataFrame 추가
        marketCap_df = pandas.concat([marketCap_df, marketCap])

    # 페이지당 50개씩 처리되는 종목의 인덱스를 초기화하여 전체 인덱스 정렬
    marketCap_df.reset_index(drop=True, inplace=True)

    # 컬럼명 변경
    marketCap_df.columns = ['eod_date', 'item_code', 'item_name', 'price_present', 'previous_ratio', 'flucuation_rate',
                            'face_value', 'market_cap',
                            'public_equity_cnt', 'foreigner_ratio', 'volume', 'PER', 'ROE']

    # 결과 파일 생성
    marketCap_df.to_csv(DATA_PATH + "d1_101_price_total_rank.csv", index=False)

    # 시가총액 DataFrame 결과 반환
    return marketCap_df
# ========================================================================================================================================================================




if __name__ == "__main__":

    # 앞의 구문에서 선언된 함수들을 사용하여 시가총액 상위 50개 종목에 대하여
    # 시세, 수급 정보 수집

    # 시가총액 상위 50개 종목의 종목코드 추출
    #code = list(collect_MarketCap(50, 0)["item_code"])
    code = ["005930"]
    # 추출된 종목 코드별로 50개의 시세 및 수급 정보 수집
    for item_code in code:
        code_equity = item_code
        print(code_equity)
        # 시세 정보를 담을 DataFrame 선언
        df_price = pandas.DataFrame()
        # 50개 종목의 일별 시세를 200 페이지까지 데이터 수집
        for page_no in range(200):
            df_price = pandas.concat([df_price, collect_equity(code_equity, page_no + 1)])
        #print(df_price.shape)
        #print(df_price.head())
        df_price.to_csv(DATA_PATH + 'd1_101_price_' + code_equity + '.csv', index=None)

        # 일별 수급 수집
        df_group = pandas.DataFrame()
        # 50개 종목의 일별 수급 정보를 100페이지가지 데이터 수집
        for page_no in range(100):
            df_group = pandas.concat([df_group, collect_group(code_equity, page_no + 1)])
        #print(df_group.shape)
        #print(df_group.head())
        df_group.to_csv(DATA_PATH + 'd1_102_group_' + code_equity + '.csv', index=None)
        line_logging(code_equity, " is done")

        # 시장 구분 코드
        code_market = 'DJI@DJI'
        name_market = code_market.replace('@', '_')
        # 일별 시장 지수 수집
        df_market = pandas.DataFrame()
        for page_no in range(200):
            df_market = pandas.concat([df_market, collect_global(code_market, page_no + 1)])
        #print(df_market.shape)
        #print(df_market.head())
        df_market.to_csv(DATA_PATH + 'd1_103_market_' + name_market + '.csv', index=None)