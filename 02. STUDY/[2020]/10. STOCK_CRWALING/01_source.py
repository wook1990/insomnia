import os
import sys
import getopt
import datetime
import uuid
import pandas
import numpy
import multiprocessing
import glob
import time
import codecs
import json

from bs4 import *

# crawling function
def get_post(p_url, p_param, p_sleep_time=1):
    import time
    import urllib
    import requests

    url_full_path = p_url + '?' + urllib.parse.urlencode(p_param)
    print(url_full_path)
    headers = {
        'content-type': 'application/json, text/javascript, */*; q=0.01',
        'User-Agent': 'Mozilla/5.0 AppleWebKit/605.1.15 (KHTML, like Gecko) Version/12.0 Safari/605.1.15',
    }
    try:
        results = requests.get(url_full_path, headers=headers)
        time.sleep(p_sleep_time)
        return results
    except:
        time.sleep(p_sleep_time * 2)
        results = requests.get(url_full_path, headers=headers)
        time.sleep(p_sleep_time)
        return results

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

def get_kospi200():
    # https://finance.naver.com/sise/entryJongmok.nhn?&page=1
    url = "https://finance.naver.com/sise/entryJongmok.nhn"

    result = list()
    for page_no in range(1, 21):
        param = {
            'page': page_no
        }
        results = get_post(url, param)
        equity_table = BeautifulSoup(results.content, "html.parser").find_all('td', {"class": "ctg"})
        for td_row in equity_table:
            equity_code = str(td_row.find('a')['href']).replace('/item/main.nhn?code=', '')
            result.append(equity_code)
    return result


def collect_global(p_base_path, p_symbol, p_page_count=2, p_sleep_time=2):
    url = "http://finance.naver.com/world/worldDayListJson.nhn"
    suffix_file_name = p_symbol.split('@')[0]

    for page_no in range(1, p_page_count):
        list_price = list()
        param = {
            "symbol": p_symbol,
            "fdtc": "0",
            "page": page_no
        }
        results = get_post(url, param, p_sleep_time=p_sleep_time)
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
        df = pandas.DataFrame(list_price)
        df = df.set_index(['eod_date', 'item_code'])
        if page_no == 1:
            df.to_csv(p_base_path + 'data/input_market_' + suffix_file_name + '.csv')
        else:
            df.to_csv(p_base_path + 'data/input_market_' + suffix_file_name + '.csv', header=False, mode='a')

def get_json_from_url(i_url, i_param, i_sleep_time=1):
    import time
    import urllib
    import requests

    url_full_path = i_url + '?' + urllib.parse.urlencode(i_param)
    line_logging(url_full_path)
    headers = {
        'content-type': 'application/json, text/javascript, */*; q=0.01',
        'User-Agent': 'Mozilla/5.0 AppleWebKit/605.1.15 (KHTML, like Gecko) Version/12.0 Safari/605.1.15',
    }
    try:
        results = requests.get(url_full_path, headers=headers)
        time.sleep(i_sleep_time)
        return results
    except:
        time.sleep(i_sleep_time * 2)
        results = requests.get(url_full_path, headers=headers)
        time.sleep(i_sleep_time)
        return results

def collect_item_info(self, i_item_code):
    self.dummy()
    site_url = "http://finance.daum.net/api/quotes/A" + i_item_code
    site_param = {
        "summary": "false",
        "changeStatistics": "true",
    }
    results = get_json_from_url(site_url, site_param)
    line_logging(results.status_code, results.reason)
    json_data = json.loads(results.text)
    line_logging(json_data)