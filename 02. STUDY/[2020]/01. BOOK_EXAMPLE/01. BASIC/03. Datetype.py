from datetime import date
from datetime import time
from datetime import datetime



today = datetime.now()
today
print(today)


# 요일의 표시는
# 월요일 : 0 , 일요일 :6 으로 0~6사이의 숫자료 표시
print(today.weekday())
print(today.year)
print(today.month)
print(today.day)
print(today.hour)
print(today.minute)
print(today.second)


# 날짜를 문자열로 변환
# strftime(지정된 출력 형식으로 날짜를 문자열로 출력)
print(today.strftime("%A %d/%m/%Y"))
# >>> 'Monday 14/12/2020'


# 문자열을 날짜로 변환
# strptime :  문자열을 날짜형식으로 변환하여 출력
date_string = '2016-02-01 12:00 PM'
d1 = datetime.strptime(date_string, '%Y-%m-%d %I:%M %p')
print(d1)
# >>> 2016-02-01 12:00:00

date_string = '02/01/2016'
d2 = datetime.strptime(date_string, '%m/%d/%Y')
print(d2)
# >>> 2016-02-01 00:00:00

# 날짜 시간 계산의 차이
from datetime import timedelta

d = datetime.now()
date_string = '2/01/2016'
d2 = datetime.strptime(date_string, '%m/%d/%Y')
print(d - d2)
# >>> 1778 days, 14:58:57.601408


# timedelta는 연, 일, 주단위로 계산이 된다.
date_diff = (d - d2)/timedelta(days=1)
print('date_diff = {} days'.format(date_diff))
#>>> date_diff = 1778.624277794074 days

date_diff = (d - d2)/timedelta(weeks=1)
print('date_diff = {} weeks'.format(date_diff))
#>>> date_diff = 254.08918254201058 weeks

date_diff = (d - d2)/timedelta(days=365)
print('date_diff = {} years'.format(date_diff))
# >>> date_diff = 4.87294322683308 years



# 날짜 더하기 빼기
date_string = '2/01/2016'
d = datetime.strptime(date_string, '%m/%d/%Y')
print(d)
#>>> 2016-02-01 00:00:00
print(d + timedelta(seconds=1)) # today + one second
#>>> 2016-02-01 00:00:01
print(d + timedelta(minutes=1)) # today + one minute
#>>> 2016-02-01 00:01:00
print(d + timedelta(hours=1)) # today + one hour
#>>> 2016-02-01 01:00:00
print(d + timedelta(days=1)) # today + one day
#>>> 2016-02-02 00:00:00
print(d + timedelta(weeks=1)) # today + one week
#>>> 2016-02-08 00:00:00
print(d + timedelta(days=1)*365) # today + one year
# >>> 2017-01-31 00:00:00

# 날짜 비교

d = datetime.now()
date_string = '2/01/2016'
d2 = datetime.strptime(date_string, '%m/%d/%Y')

# 현재 날짜는 d2 로 부터 6년 이상 지났는가?
print(d < (d2 +(timedelta(days=365*6))))
#>>> True

# 선언된 날짜 d와 d2는 같은날이 아닌가?
print(d != d2)
#>>> True

# 선언된 날짜 d와 d2는 같은날인가?
print(d == d2)
#>>> True

# 시간대 설정
import pytz
# 미국의 토론토의 시간대 설정
timezone = pytz.timezone("America/Toronto")
dtz = timezone.localize(d)
print(dtz.tzinfo)
#>>> America/Toronto
print(dtz)
#>>> 2020-12-14 15:14:03.361331-05:00

# 상하이의 시간대 설정
shanghai_dt = dtz.astimezone(pytz.timezone("Asia/Shanghai"))
print(shanghai_dt)
#>>> 2019-12-23 02:18:27.386763+08:00

# 두 시간대를 비교
print((dtz - shanghai_dt)/timedelta(days=1))
#>>> 0.0



tz = pytz.all_timezones
print(tz)


# Unix time stampe
dt_now = datetime.now()
print(dt_now)
#>>> 2020-12-14 15:22:34.312976
print(dt_now.tzinfo)
#>>> None
print(dt_now.timestamp())
#>>> 1607926954.312976


# timestamp to datetime
timezone = pytz.timezone("Asia/Seoul")
utc_timestamp = 1377050861.206272
unix_ts_dt = datetime.fromtimestamp(utc_timestamp, timezone)
print(unix_ts_dt)
#>>> 2013-08-21 11:07:41.206272+09:00
print(unix_ts_dt.astimezone(pytz.timezone("America/Toronto")))
#>>> 2013-08-20 22:07:41.206272-04:00
print(unix_ts_dt.astimezone(pytz.timezone("Asia/Shanghai")))
#>>> 2013-08-21 10:07:41.206272+08:00
