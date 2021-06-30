# 정수형 : class 'int'
import sys

# 정수 표현
number1 = 123
number2 = 0
number3 = -123
print(number1, type(number1))
# 127 <class 'int'>
print(number2, type(number2))
# 0 <class 'int'>
print(number3, type(number3))
# -128 <class 'int'>

# 진수 표현
bin = 0b10  # 2진수
oct = 0o10  # 8진수
dec = 10    # 10진수
hex = 0x10  # 16진수
print(bin, type(bin))
# 2 <class 'int'>
print(oct, type(oct))
# 8 <class 'int'>
print(dec, type(dec))
# 10 <class 'int'>
print(hex, type(hex))
# 16 <class 'int'>


# 정수 최대값 최소값
max = sys.maxsize
min = -max -1
print(max, type(max))
# 9223372036854775807 <class 'int'>
print(min, type(min))
# -9223372036854775808 <class 'int'>

# 사용 가능한 메모리의 한계까지 확장될 수 있기 때문에, 큰 수를 바로 변수에 저장할 수 있음
print(max+1, type(max+1))
# 9223372036854775808 <class 'int'>
print(10**40, type(10**40))
# 10000000000000000000000000000000000000000 <class 'int'>
print(10**100, type(10**100))
# 10000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000 <class 'int'>
x = 10000000000000000000000000000000000000000000
x = x + 1
print(x)
# 10000000000000000000000000000000000000000001


# 실수형 : class 'float'
int_number = 1
float_number = 1.0
print(int_number, type(int_number))
# 1 <class 'int'>
print(float_number, type(float_number))
# 1.0 <class 'float'>

float_number1 = 123.456
float_number2 = -123.456
print(float_number1, type(float_number1))
# 123.456 <class 'float'>
print(float_number2, type(float_number2))
# -123.456 <class 'float'>

float_number3 = 123.e2
float_number4 = -123.456e-3
print(float_number3, type(float_number3))
print(float_number4, type(float_number4))

float_number5 = 10/3
float_number6 = 1.123456789012345678901234567890
print(float_number5, type(float_number5))
# 3.3333333333333335 <class 'float'>
print(float_number6, type(float_number6))
# 1.1234567890123457 <class 'float'>


# 복소수형 : class 'complex'
complex_num1 = 4.2 + 3.4j
complex_num2 = 5.6 - 7.8j

print(complex_num1, type(complex_num1))
# (4.2+3.4j) <class 'complex'>
print(complex_num2, type(complex_num2))
# (5.6-7.8j) <class 'complex'>

complex_num3 = complex_num1 + complex_num2
complex_num4 = complex_num1 - complex_num2
complex_num5 = complex_num1 * complex_num2
complex_num6 = complex_num1 / complex_num2

print(complex_num3, type(complex_num3))
# (9.8-4.4j) <class 'complex'>
print(complex_num4, type(complex_num4))
# (-1.3999999999999995+11.2j) <class 'complex'>
print(complex_num5, type(complex_num5))
# (50.04-13.719999999999999j) <class 'complex'>
print(complex_num6, type(complex_num6))
# (-0.03253796095444685+0.5618221258134489j) <class 'complex'>

print("실수부 : " ,complex_num1.real,", 허수부 : " ,complex_num1.imag)
# 실수부 :  4.2 , 허수부 :  3.4
print("실수부 : " ,complex_num2.real,", 허수부 : " ,complex_num2.imag)
# 실수부 :  5.6 , 허수부 :  -7.8

#----------------------------------------------------------------------------#
# 문자열 자료형 선언
# 1. 작은따옴표(')로 양쪽 둘러싸기
str1 = 'Hello Python!!!'
# 2. 큰따옴표(")로 양쪽 둘러싸기
str2 = "Hello Python!!!"
# 3. 작은따옴표 3개를 연속(''')으로 써서 양쪽 둘러싸기
str3 = '''Hello Python!!!'''
# 4. 큰따옴표 3개를 연속(""")으로 써서 양쪽 둘러싸기
str4 = """Hello Python!!!"""



# 문자열 안에 따옴표 자체를 포함하는 방법
str5 = 'Hello Python! "만나서 반가워요" 1234!@#$%^'
str6 = 'Hello Python! \' 만나서 반가워요 \' 1234!@#$%^'
str7 = "Hello Python! ' 만나서 반가워요 ' 1234!@#$%^"
str8 = "Hello Python! \" 만나서 반가워요 \" 1234!@#$%^"

print(str5, str6, str7, str8, sep="\n")

# 여러줄의 문자열 입력
str1 = '사과 딸기 포도 배'
str2 = '사과\n딸기\n포도\n배\n'
str3 = '''사과
딸기
포도
배
'''
str4 = """사과
딸기
포도
배
"""

print(str1, str2, str3, str4, sep="\n")
# 문자열 길이 구하기
str1 = "hello Python"
print("str1 Length: " , len(str1))
# str1 Length:  12
str2 = "안녕하세요 만나서 반갑습니다."
print("str2 Length: " , len(str2))
# str2 Length:  16
str3 = "1234%^&&*(*       "
print("str3 Length: " , len(str3))
# str3 Length:  18

# 문자열 연결과 반복
str1 = "hello Python!!"
str2 = "안녕하세요 만나서 반갑습니다."
str3 = str1 + str2
print(str3)
str4 = "^"
print(str4 * 30)

# 문자열 잘라내기
str1 = "오늘은 날씨가 정말로 화창하네요"
print(str1[0], str1[6])  # 1글자 잘라내기
# 오 가
print(str1[3:7])  # [n:m] : n부터 m-1까지 문자열을 리턴합니다.
#  날씨가
print(str1[:7])   # 앞의 숫자를 생략하면 0부터 입니다.
# 오늘은 날씨가
print(str1[7:])   # 뒤의 숫자를 생략하면 끝 글자까지 입니다.
#  정말로 화창하네요
print(str1[:])    # 모두 생략하면 전체입니다.
# 오늘은 날씨가 정말로 화창하네요
print(str1[-2:])  # 음수 값을 지정하면 뒤에서부터 카운팅합니다.
# 네요

# 앞뒤 공백 없애기
str5 = "  앞 뒤에 공백이 있습니다.  "
print('[', str5, ']',sep='')
# [  앞 뒤에 공백이 있습니다.  ]
print('[', str5.lstrip(), ']',sep='')
# [앞 뒤에 공백이 있습니다.  ]
print('[', str5.rstrip(), ']',sep='')
# [  앞 뒤에 공백이 있습니다.]
print('[', str5.strip(), ']',sep='')
# [앞 뒤에 공백이 있습니다.]

str5 = "  앞 뒤에 공백이 있습니다.  "
str6 = 'Hello Python! "만나서 반가워요" 1234!@#$%^'
print('"', str5,'"', '의 공백의 개수는 ', str5.count(' '), '개입니다.', sep='')
# "  앞 뒤에 공백이 있습니다.  "의 공백의 개수는 7개입니다.
print('"', str6, '"', '의 ll의 개수는 ', str6.count('ll'), '개입니다.', sep='')
# "Hello Python! ' 만나서 반가워요 ' 1234!@#$%^"의 ll의 개수는 1개입니다.

# 대, 소문자 변경
str1 = "hello python!!"
str2 = "HI MY NAME IS PYTHON"

print("변경전: " , str1, " 변경후 : ", str1.upper())
print("변경전: " , str2, " 변경후 : ", str2.lower())

# 문자열 찾기
str1 = "HI MY NAME IS PYTHON"
print(str1.find("NAME"))
# 6
print(str1.index("NAME"))
# 6


# 분자열 바꾸기, 나누기, 합치기
# str1 = "Hello Python"
# print("변경전: " , str1 , ", 변경후: ", str1.replace("Python","World"))
# # 변경전:  Hello Python , 변경후:  Hello World
# print("분할전: ", str1, ", 분할후: ", str1.split(" "))
# # 분할전:  Hello Python , 분할후:  ['Hello', 'Python']
#
# str2 = str1.split(" ")
# str3 = " - "
# print(str2)
# # ['Hello', 'Python']
#
# print(str3.join(str2))
# # Hello - Python

print("-" * 50)
# --------------------------------------------------
# 리스트 길이 확인
a = [1, 2, 3, 4, 5]
print("리스트의 길이: ", len(a))
# 리스트의 길이:  5

# 리스트 인덱싱
a = [1, 2, 3, 4, 5]
print(a[0:2])
# [1, 2]
print(a[:3])
# [1, 2, 3]
print(a[3:])
# [4, 5]

# 리스트애 요소 추가
a = [1, 2, 3]
a.append(4)
print(a)
# [1, 2, 3, 4]

# 지정한 위치에 요소 추가
a.insert(2,5)
print(a)

# 리스트 정렬
a = [1, 4, 6, 2, 5, 9]
befor = "정렬전: " + str(a)
a.sort()
after = ", 정렬후 : " + str(a)
print(befor+after)

# 리스트 요소값 제거
a = ["사과", " 딸기", "포도" ," 배", " 오렌지"]
before = "제거전: " + str(a)
a.remove("포도")
after = "제거후: " + str(a)

print(before + after)

a = ["사과", " 딸기", "포도" ," 배", " 오렌지"]
before = "제거전: " + str(a)
print(a.pop())
#  오렌지
after = "제거후: " + str(a)

print(before + after)
# 제거전: ['사과', ' 딸기', '포도', ' 배', ' 오렌지']제거후: ['사과', ' 딸기', '포도', ' 배']


a = ["사과", " 딸기", "포도" ," 배", " 오렌지"]
before = "제거전: " + str(a)
a.clear()
after = "제거후: " + str(a)
print(before + ", " + after)
# 제거전: ['사과', ' 딸기', '포도', ' 배', ' 오렌지'], 제거후: []




# Tuple
t1 = ( 1, 2, 3, 'a', 'b')
del t1[1]
# Traceback (most recent call last):
# TypeError: 'tuple' object doesn't support item deletion

t1[0] ='c'
# Traceback (most recent call last):
# TypeError: 'tuple' object does not support item assignment

# 인덱싱
t1 = (1, 2, 'a', 'b')
print(t1[0])
# 1
print(t1[1:])
# (2, 'a', 'b')

# 튜플 더하기
t2 = (3, 4)
print(t1 + t2)
# (1, 2, 'a', 'b', 3, 4)

# 튜플 곱하기
t2 = (3,4)
print( t2 * 3)
# (3, 4, 3, 4, 3, 4)

# 튜플 길이 구하기
t1 = (1, 2, 3, 'a', 'b' , 'c')
print(len(t1))
# 6



# set
s1 = set([1, 2, 3])
print(s1)
# {1, 2, 3}

s2 = set("hello")
print(s2)
# {'e', 'h', 'o', 'l'}

# 교집합 , 합집합, 차집합 구하기

s1 = set([1, 2, 3, 4, 5, 6])
s2 = set([4, 5, 6, 7, 8, 9])

# 교집합
print("s1 과 s2의 교집합: ", s1 & s2)
# s1 과 s2의 교집합:  {4, 5, 6}

# 합집합
print("s1 과 s2의 합집합: ", s1 | s2)
# s1 과 s2의 합집합:  {1, 2, 3, 4, 5, 6, 7, 8, 9}

# 차집합
print("s1 과 s2의 차집합: ", s1  - s2)
# s1 과 s2의 차집합:  {1, 2, 3}
print("s2 과 s1의 차집합: ", s2 - s1)
# s2 과 s1의 차집합:  {8, 9, 7}

# 함수
# 1) 값 1개 추가하기
s1 = set([1, 2, 3])
before = "추가전 : " + str(s1)
s1.add(4)
after = ", 추가후: " + str(s1)
print(before + after)
# 추가전 : {1, 2, 3}, 추가후: {1, 2, 3, 4}

# 2) 값 여러개 추가하기
s1 = set([1, 2, 3])
before = "추가전 : " + str(s1)
s1.update([4, 5, 6])
after = ", 추가후: " + str(s1)
print(before + after)
# 추가전 : {1, 2, 3}, 추가후: {1, 2, 3, 4, 5, 6}


# 3) 특정 값 제거
s1 = set([1, 2, 3])
before = "제거전 : " + str(s1)
s1.remove(3)
after = ", 제거후: " + str(s1)
print(before + after)
# 제거전 : {1, 2, 3}, 제거후: {1, 2}


# 사전형 자료형

#1. 딕셔너리 키/값 추가하기
a = {1: 'a'}
before = "추가전 : " + str(a)
a[2] = 'b'
a['name'] = 'pey'
a[3] = [1, 2, 3,]
after = ", 추가후 : " + str(a)
print(before + after)
# 추가전 : {1: 'a'}, 추가후 : {1: 'a', 2: 'b', 'name': 'pey', 3: [1, 2, 3]}

# 2. 딕셔너리 요소 삭제하기
del a[2]
print("제거후 : " +  str(a))
# 제거후 : {1: 'a', 'name': 'pey', 3: [1, 2, 3]}

# 3. 딕셔너리에서 키를 사용해 값을 얻기
grade = {'pey' : 10 , 'juliet' : 99 }
print("pey의 점수는: ", grade['pey'])
# pey의 점수는:  10
print("juliet의 점수는: ", grade['juliet'])
# juliet의 점수는:  99


# 관련 함수
a = {'name': 'pey', 'phone': '0119993323', 'birth': '1118'}

print("사전 자료형의 key: " + str(a.keys()))
# 사전 자료형의 key: dict_keys(['name', 'phone', 'birth'])

print("사전 자료형의 values: " + str(a.values()))
# 사전 자료형의 values: dict_values(['pey', '0119993323', '1118'])

print("사전 자료형의 key, value 쌍: " + str(a.items()))
# 사전 자료형의 key, value 쌍: dict_items([('name', 'pey'), ('phone', '0119993323'), ('birth', '1118')])

print("사전 자료형의 name 키의 value: ", a.get("name"))
# 사전 자료형의 name 키의 value:  pey

a.clear()
print("사전 자료형 키/값 모두 지우기: "+ str(a))
# 사전 자료형 키/값 모두 지우기: {}

