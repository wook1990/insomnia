##############################################################################
#
#                      평가용 채점 스크립트 작성
#
##############################################################################

'''

Abuse Case는 매우 불균형한 형태의 레이블을 가지고 데이터이다.
불균형한 데이터의 모델의 성능을 정확히 판단할 수 있는 F1 Score 지표를 활용하여
모델의 성능을 평가한다.

F1 Score는 Precision(정밀도) Recall(재현율)의 조화평균이다.

'''


##############################################################################
#                           Ⅰ. 라이브러리 로드
##############################################################################

'''

평가 데이터와 결과 데이터의 형태는 CSV 테이블 형식의 데이터로 입력한다.
기본 제공되는 라이브러리 3가지를 활용하여 평가 스크립트를 작성한다.
    1) sys: 터미널에서 코드 실행시에 입력받는 테스트셋 정답파일명과 채점할 결과
            파일명을 인자로 가져올 수 있도록 sys 라이브러리를 임포트합니다.
    2) numpy: rmse 계산에 필요한 수치연산을 위해 numpy 라이브러리를 임포트합니다.
    3) pandas: .csv 형식으로 저장된 테이블을 손쉽게 가져와서 계산할 수 있도록
               pandas 라이브러리를 임포트합니다.
    

'''
import sys
import pandas as pd
import numpy as np

##############################################################################


##############################################################################
#                       Ⅱ. 인자 정의 및 데이터 불러오기
##############################################################################


'''

터미널에서 실행되는 명령어는 다음과 같다.

    $ python 채점스크립트.py 테스트셋정답 채점할결과
    

'''

# local test code
answer_df = pd.read_csv("C:/Users/wook1/Documents/WAI/2020/02.K스타트업_빅데이터경진대회/개발/answer_df.csv")
predict_df = pd.read_csv("C:/Users/wook1/Documents/WAI/2020/02.K스타트업_빅데이터경진대회/개발/ai_dat/final/predict_data.csv")



# 정답 데이터를 pandas로 불러와 answer_df로 할당
answer_df = pd.read_csv(sys.argv[1])
# 채점할결과를 pandas로 불러와 predict_df로 할당
predict_df = pd.read_csv(sys.argv[2])



##############################################################################
#                                Ⅲ. 전처리
##############################################################################

'''

읽어들인 정답지와 예측결과의 target값인 abuse_yn을 추출하여 
numpy array로 변경하는 코드이다.

정답지 기간 정의 : 2020-09-09 ~ 2020-09-15

'''
# 정답지와 채점파일 정렬
answer_df = answer_df.sort_values(["shop_no","ord_dt"])
predict_df = predict_df.sort_values(["shop_no","ord_dt"])

y_true = np.array(answer_df["abuse_yn"])
y_pred = np.array(predict_df["abuse_yn"])

##############################################################################



##############################################################################
#                             Ⅳ. 평가용 함수 정의
##############################################################################


'''
평가지표인 F1 Score는 Precision(정밀도) 와 Recall(재현율)의 조화 평균으로
Precision, Recall, F1 socre를 구하는 사용자 정의함수를 선언합니다.
정답(y_true) , 예측값(y_pred)를 인자로 입력받아 score로 반환됩니다.
positive label은 기본적으로 1로 선언하여, 각각의 TP, FP, FN의 값을 구할 수 있도록
활용합니다.

'''


def _precision(y_true, y_pred, positive_label=1):
    # TP, FP 초기화
    true_positive = 0
    false_positive = 0
    for (idx, p) in enumerate(y_pred):
        # TP CASE
        if p == positive_label and y_true[idx] == positive_label:
            true_positive +=1
        # FP CASE
        elif p == positive_label and y_true[idx] != positive_label:
            false_positive += 1
    return true_positive / (true_positive + false_positive)


def _recall(y_true, y_pred, positive_labele=1):
    # TP, FN 초기화
    true_positive = 0
    false_negative = 0
    for idx, p in enumerate(y_pred):
        # TP CASE
        if p == positive_labele and y_true[idx] == positive_labele:
            true_positive +=1
        # FN CASE
        elif p != positive_labele and y_true[idx] == positive_labele:
            false_negative += 1
    return true_positive / (true_positive + false_negative)

def _f1_score(y_true, y_pred, positive_label=1):
    precision = _precision(y_true,y_pred, positive_label)
    recall = _recall(y_true, y_pred, positive_label)
    return 2.0 / (1/precision + 1/recall)

###############################################################################

##############################################################################
#                             Ⅴ. 평가 수행
##############################################################################

score = _f1_score(y_true,y_pred, positive_label=1)
print('score:', round(score,5))
