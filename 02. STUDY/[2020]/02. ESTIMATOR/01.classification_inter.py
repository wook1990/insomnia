import pandas as pd
import warnings
from sklearn import metrics
import sklearn.metrics as metric
import matplotlib.pyplot as plt
warnings.filterwarnings(action='ignore')


# 테스트 데이터 셋
file_path = "D:/git_study/Haram/hospital-open-close/data/train_mod_example.csv"
data = pd.read_csv(file_path, low_memory=False)
data = data.set_index("inst_id")
actual = data["OC"]
score = data["score"]

# 예측확률의 기준값을 지정하여 True False 분류
data.loc[data.score >= 0.5, 'prediction'] = 1
data.loc[data.score < 0.5, 'prediction'] = 0
# 분류예측 결과
prediction = data.prediction
# Confusion Matrix (혼동행렬)
# Sklearn 패키지내 Metrics 모듈의 confusion_matrix 함수 사용
"""
            Predicted True  Predicted False
Actual True    [0,0] TP         [0,1] FN
Actual False   [1,0] FP         [1,1] TN
"""

confusion_matix = metric.confusion_matrix(actual, prediction, labels=[1,0])
"""
            Predicted True  Predicted False
Actual True    155               131
Actual False    7                 8 
"""
TP = confusion_matix[0,0]
FN = confusion_matix[0,1]
FP = confusion_matix[1,0]
TN = confusion_matix[1,1]
# 1. Precision(정밀도)
# TP/(TP+FP) : True 라고 예측한것중 실제값이 True인것 (내가 푼 문제중 맞춘 정답의 개수)
Precision = TP/(TP+FP)
print(Precision.round(5))
# sklearn의 서브패키지 metrics지원 함수 사용
Precision = metric.precision_score(actual, prediction)
print(Precision.round(5))

# 2.Recall(Sesitivity) : 실제값이 True인것중 True라고 예측한것(전체중에 내가 맞춘 문제 수)
# TP/(TP+FN)
Recall = TP/(TP+FN)
print(Recall.round(5))
# sklearn의 서브패키지 metrics지원 함수 사용
Recall = metric.recall_score(actual, prediction)
print(Recall.round(5))

# 3. Accuracy(정확도) : 전체중에 올바르게 예측한 것(전체 경우의 수중에서 True로 예측한 비율)
# TP+TN/(TP+TN+FN+FP)
Accuracy = (TP+TN)/(TP+TN+FN+FP)
print(Accuracy.round(5))
# sklearn의 서브패키지 metrics지원 함수 사용
Accuracy = metric.accuracy_score(actual, prediction)
print(Accuracy.round(5))

# 4.F1-Score: 정밀도와 재현율의 가중 조화 평균으로 데이터가 불균형 구조일때
#             모델의 성능평가
# F1-Score = 2/((1/Recall + 1/Precision)) = 2 * ((precision * recall)/(precision + recall))

f1_score =  2 * ((Precision * Recall)/(Precision + Recall))
print(f1_score.round(5))
# sklearn의 서브패키지 metrics지원 함수 사용
f1_score = metric.f1_score(actual, prediction)
print(f1_score.round(5))


#5. FPR(위양성율 : False Positive Rate)
# Fall-out, 1-특이도, 진음성율이라고도 하며 실제 False인 것들 중에
# 모델이 True라고 분류한 것의 비율
# FPR = FP/(FP+TN)
fpr = FP/(FP+TN)
print(fpr)

# ROC Curve & AUC
# 0~1사이의 값을 가지는 위양성율과 민감도는 무한대의 혼동행렬을 가지게 되므로
# 기준값 변화에 다른 위양성율과 재현율의 변화를 시각화한 그래프
fpr, tpr, thresholds = metrics.roc_curve(actual, score)
print("FPR : ",fpr,", TPR : ", tpr,", Thresholds : ", thresholds)
auc = metric.auc(fpr,tpr)
print(auc)

plt.plot(fpr,tpr, 'o-')
plt.plot([0,1],[0,1],'k--')
plt.plot([fpr], [tpr], 'ro', ms=10)



