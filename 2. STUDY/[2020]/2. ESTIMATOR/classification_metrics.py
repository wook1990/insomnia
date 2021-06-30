import pandas as pd
import warnings
from sklearn import metrics
import sklearn.metrics as metric
import numpy as np
warnings.filterwarnings(action='ignore')


def assesment(data, actual_colname, score_colname, cut_off):

    data.loc[data[score_colname] >= cut_off, 'prediction'] = 1
    data.loc[data[score_colname] < cut_off, 'prediction'] = 0
    data.prediction = data.prediction.apply(int)
    y_real = np.array(data[actual_colname])
    y_pre = np.array(data["prediction"])
    y_score = np.array(data[score_colname])

    # ROC Curve를 위한 값 계산
    fpr, tpr, threshols = metrics.roc_curve(y_real, y_score)
    # AUC 계산
    auc = metrics.auc(fpr, tpr)
    # F1 score
    f1_score = metric.f1_score(y_real, y_pre)
    # Accuracy
    accuracy = metric.accuracy_score(y_real, y_pre)
    # Precision
    precision = metric.precision_score(y_real, y_pre)
    # Recall
    recall = metric.recall_score(y_real, y_pre)
    # FPR
    fpr = 1-recall

    col = {"auc": [auc], "f1_score": [f1_score],"accuracy": [accuracy], "precision": [precision], "recall": [recall],
           "fpr":[fpr]}

    as_df = pd.DataFrame(col, columns=['auc', 'f1_score', 'accuracy', 'precision','recall','fpr'])
    as_df.auc = as_df.auc.round(5)
    as_df.f1_score = as_df.f1_score.round(5)
    as_df.accuracy = as_df.accuracy.round(5)
    as_df.precision = as_df.precision.round(5)
    as_df.recall = as_df.recall.round(5)

    return as_df

if __name__ == "__main__":

    # 샘플 예제 데이터
    file_path = "D:/git_study/Haram/hospital-open-close/data/train_mod_example.csv"
    data = pd.read_csv(file_path, low_memory=False)
    data = data.set_index("inst_id")
    as_df = assesment(data, "OC", "score",0.5)
    print(as_df)
