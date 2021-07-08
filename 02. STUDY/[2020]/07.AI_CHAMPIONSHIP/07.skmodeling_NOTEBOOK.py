import pandas as pd
from sklearn import *
# from pycaret.classification import *
PATH = "C:/Users/wook1/Documents/WAI/2020/02.K스타트업_빅데이터경진대회/개발/ai_dat/final/"
df = pd.read_csv(PATH + "train_final_data_x.csv")
print(df.groupby("abuse_yn").size().reset_index(name="count"))


#clf = setup(df, target="abuse_yn", ignore_features=["shop_no","ord_dt"])
#compare_models()



# sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV