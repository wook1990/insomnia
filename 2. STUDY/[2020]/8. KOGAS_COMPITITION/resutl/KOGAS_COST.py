import pandas as pd

df_1 = pd.read_csv("G:/내 드라이브/WAI_WORK/Project_I/2020/03. 가스공사/kogas/data/kogas_tests.csv")
df_score = pd.read_csv("C:/Users/WAI/Documents/kogas/models/score_ensemble.RandomForestClassifier.csv")
df_cost = pd.read_csv("G:/내 드라이브/WAI_WORK/Project_I/2020/03. 가스공사/kogas/data/cost.csv")

rpm = [400,450,500,550,600,650,700]
fee = [46.7,58.9,71.07,87.32,105.6,127.93,152.3]
power_fee = pd.DataFrame({"fan_rpm":rpm,"min_fee":fee})

df_1.drop("target",axis=1, inplace=True)

df_t = pd.merge(df_1,pd.merge(df_score, df_cost, on="raw_id", how="left"), on ="raw_id", how="left")
df_t["e_value"] = df_t["score"] * df_t["cost"]
df_t.to_csv("C:/Users/WAI/Documents/kogas/result/result.csv",index=False)