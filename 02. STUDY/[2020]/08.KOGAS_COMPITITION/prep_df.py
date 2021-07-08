import pandas as pd
import numpy as np

df = pd.read_csv("/home/data/prep/total/total_data.csv", index_col=False)
df = df.round(2)
df["hr_gr"] = df.min_time.str.replace(":\d+","",regex=True)
df = df.set_index(["min_time","hr_gr","대기온도","습도"])
df = df.fillna(method="pad")
df.reset_index(inplace=True)
fill_na_gr = lambda g:g.fillna(g.mean())

df_sub = df[["hr_gr","min_time","대기온도","습도"]].groupby("hr_gr").apply(fill_na_gr)
df_sub.reset_index(drop=True, inplace=True)
df_sub = df_sub.round(2)

df["대기온도"] = df_sub["대기온도"]
df["습도"] = df_sub["습도"]

df.drop("hr_gr", axis=1,inplace=True)
#df = df.fillna(method="pad")

df.to_csv("/home/data/prep/final/total_prep_1.csv", index=False)

