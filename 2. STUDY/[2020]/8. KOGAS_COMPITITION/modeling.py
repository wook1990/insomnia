import pycaret
import pandas as pd

df = pd.read_csv("C:/Users/wook1/documents/data/train_set.csv",index_col=False)

rpm = [400,450,500,550,600,650,700]
fee = [46.7,58.9,71.07,87.32,105.6,127.93,152.3]
power_fee = pd.DataFrame({"fan_rpm":rpm,"min_fee":fee})

df_2 = pd.merge(df,power_fee, on="fan_rpm",how="left")
df_2["cost"] = df_2["working_time"] * df_2["min_fee"]

df_2.drop("min_fee",inplace=True,axis=1)
df_2 = df_2.round(2)
df_2.to_csv("C:/Users/wook1/documents/data/train_set_add_cost.csv",index=False)