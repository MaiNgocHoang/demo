import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.core.common import random_state
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
# from lazypredict.Supervised import LazyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from imblearn.under_sampling import NearMiss
from imblearn.under_sampling import NearMiss

df = pd.read_csv("clean_feature.csv")
print(len(df))
df["Day"] = pd.to_datetime(df['Day'])
service_dummies = pd.get_dummies(df['ServiceID'], prefix='ServiceID')
df = pd.concat([df, service_dummies], axis=1)
df = df.drop(['ServiceID'], axis=1)
print(service_dummies)
# df= df.loc[~df.duplicated()].reset_index(drop=True).copy()
# print(len(df))
# upper_limit = df["count"].mean() + 3*df["count"].std()
# lower_limit = df["count"].mean() - 3*df["count"].std()
# df = df.loc[(df["count"] < upper_limit) & (df["count"] > lower_limit)]
# print(len(df))
# df["Outcome"] = np.where(df["count"] - df["passed"] != 0, 1, 0)
# df['Datetime'] = df['Day'] + pd.to_timedelta(df['hour'], unit='h')
# df["Datetime"] = df["Datetime"].astype("int64") //  10**9
# x = df.drop(["Outcome", "Day", "hour"], axis=1)
# y = df["Outcome"]
# cc = NearMiss(version=1, sampling_strategy=1)
# x_res, y_res = cc.fit_resample(x, y)
# df = pd.merge(x_res,df, how= "inner")
# df['Timestamp'] = df.apply(lambda row: row['Day'] + pd.Timedelta(hours=row['hour']), axis=1)
# df = df.sort_values(by='Timestamp')
# df = df.drop(["Datetime", "Day"], axis=1)
# df= df.iloc[:,[8,0,6,1,2,3,4,5,7]]
#
#
#
# # 2022-10-10 10:00:00
# df = df[["Timestamp", "DoWeek", "hour", "count", "passed", "ServiceID", "Outcome"]]
#     while True:
#         start_time = input("Nhập giá trị timestamp (định dạng: yyyy-mm-dd hh:mm:ss): ")
#         try:
#             start_time = pd.to_datetime(start_time)
#             break
#         except ValueError:
#             print("Vui lòng nhập chính xác theo định dạng yêu cầu")
# train_set = df[df["Timestamp"] < start_time]
# test_set = df[(df["Timestamp"] >= start_time) & (df["Timestamp"] <= (start_time + pd.Timedelta(hours= 2)))]
# x_train = train_set.drop("Outcome", axis=1)
# y_train = train_set["Outcome"]
# x_test = test_set.drop("Outcome", axis= 1)
# y_test = test_set["Outcome"]
# print(x_test)