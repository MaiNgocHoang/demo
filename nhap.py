import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split


df = pd.read_csv("clean_feature.csv")
# Sử dụng One-Hot Encoding cho 'ServiceID' để tạo các cột nhị phân
# service_dummies = pd.get_dummies(df['ServiceID'], prefix='ServiceID')
# df = pd.concat([df, service_dummies], axis=1)
#
# service_columns = [col for col in df.columns if col.startswith("ServiceID_")]
# df.loc[df['count'] == df['passed'], service_columns] = False
# df['Outcome'] = df[service_columns].apply(lambda x: ''.join(['1' if v else '0' for v in x]), axis=1)
# print(df["Outcome"].dtypes)
# df = df.drop(service_columns, axis=1)
#
# # def bitwise_or_outcome(series):
# #     # Bắt đầu với chuỗi đầu tiên
# #     result = series.iloc[0]
# #     for outcome in series[1:]:
# #         # Thực hiện phép or bit cho từng ký tự
# #         result = ''.join(['1' if r == '1' or o == '1' else '0' for r, o in zip(result, outcome)])
# #     return result
# #
# # # Nhóm dữ liệu theo các cột 'Day', 'DoWeek', 'hour' và thực hiện phép or bit trên cột 'Outcome'
# # grouped_df = df.groupby(['Day', 'DoWeek', 'hour', 'period']).agg({'Outcome': bitwise_or_outcome}).reset_index()
# #
#
#
#
#
# # Giả sử bạn đã đọc dữ liệu vào DataFrame tên là df
# # df = pd.read_csv("your_data.csv")
#
# # Tách cột Outcome thành các cột nhị phân (mỗi dịch vụ một cột)
# outcome_cols = [f'Service_{i}' for i in range(len(df['Outcome'][0]))]
# outcome_df = df['Outcome'].apply(lambda x: pd.Series(list(x), index=outcome_cols)).astype(int)
#
# # Ghép outcome_df vào df ban đầu
# df = pd.concat([df, outcome_df], axis=1)
#
# # In thử để kiểm tra kết quả
# print(df.head())
#
#
# # Đặc trưng (features) và nhãn (labels)
# features = df[['Day', 'DoWeek', 'hour', 'count', 'passed', 'period', 'data', 'ServiceID']]
# labels = outcome_df
# print(labels)
# # # # Chuyển đổi `Day` sang dạng số, nếu cần thiết (vì hiện tại nó là chuỗi)
# # # features['Day'] = pd.to_datetime(features['Day']).map(pd.Timestamp.toordinal)
# #
# # # Chia dữ liệu thành tập huấn luyện và kiểm tra
# X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
# #
# # Khởi tạo và huấn luyện mô hình
# model = RandomForestClassifier(n_estimators=100, random_state=42)
# model.fit(X_train, y_train)
#
# # Dự đoán trên tập kiểm tra
# y_pred = model.predict(X_test)
#
# # Đánh giá mô hình
# print(classification_report(y_test, y_pred, target_names=outcome_cols))
a = df["ServiceID"].unique()

print(a.max())