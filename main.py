import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from fontTools.misc.classifyTools import Classifier
from imblearn.under_sampling import RandomUnderSampler
from sklearn.feature_selection import mutual_info_classif
from  sklearn.ensemble import  ExtraTreesClassifier


df = pd.read_csv("clean_feature.csv")
df["Outcome"] = np.where( df["passed"] - df["count"] != 0, 1, 0)
# df= df.drop('Day', axis = 1)
# # sns.heatmap(df.corr(method='kendall'), annot=True)
# # plt.show()


x = df.drop(["Outcome", 'Day'], axis=1)
y = df["Outcome"]
model = ExtraTreesClassifier()
model.fit(x,y)
print(model.feature_importances_)
feat_importances = pd.Series(model.feature_importances_, index= x.columns)
print(feat_importances)

#
# plt.figure(figsize=(10, 6))
# sns.countplot(data=df, x='ServiceID', hue='Outcome')
# plt.xlabel('ServiceID')
# plt.ylabel('Count of Outcome')
# plt.title('Distribution of Outcome for each ServiceID')
# plt.legend(title='Outcome')
# plt.show()

# df['Day'] = pd.to_datetime(df['Day'])
# df['Timestamp'] = df.apply(lambda row: row['Day'] + pd.Timedelta(hours=row['hour']), axis=1)
#
#
# x = df.drop(["Outcome", "Day"], axis= 1)
# y = df["Outcome"]
#
# rus = RandomUnderSampler(random_state=0)
# x_res, y_res = rus.fit_resample(x, y)
# df1 = x_res
# df1["Outcome"] = y_res
# df1_sorted = df1.sort_values(by='Timestamp').reset_index(drop=True)
#
#

#