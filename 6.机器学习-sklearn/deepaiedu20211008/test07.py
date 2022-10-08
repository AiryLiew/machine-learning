from sklearn import preprocessing
from sklearn.impute import SimpleImputer
import numpy as np

#缺失数据补全
# allowed_strategies = ["mean", "median", "most_frequent", "constant"]
imp=SimpleImputer(missing_values=np.nan,strategy="mean")
y_imp=imp.fit_transform([[np.nan,2],
                       [6,np.nan],
                       [7, 6]])
print(y_imp)
#使用训练集的填充值填充测试集缺失数据
imp.fit([[1,2],[np.nan,3],[7,6]])#4,11/3=3.667
y_test=imp.transform([[np.nan,2],[6,np.nan],[7,6]])
print(y_test)