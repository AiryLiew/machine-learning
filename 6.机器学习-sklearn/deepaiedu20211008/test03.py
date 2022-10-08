from sklearn import preprocessing
import numpy as np

#1.preprocessing.scale()
#一般二维数据标准化处理是在第0轴
x=np.array([[1.,-1.,2.],
            [2.,0.,0.],
            [0.,1.,-1.]])

# x_scale=preprocessing.scale(x)
x_scale=(x-x.mean(axis=0))/x.std(axis=0)
print(x_scale)
print(x_scale.mean(axis=0),x_scale.std(axis=0))
"""
[[ 0.         -1.22474487  1.33630621]
 [ 1.22474487  0.         -0.26726124]
 [-1.22474487  1.22474487 -1.06904497]]
"""

#2.preprocessing.StandardScaler(),保存训练集的均值和标准差，使用在测试集数据上缩放
scale=preprocessing.StandardScaler().fit(x)
#缩放训练集
train_data=scale.transform(x)
test_data=scale.transform([[-1,1,2]])

print(train_data)
print(test_data)