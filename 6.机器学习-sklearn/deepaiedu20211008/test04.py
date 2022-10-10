from sklearn import preprocessing
import numpy as np

# 3.preprocessing.MinMaxScaler()
x=np.array([[1.,-1.,2.],
            [2.,0.,0.],
            [0.,1.,-1.]])
min_max_scale=preprocessing.MinMaxScaler()
x_train=min_max_scale.fit_transform(x)
# x_train=(x-x.min(axis=0))/(x.max(axis=0)-x.min(axis=0))
print(x_train)
"""
[[0.5        0.         1.        ]
 [1.         0.5        0.33333333]
 [0.         1.         0.        ]]
"""

x_test=np.array([[-3,-1,4]])
test_scale=min_max_scale.transform(x_test)
print(test_scale)