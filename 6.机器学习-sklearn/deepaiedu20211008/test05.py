from sklearn import preprocessing
import numpy as np

# 4.preprocessing.MaxAbsScaler,最大绝对值缩放，好处是不移动中心点

x=np.array([[1.,-1.,2.],
            [2.,0.,0.],
            [0.,1.,-1.]])
scale=preprocessing.MaxAbsScaler()
# x_scale=scale.fit_transform(x)
x_scale=x/np.max(np.abs(x),axis=0)
print(x_scale)

"""
[[ 0.5 -1.   1. ]
 [ 1.   0.   0. ]
 [ 0.   1.  -0.5]]
"""