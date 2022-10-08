import numpy as np
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
#核岭回归：非线性回归
rng=np.random.RandomState(0)
# rng=np.random

X=5*rng.rand(100,1)
y=np.sin(X).ravel()
y[::5] += 3 * (0.5 - rng.rand(20,1).ravel())
# kr=KernelRidge(kernel="linear")
# kr=KernelRidge(kernel="sigmoid")
#表格搜索，找出最优的核
kr=GridSearchCV(KernelRidge(),param_grid={"kernel":["rbf", "laplacian", "polynomial","sigmoid"],
                                       "alpha":[1,0.1,0.01,0.001],
                                       "gamma":np.logspace(-2,2,5)})


kr.fit(X,y)
x_=np.linspace(0,5,100)
y_=kr.predict(x_[:,None])
plt.scatter(X,y)
plt.plot(x_,y_)
plt.show()
print(kr.best_score_,kr.best_params_)