from sklearn.svm import SVC,SVR
import numpy as np
import matplotlib.pyplot as plt

ran=np.random.RandomState(0)
x=5*ran.rand(100,1)
y=np.sin(x).ravel()
y[::5]+=3*(0.5-ran.rand(20,1).ravel())

svr=SVR()
svr.fit(x,y)

x_test=np.linspace(0,5,100)
y_test=svr.predict(x_test[:,None])

plt.scatter(x,y)
plt.plot(x_test,y_test)
plt.show()

