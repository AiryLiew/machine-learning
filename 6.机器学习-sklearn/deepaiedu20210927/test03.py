from sklearn.datasets import make_regression
import matplotlib.pyplot as plt
'回归模型的演示'

x,y,k=make_regression(n_samples=100,n_features=1,noise=30,coef=True)
plt.scatter(x,y)
plt.plot(x,x*k,color="red",linewidth=2)
plt.show()
