from sklearn import linear_model,datasets,model_selection,metrics
import matplotlib.pyplot as plt
import numpy as np

#岭回归

x,y=datasets.make_regression(n_samples=1000,n_features=1,n_targets=1,noise=30)
x_train,x_test,y_train,y_test=model_selection.train_test_split(x,y,test_size=0.3)

'''
岭回归算法是在最小二乘法的基础上加入了正则化项，为了增加模型的泛化能力和稳定性
注意：岭回归算法不能处理非线性回归问题
'''
reg=linear_model.LinearRegression()#y=x,线性回归
# reg=linear_model.Ridge()#L2,岭回归
# reg=linear_model.Lasso()#L1,Lasso回归
# reg=linear_model.ElasticNet()#弹性网络：基于L1和L2的综合考虑
# reg=linear_model.LogisticRegression()#sigmoid,逻辑斯蒂回归,是用来二分类的。
# reg=linear_model.BayesianRidge()#贝叶斯岭回归

#训练
reg.fit(x_train,y_train)
#预测
y_pred=reg.predict(x_test)
#均方误差
print(metrics.mean_squared_error(y_test,y_pred))
#绝对误差
print(metrics.mean_absolute_error(y_test,y_pred))
#R2分数
print(metrics.r2_score(y_test,y_pred))
#可解释性方差
print(metrics.explained_variance_score(y_test,y_pred))

_x=np.array([-5,5])
print(_x[:,None])
_y=reg.predict(_x[:,None])

plt.scatter(x_test,y_test)
plt.scatter(x_test,y_pred)
plt.plot(_x,_y,linewidth=2,color="red")
plt.show()