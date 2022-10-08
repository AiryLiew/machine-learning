from sklearn import neighbors,datasets,preprocessing
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.metrics import accuracy_score
'KNN分类模型'

#加载数据，内置数据（鸢尾花）
iris=datasets.load_iris()
x,y=iris.data,iris.target
print(x.shape)
print(y.shape)

#划分训练集和测试集
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=0)
print(x_train[0])
#数据预处理
StandardScaler=preprocessing.StandardScaler().fit(x_train)#创建数据标准化的对象
x_train=StandardScaler.transform(x_train)#对训练数据进行标准化处理
x_test=StandardScaler.transform(x_test)#对测试数据进行标准化处理

print(x_train[0])
print(x_train.mean(),x_train.std())

#创建K紧邻算法模型
knn=neighbors.KNeighborsClassifier()#n_neighbor被分类数据周围的已分类数据

#给模型传入数据
knn.fit(x_train,y_train)

#训练数据：使用交叉验证训练方法
scoring=cross_val_score(knn,x_train,y_train,cv=5,scoring="accuracy")
print(scoring.mean())

#预测
y_pred=knn.predict(x_test)
print(y_pred)
print(y_test)
print(accuracy_score(y_test,y_pred))









