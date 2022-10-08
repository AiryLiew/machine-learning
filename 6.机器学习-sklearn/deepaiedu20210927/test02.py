from sklearn import neighbors,datasets,preprocessing
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
'KNN模型对K值的选择'
iris=datasets.load_iris()
x=iris.data
y=iris.target
StandardScaler=preprocessing.StandardScaler().fit(x)
x=StandardScaler.transform(x)

k_range=range(1,51)
k_score=[]
for k in k_range:
    knn=neighbors.KNeighborsClassifier(n_neighbors=k)
    knn.fit(x,y)
    score=cross_val_score(knn,x,y,cv=5,scoring="accuracy")
    k_score.append(score.mean())
plt.plot(k_range,k_score)
plt.show()
