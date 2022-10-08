import numpy as np
from sklearn import linear_model,svm,neighbors,datasets,preprocessing
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.metrics import accuracy_score

iris=datasets.load_iris()
x,y=iris.data,iris.target

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3)

scaler=preprocessing.StandardScaler().fit(x_train)
x_train=scaler.transform(x_train)
x_test=scaler.transform(x_test)

model1=linear_model.SGDClassifier()
model2=neighbors.KNeighborsClassifier()
model3=linear_model.LogisticRegression()
model4=svm.SVC()

model1.fit(x_train,y_train)
model2.fit(x_train,y_train)
model3.fit(x_train,y_train)
model4.fit(x_train,y_train)

y_predict1=model1.predict(x_test)
y_predict2=model2.predict(x_test)
y_predict3=model3.predict(x_test)
y_predict4=model4.predict(x_test)

score1=accuracy_score(y_test,y_predict1)
score2=accuracy_score(y_test,y_predict2)
score3=accuracy_score(y_test,y_predict3)
score4=accuracy_score(y_test,y_predict4)
print(score1,score2,score3,score4)