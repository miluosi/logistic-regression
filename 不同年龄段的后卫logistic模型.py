from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import pandas as pd # 用选择的变量重构矩阵
import heapq
import joblib
scaler =StandardScaler()
from sklearn.preprocessing import StandardScaler
data= pd.read_csv(r'C:\Users\86139\Desktop\football4.csv')
X =data[data['position']==3]
x1 =X[(X['Age']<25) &(X['Rating']>70)]
x2 =X[(X['Age']<25) &(X['Rating']<70)]
x1['position1']=1
x2['position1']=0
data11=pd.concat([x1,x2],ignore_index=True)
X =data11.iloc[:,12:-2]
Y =data11.iloc[:,-1]
Xnew =scaler.fit_transform(X)
X_train,X_test,y_train,y_test=train_test_split(Xnew,Y,stratify=Y,random_state=42)
models=[]
for i in [0.1,1,10,100,1000]:
    logreg =LogisticRegression(C=i,max_iter=1000).fit(X_train,y_train)
    print("C={},训练集精度{}".format(i,logreg.score(X_train,y_train)))
    print("C={},测试集精度{}".format(i,logreg.score(X_test,y_test)))
    models.append((i,logreg))

import matplotlib.pyplot as plt
logreg =LogisticRegression(C=1,max_iter=1000).fit(X_train,y_train)
plt.plot(logreg.coef_.T,'o',label="C=1")
plt.xticks(range(X.shape[1]),X.columns,rotation=90)
plt.hlines(0,0,X.shape[1]) # 横轴0~30
plt.ylim(-5,5)
plt.xlabel("coefficient index")
plt.ylabel("coefficient magnitude")
plt.legend()
plt.show()
print(logreg.coef_.T)

import matplotlib.pyplot as plt
import numpy as np
import math

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

a =logreg.coef_ .T
index=heapq.nlargest(5, range(len(a)),a.take)
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
langs = list(X.columns[index].T)
students =(a[index].T).tolist()[0]
ax.bar(langs,students)
joblib.dump(logreg,'25岁以下后卫logistic模型.pkl')
plt.title("25岁以下的后卫运动员最重要的五个指标")
plt.show()
scaler =StandardScaler()
from sklearn.preprocessing import StandardScaler
data= pd.read_csv(r'C:\Users\86139\Desktop\football4.csv')
X =data[data['position']==3]
x1 =X[(X['Age']>25) &(X['Rating']>70) & (X['Age']<30)]
x2 =X[(X['Age']>25) &(X['Rating']<70)& (X['Age']<30)]
x1['position1']=1
x2['position1']=0
data11=pd.concat([x1,x2],ignore_index=True)
X =data11.iloc[:,12:-2]
Y =data11.iloc[:,-1]
Xnew =scaler.fit_transform(X)
X_train,X_test,y_train,y_test=train_test_split(Xnew,Y,stratify=Y,random_state=42)
models=[]
for i in [0.1,1,10,100,1000]:
    logreg =LogisticRegression(C=i,max_iter=1000).fit(X_train,y_train)
    print("C={},训练集精度{}".format(i,logreg.score(X_train,y_train)))
    print("C={},测试集精度{}".format(i,logreg.score(X_test,y_test)))
    models.append((i,logreg))

import matplotlib.pyplot as plt
logreg =LogisticRegression(C=1,max_iter=1000).fit(X_train,y_train)
plt.plot(logreg.coef_.T,'o',label="C=1")
plt.xticks(range(X.shape[1]),X.columns,rotation=90)
plt.hlines(0,0,X.shape[1]) # 横轴0~30
plt.ylim(-5,5)
plt.xlabel("coefficient index")
plt.ylabel("coefficient magnitude")
plt.legend()
plt.show()
print(logreg.coef_.T)

import matplotlib.pyplot as plt
import numpy as np
import math

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

a =logreg.coef_ .T
index=heapq.nlargest(5, range(len(a)),a.take)
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
langs = list(X.columns[index].T)
students =(a[index].T).tolist()[0]
ax.bar(langs,students)
joblib.dump(logreg,'25到30岁的后卫logistic模型.pkl')
plt.title("25到30岁的后卫运动员最重要的五个指标")
plt.show()
scaler =StandardScaler()
from sklearn.preprocessing import StandardScaler
data= pd.read_csv(r'C:\Users\86139\Desktop\football4.csv')
X =data[data['position']==3]
x1 =X[(X['Age']>30) &(X['Rating']>70)]
x2 =X[(X['Age']>30) &(X['Rating']<70)]
x1['position1']=1
x2['position1']=0
data11=pd.concat([x1,x2],ignore_index=True)
X =data11.iloc[:,12:-2]
Y =data11.iloc[:,-1]
Xnew =scaler.fit_transform(X)
X_train,X_test,y_train,y_test=train_test_split(Xnew,Y,stratify=Y,random_state=42)
models=[]
for i in [0.1,1,10,100,1000]:
    logreg =LogisticRegression(C=i,max_iter=1000).fit(X_train,y_train)
    print("C={},训练集精度{}".format(i,logreg.score(X_train,y_train)))
    print("C={},测试集精度{}".format(i,logreg.score(X_test,y_test)))
    models.append((i,logreg))

import matplotlib.pyplot as plt
logreg =LogisticRegression(C=1,max_iter=1000).fit(X_train,y_train)
plt.plot(logreg.coef_.T,'o',label="C=1")
plt.xticks(range(X.shape[1]),X.columns,rotation=90)
plt.hlines(0,0,X.shape[1]) # 横轴0~30
plt.ylim(-5,5)
plt.xlabel("coefficient index")
plt.ylabel("coefficient magnitude")
plt.legend()
plt.show()
print(logreg.coef_.T)

import matplotlib.pyplot as plt
import numpy as np
import math

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

a =logreg.coef_ .T
index=heapq.nlargest(5, range(len(a)),a.take)
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
langs = list(X.columns[index].T)
students =(a[index].T).tolist()[0]
ax.bar(langs,students)
joblib.dump(logreg,'30岁以上的后卫logistic模型.pkl')
plt.title("30岁以上的后卫运动员最重要的五个指标")
plt.show()