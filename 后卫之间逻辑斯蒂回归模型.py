from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.feature_selection import SelectPercentile,f_classif
from sklearn.feature_selection import VarianceThreshold,f_classif,mutual_info_classif as MIC
import pandas as pd # 用选择的变量重构矩阵
import joblib
print("数据规范化处理")
from sklearn.preprocessing import StandardScaler
hou=pd.read_csv(r'C:\Users\86139\Desktop\hou.csv')
hou.loc[:993,'position']=1
hou.loc[994:,'position']=0
X=hou.iloc[:,9:-1]
Y=hou['position']
scaler =StandardScaler()
Xnew =scaler.fit_transform(X)
X_train,X_test,y_train,y_test=train_test_split(Xnew,Y,stratify=Y,random_state=42)
models=[]
for i in [0.1,1,10,100,1000]:
    logreg =LogisticRegression(C=i,max_iter=1000).fit(X_train,y_train)
    print("C={},训练集精度{}".format(i,logreg.score(X_train,y_train)))
    print("C={},测试集精度{}".format(i,logreg.score(X_test,y_test)))
    models.append((i,logreg))

import matplotlib.pyplot as plt
plt.plot(models[1][1].coef_.T,'o',label="C=1")
plt.xticks(range(X.shape[1]),X.columns,rotation=90)
plt.hlines(0,0,X.shape[1]) # 横轴0~30
plt.ylim(-5,5)
plt.xlabel("coefficient index")
plt.ylabel("coefficient magnitude")
plt.legend()
plt.show()


logreg = LogisticRegression(C=1, max_iter=1000).fit(X_train, y_train)
plt.plot(logreg.coef_.T,'o',label="C=1")
plt.xticks(range(X.shape[1]),X.columns,rotation=90)
plt.hlines(0,0,X.shape[1]) # 横轴0~30
plt.ylim(-5,5)
plt.xlabel("coefficient index")
plt.ylabel("coefficient magnitude")
joblib.dump(logreg,'所有的后卫logistic模型.pkl')
print(logreg.coef_.T)
plt.legend()
plt.show()
print("选择变量")
hou=pd.read_csv(r'C:\Users\86139\Desktop\hou.csv')
hou.loc[:993,'position']=1
hou.loc[994:,'position']=0
X=hou.iloc[:,9:-1]
Y=hou['position']
print("选择f检验前25的数据")
selector2= SelectPercentile(f_classif, 25)
Xnew=selector2.fit_transform(X, Y)
#运用PCA计算特征的占比是多少
from PCA import PCA
pca=PCA(Xnew)
ans=pca.SVDdecompose()
print(ans[1])
pca.plotScore(Y,xAxis=2,inOne=True)
indx=np.argwhere(selector2.get_support())[:,0]
print(indx)
X=X.iloc[:,indx]
X1=X
X =scaler.fit_transform(X)
X_train,X_test,y_train,y_test=train_test_split(X,Y,stratify=Y,random_state=42)
models=[]
for i in [0.1,1,10,100,1000]:
    logreg =LogisticRegression(C=i,max_iter=1000).fit(X_train,y_train)
    print("C={},训练集精度{}".format(i,logreg.score(X_train,y_train)))
    print("C={},测试集精度{}".format(i,logreg.score(X_test,y_test)))
    models.append((i,logreg))
import matplotlib.pyplot as plt
logreg100=LogisticRegression(C=100,max_iter=10000).fit(X_train,y_train)
plt.plot(logreg100.coef_.T,'^',label="C=100")
print(logreg100.coef_.T)
plt.xticks(range(X1.shape[1]),X1.columns,rotation=90)
plt.hlines(0,0,X.shape[1]) # 横轴0~30
plt.ylim(-8,8)
plt.xlabel("coefficient index")
plt.ylabel("coefficient magnitude")
plt.legend()
plt.show()