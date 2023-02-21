import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_curve,auc
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score as ACC
from sklearn.metrics import matthews_corrcoef as MCC
from sklearn.metrics import roc_auc_score as AUC
from sklearn.metrics import f1_score as F1
from sklearn.model_selection import train_test_split
data=pd.read_csv('data_all.csv')
data=data.replace(np
x=np.array(data.iloc[:,:-1])

y=np.array(data.iloc[:,-1])
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=51,stratify=y)
import joblib
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
def maxabs(a, axis=None):
    """Return slice of a, keeping only those values that are furthest away
    from 0 along axis"""
    maxa = a.max(axis=axis)
    mina = a.min(axis=axis)
    p = abs(maxa) > abs(mina) # bool, or indices where +ve values win
    n = abs(mina) > abs(maxa) # bool, or indices where -ve values win
    if axis == None:
        if p: return maxa
        else: return mina
    shape = list(a.shape)
    shape.pop(axis)
    out = np.zeros(shape, dtype=a.dtype)
    out[p] = maxa[p]
    out[n] = mina[n]
    return out
avgs=maxabs(x_train, axis=0)
avgs[avgs==0]=1
finger11=x_train/avgs
finger22=x_test/avgs
pca2 = PCA(n_components=3)
newX12 = pca2.fit_transform(finger11)
newX22 = pca2.transform(finger11)
newX32 = pca2.transform(finger22)
def test(d):
    f=[]
    for i in range(d.shape[0]):
        if d[i][0]>0.4:
            continue
        else:
            if d[i][1]>0.2:
                continue
            elif d[i][2]>0.4:
                continue
            else:
                f.append(d[i])
    return np.array(f)
newX22_1=test(newX22)
newX32_1=test(newX32)
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')

# ax.scatter(newX22[:, 0], newX22[:, 1],newX22[:, 2],
#   marker='x', color='blue', s=40, label='class 1')
ax.scatter(newX22[:, 0], newX22[:, 1], newX22[:, 2],
           marker='^', s=20, label='train set')
ax.scatter(newX32[:, 0], newX32[:, 1], newX32[:, 2],
           marker='o', s=20, label='test set')
#ax.scatter(newX22_1[:, 0], newX22_1[:, 1], newX22_1[:, 2],
#           marker='^', s=20, label='train set')
#ax.scatter(newX32_1[:, 0], newX32_1[:, 1], newX32_1[:, 2],
#          marker='o', s=20, label='test set')
ax.set_xlabel('PCA1')
ax.set_ylabel('PCA2')
ax.set_zlabel('PCA3')

# plt.title('3D Scatter Plot')
# plt.savefig('test.jpg')
plt.show()
