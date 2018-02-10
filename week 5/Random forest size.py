import numpy
import pandas
from sklearn.ensemble import RandomForestRegressor
from sklearn.cross_validation import *
from sklearn.metrics import r2_score
abalone=pandas.read_csv('D:\\abalone.csv')
abalone['Sex'] = abalone['Sex'].map(lambda x: 1 if x == 'M' else (-1 if x == 'F' else 0))
X=abalone.iloc[:,0:8]
y=abalone.iloc[:,8]
predictions = clf.predict(X)
predictions
for n in range(100,101):
    clf = RandomForestRegressor(random_state=1,n_estimators=n)
    clf.fit(X, y)
    kf = KFold(len(X),n_folds=5,shuffle=True,random_state=1)
    arr=cross_val_score(estimator=clf,X=X,y=y,cv=kf,scoring='r2')
    m=arr.mean()
    print(n,') ',m)

for train_index, test_index in kf:
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
