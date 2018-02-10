from sklearn.ensemble import GradientBoostingClassifier
data=pandas.read_csv('D:\\grm.csv')
X=data.iloc[:,1:1776]
y=data.iloc[:,0]
X_train, X_test, y_train, y_test =train_test_split(X, y, test_size=0.8, random_state=241)
clf=GradientBoostingClassifier(n_estimators=250,verbose=True, random_state=241)
clf.fit(X_train,y_train)