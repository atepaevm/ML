import pandas
from sklearn.feature_extraction import DictVectorizer
from scipy.sparse import hstack
from sklearn.linear_model import Ridge
from sklearn.feature_extraction.text import TfidfVectorizer
def get_data(path):
    df=pandas.read_csv(path)
    df.FullDescription=data_train.FullDescription.str.lower();
    #df.LocationNormalized=data_train.LocationNormalized.str.lower();
    #df.ContractTime=data_train.ContractTime.str.lower();
    df.LocationNormalized = data_train.LocationNormalized.replace('[^a-zA-Z0-9]', ' ', regex = True);
    #df.ContractTime = data_train.ContractTime.replace('[^a-zA-Z0-9]', ' ', regex = True);    
    #df.FullDescription = data_train.FullDescription.replace('[^a-zA-Z0-9]', ' ', regex = True);
    df['LocationNormalized'].fillna('nd', inplace=True)
    df['ContractTime'].fillna('nd', inplace=True)
    return df   
   
data_train=get_data('D:\\train.csv')
data_test=get_data('D:\\test.csv')

vectorizer = TfidfVectorizer(min_df=2)
enc = DictVectorizer()
clf = Ridge(alpha=1.0,random_state=241)


X_train=vectorizer.fit_transform(data_train.FullDescription)
X_train_categ = enc.fit_transform(data_train[['LocationNormalized', 'ContractTime']].to_dict('records'))

glue=hstack((X_train,X_train_categ))
clf.fit(glue,data_train.SalaryNormalized)

X_test=vectorizer.transform(data_test.FullDescription)
X_test_categ = enc.transform(data_test[['LocationNormalized', 'ContractTime']].to_dict('records'))
test_glue=hstack((X_test,X_test_categ))
test_glue
res=clf.predict(test_glue)
res
