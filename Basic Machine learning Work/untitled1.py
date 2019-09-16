#simople linear regresion
import numpy as np
import  pandas as pd
import matplotlib.pyplot as plt
dataset = pd.read_csv('Salary_Data.csv')
X= dataset.iloc[:,:-1].values
y= dataset.iloc[:,3].values

from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean' , axis = 0)
imputer = imputer.fit(X[:, 1:3])
X[:, 1:3 ]= imputer.transform(X[:, 1:3])
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
label = LabelEncoder()
X[:,0]=label.fit_transform(X[: , 0])
one = OneHotEncoder(categorical_features =[0])
X=one.fit_transform(X).toarray()
label1 = LabelEncoder()
y=label1.fit_transform(y)
#splitting it into tranning nd test set
from sklearn.cross_validation import train_test_split
X_test,X_train,y_test,y_train = train_test_split(X,y ,test_size = 0.2 ,random_state = 0)
