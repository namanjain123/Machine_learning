#multi  linear regresion
#Import the libraries
 import numpy as np
import  pandas as pd
import matplotlib.pyplot as plt

#data enteres
dataset = pd.read_csv('50_Startups.csv')
x= dataset.iloc[:,:-1].values
y= dataset.iloc[:,4].values

#encode the column
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
label = LabelEncoder()
x[:,3]=label.fit_transform(x[: , 3])
one = OneHotEncoder(categorical_features =[3])
x=one.fit_transform(x).toarray()

#avoiding the dumy variable trap
x=x[:, 1:]

#spliting into train and test set
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y ,test_size = 0.2 ,random_state = 0) 

#fitting multiple linear regression
from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(x_train,y_train)

#predict the test Set result
pred=reg.predict(x_test)

#Building the optimal model using backward elimination
import statsmodels.formula.api as sm
x = np.append(arr = np.ones((50,1)).astype(int), values =x ,axis =1)
x_opt = x[:, [0,1,2,3,4,5]]
reg_ols= sm.OLS(endog =y,exog=x_opt).fit()
 reg_ols.summary() 
 x = np.append(arr = np.ones((50,1)).astype(int), values =x ,axis =1)
x_opt = x[:, [0,3]]
reg_ols= sm.OLS(endog =y,exog=x_opt).fit()
 reg_ols.summary() 
 