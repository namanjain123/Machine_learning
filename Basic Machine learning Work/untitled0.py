# polynomial regresion
#I mport the libraries
 import numpy as np
import  pandas as pd
import matplotlib.pyplot as plt

#data enteres
dataset = pd.read_csv('Position_Salaries.csv')
x= dataset.iloc[:,1:2].values
y= dataset.iloc[:,2].values
#polynomial fitting
from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(x,y)
#fitting the polynomial reggresion to the datset
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree = 4)
xpoly= poly.fit_transform(x) 

lin=LinearRegression()
lin.fit(xpoly,y)

#visualising the linear regression result
plt.scatter(x,y,color='blue')
plt.plot(x ,reg.predict(x), color='red')
plt.title('salary vs time' )
plt.xlabel('position')
plt.ylabel('salary')
plt.show()
#polynomial reggression visuallisation result
plt.scatter(x,y,color='blue')
plt.plot(x,lin.predict( poly.fit_transform(x) ), color='red')
plt.title('salary vs time' )
plt.xlabel('position')
plt.ylabel('salary')
plt.show()
#predict a new result with linear regrresion
reg.predict(6.5)

#ploynomial regression 
lin.predict( poly.fit_transform(6.5))

