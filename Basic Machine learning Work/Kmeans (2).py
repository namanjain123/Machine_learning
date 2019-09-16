# Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 3].values

#using the elbow method to find the optimal number of clusters
from sklearn.cluster import KMeans
wc=[]
for i in range(1,11):
    kmeans=KMeans(n_cluster= i ,init ='k-means++',max_iter=300,n_init=10,random_sate=0)
    kmeans.fit(X)
    wc.append(kmeans.inertia_) 
plt.plot(range(1,11),wc)
plt.show()
# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""