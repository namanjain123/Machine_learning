#Upper Confidence Bound (UCB)
#import the files
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#import the dataset
dataset = pd.read_csv('Ads_CTR_Optimisation.csv')
 
"""#implement of the dataset by random selection
# Implementing Random Selection
import random
N = 10000
d = 10
ads_selected = []
total_reward = 0
for n in range(0, N):
    ad = random.randrange(d)
    ads_selected.append(ad)
    reward = dataset.values[n, ad]
    total_reward = total_reward + reward

# Visualising the results
plt.hist(ads_selected)
plt.title('Histogram of ads selections')
plt.xlabel('Ads')
plt.ylabel('Number of times each ad was selected')
plt.show()"""
#implementation of ucb algorithim
import math
N=10000
d=10
totalrewards=0
adsselected=[]
numbersofselection =[0]*d
sumofrewards=[0]*d
for n in range(0,N):
    maxupperbound=0
    ad=0
    for i in range(0,d):
        if(numbersofselection[i]>0):
            averagereward=sumofrewards[i]/numbersofselection[i]
            delta=math.sqrt(3/2*math.log(n)/numbersofselection[i])
            ucb=averagereward+delta
        else:
            ucb=1e400
        if (ucb>maxupperbound):
            maxupperbound=ucb
            ad = i
    adsselected.append(ad)
    numbersofselection[ad] = numbersofselection[ad] + 1
    reward = dataset.values[n, ad]
    sumofrewards[ad] = sumofrewards[ad] + reward
    totalrewards = totalrewards + reward


       
       
       
       
       

  