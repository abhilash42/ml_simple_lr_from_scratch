import numpy as np
from numpy import mean
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
plt.rcParams['figure.figsize'] = (10.0, 9.0)
data = pd.read_csv('headbrain.csv')
print(data.shape)
data.head()

X = data['Head Size(cm^3)'].values
Y = data['Brain Weight(grams)'].values

mean_x = mean(X)
mean_y = mean(Y)

m = len(X)


b1 = ((mean_x)*(mean_y) - mean(X*Y))/((mean_x**2 - mean(X**2))) #best-fit-slope
b0 = mean_y - b1*mean_x

print(b1,b0)

max_x = np.max(X)+100
min_x = np.min(X)-100

x = np.linspace(min_x,max_x,1000)

y = b0 + b1*x

plt.plot(x,y,color='#58b970',label='Regression line ')

plt.scatter(X,Y,c='#ef5423',label='Scatter Plot')


rmse = 0
for i in range(m):
    y_pred = b0+b1*X[i]
    rmse += (Y[i]-y_pred)**2
rmse = np.sqrt(rmse/m)

ss_m=0
ss_cap=0

for i in range(m):
    y_pred = b0 + b1*X[i]
    ss_cap += (y_pred-Y[i])**2
    ss_m += (mean_y-Y[i])**2
r2 = 1 - (ss_cap/ss_m)

print(rmse)
print(r2)
