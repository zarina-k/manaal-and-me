#!/usr/bin/env python
# coding: utf-8

# In[411]:


#%% import necessary modules
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import numpy.random as rn
from scipy.stats import binom


# In[412]:


#%% Q1 setup
x = np.reshape(range(20), (5, 4))
print('X is: \n', x)
#%% Example: Write the python code to calculate the sum of each row. 

print('Example: Write the python code to calculate the sum of each row.')
    
print('Code is: x.sum(1)',x.sum(1) )

print('Value is:', x.sum(1))


#%% Q1a. Write the python code to calculate the sum of each column.
print('Q1a: ',x.sum(0))


#%% Q1b. Write the python code to calculate the max of each row.
print('Q1b: ',x.max(1))


#%% Q1c. Write the python code to calculate the mean of the whole matrix.
print('Q1c: ',x.mean())


#%% Q1d. Write the python code to multiple each column by its (column index + 1).

print('Q1d: ',x*(x[0]+1))


#%% Q1e. Write the python code to divide each value by the max value of its row.
print('Q1e: ',x/x.max(1).reshape(5,1))


#%% Q1f. Write the python code to calculate a weighted sum of each row, where each column contributes 1 / 2.0**(column index+1) of its value.
print('Q1f: ',(x*(1/2.0)**(x[0]+1)).sum(1))


#%% Q1g. Write the python code to calculate a weighted sum of each row, where each row contribute 1/(row index + 1) of its value.
print('Q1g: ',(x*(1/(x[1]+1))).sum(0))


#%% Q1h. Write the python code to return the rows of x where its sum is a multiple of 3.
print('Q1h: ',x[x.sum(axis=1)%3 == 0])


#%% Q1i. Write the python code to return the columns of x where its sum is a multiple of 8.
print('Q1i: ',[a for a in zip(*x) if sum(a)%8 == 0])

#%% Q1j. Write the python code to change the odd numbers in x to 0.
print('Q1j: ', np.where(x%2 != 0, 0, x))


# In[413]:


#%% Q2 setup
N = 16;
rn.seed(0)
nHeads=[]
arr = rn.uniform(size=(10**5,16))
arr[arr>0.5]=1
arr[arr<=0.5]=0
nHeads=arr.sum(axis=1)
plt.hist(nHeads, bins=range(18))
plt.title('Q2a: Histogram')
plt.xlabel("nHeads")
plt.ylabel("Number of coins")
plt.show()

# In[414]:


#%% Q2b
plt.hist(nHeads, bins=range(18),density=True)
plt.title('Q2b: Probability Mass Function')
plt.xlabel("k")
plt.ylabel("P(nHeads=k)")
plt.show()

# In[415]:


#%% Q2c
hist_vals,x_locs = np.histogram(nHeads,bins=range(18))
arr2 = np.cumsum(hist_vals)/10**5
plt.plot(x_locs[:-1],arr2)
plt.title('Q2c: Cumulative Distribution Function')
plt.xlabel("k")
plt.ylabel("P(nHeads<=k)")
plt.show()

# In[416]:


#%% Q2d
def binomcdf():
    p=0.5
    n=16
    x=0
    result = []
    for a in range(17):
        result.append(binom.cdf(x,n,p))
        x+=1
    return result
prob = binomcdf()

plt.scatter(arr2,prob)
plt.title('Q2d: Scatter Plot')
plt.xlabel("Empirical CDF")
plt.ylabel("Theoretical CDF")
plt.show()

# In[417]:


plt.plot(arr2,linestyle='solid',marker='o')
plt.plot(prob,linestyle='solid',marker='D')
plt.legend(['Empirical CDF','Theoretical CDF'])
plt.title('Q2d: Line Plot')
plt.ylabel("CDF")
plt.xlabel("k")
plt.show()

# In[418]:


plt.loglog(arr2,prob,'bo')
plt.title('Q2d: Loglog Plot')
plt.xlabel("Empirical CDF")
plt.ylabel("Theoretical CDF")
plt.show()

# In[419]:


#%% Q3 setup
data = pd.read_csv('brfss.csv', index_col=0)

# data is a numpy array and the columns are age, current weight (kg), 
# last year's weight (kg), height (cm), and gender (1: male; 2: female).
data = data.drop('wtkg2',axis=1).dropna(axis=0, how='any').values

weight_change = data[:,1] - data[:,2] 
p1 = np.corrcoef(data[:,0],weight_change)[0][1].round(2)
plt.scatter(data[:,0],weight_change)
plt.title('Q3a: Correlation: %.2f' %p1)
plt.xlabel('age')
plt.ylabel('weight_change')
plt.show()

# In[420]:


p2 = np.corrcoef(data[:,1],weight_change)[0][1].round(2)
plt.scatter(data[:,1],weight_change)
plt.title('Q3b: Correlation: %.2f' %p2)
plt.xlabel('current_weight')
plt.ylabel('weight_change')
plt.show()

# In[421]:


p3 = np.corrcoef(data[:,2],weight_change)[0][1].round(2)
plt.scatter(data[:,2],weight_change)
plt.title('Q3c: Correlation: %.2f' %p3)
plt.xlabel('weight_a_year_ago')
plt.ylabel('weight_change')
plt.show()

# In[ ]:





# In[ ]:




