#!/usr/bin/env python
# coding: utf-8

# In[2]:


import matplotlib.pyplot as plt

plt.plot([5,2,1],[7,5,3])


# In[8]:


#PLOTTING LINES
x = [1,6,3]
y = [7,2,5]

x2 =[1,6,2] 
y2 =[10,14,12]

plt.plot(x,y, label = 'First line')
plt.plot(x2,y2, label = 'Second line' )

plt.xlabel('plot number')
plt.ylabel('importanta var')

plt.title('interesting graph \nCheck it out')

plt.legend()

plt.show()


# In[14]:


#BAR CHART
x = [2,4,6,8,10]
y = [4,6,3,1,6]
plt.bar(x,y, label='Bars1', color ='r')

x2 = [1,3,5,7,9]
y2 = [2,6,3,1,4]
plt.bar(x2,y2, label='Bars2', color ='g')

plt.xlabel('x')
plt.ylabel('y')

plt.title('interesting graph \nCheck it out')

plt.legend()

plt.show()


# In[18]:


#HISTOGRAM
population_ages = [22,65,32,65,23,76,43,23,77,54,23,12,99] #13

#ids = [x for x in range(len(population_ages))]
bins = [0 ,10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

plt.hist(population_ages, bins, histtype='bar', rwidth=0.8)

plt.xlabel('x')
plt.ylabel('y')

plt.show()


# In[26]:


#SCATTER GRAPHS

x = [1,2,3,4,5,6,7,8,9]
y = [5,3,7,2,8,4,1,8,4]

plt.scatter(x,y, label='scatt', color='r', marker='*', s=100) #MORE MARKERS : https://matplotlib.org/3.1.0/api/markers_api.html 

plt.xlabel('x')
plt.ylabel('y')
plt.legend()

plt.show()


# In[32]:


#STACK PLOTS - one section with many sections inside representing things

days = [1,2,3,4,5]

sleeping = [4,12,7,10,9]
eating = [1,1,1,2,1]
working = [8,8,8,4,5]
playing = [2,3,3,2,4]

#can't show labels
plt.stackplot(days, sleeping, eating, working, playing, colors=['r','c','b','k'])

#way around labels
plt.plot([],[], color='r', label='Sleeping', linewidth = 5)
plt.plot([],[], color='c', label='Eating', linewidth = 5)
plt.plot([],[], color='b', label='Working', linewidth = 5)
plt.plot([],[], color='k', label='Playing', linewidth = 5)

plt.xlabel('x')
plt.ylabel('y')
plt.legend()

plt.show()


# In[41]:


#PIE CHART
days = [1,2,3,4,5]

sleeping = [4,12,7,10,9]
eating = [1,1,1,2,1]
working = [8,8,8,4,5]
playing = [2,3,3,2,4]

slices = [9,1,5,4]
activities = ['sleeping', 'eating', 'working', 'playing']
cols = ['r','c','b','m']

plt.pie(slices, 
        labels=activities, 
        colors=cols, 
        startangle=90, # starts on 90 and goes anti clockwise 
        shadow=True, 
        explode=(0,0.1,0,0), # takes out a piece
        autopct='%1.1f%%') # adds percentages


# In[48]:


#loading data from files
import csv
#WAY NR 1
'''
x = []
y = []

with open('example.txt', 'r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    for row in plots:
        x.append(int(row[0]))
        y.append(int(row[1]))

plt.plot(x,y, label='loaded from file')
plt.legend()
plt.xlabel('x')
plt.ylabel('y')
'''
#BETTER WAY WITH numpy
import numpy as np

x, y = np.loadtxt('example.txt', delimiter=',', unpack=True) #works only with 2 variables


plt.plot(x,y, label='loaded from file')
plt.legend()
plt.xlabel('x')
plt.ylabel('y')
plt.show()


# In[62]:


#Getting data from the internet 
#using pandas

import pandas_datareader as data

def graph_data(stock):
    df = data.DataReader(stock,'yahoo')
    plt.plot_date(df.index, df.Close, '-')
    
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title('Stock Graph')
    plt.legend()
    plt.show()

graph_data('TSLA')


# In[83]:


#BASIC CUSTOMIZATION
import pandas_datareader as data

def graph_data(stock):
    
    fig = plt.figure()
    ax1 = plt.subplot2grid((1,1), (0,0))
    
    df = data.DataReader(stock,'yahoo')
    ax1.plot_date(df.index, df.Close, '-')
    
    plt.xticks(rotation=45) # rotating date labels
    
    ax1.grid(True, color='g', linewidth=1) # adds grid
    
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title('Stock Graph')
    plt.legend()
    plt.show()

graph_data('TSLA')


# In[84]:


#handling unix time
import datetime as dt
dateconv = np.vectorize(dt.datetime.fromtimestamp)
date = dateconv(date)


# In[128]:


#more customization - color and fills
import pandas_datareader as data

def graph_data(stock):
    
    fig = plt.figure()
    ax1 = plt.subplot2grid((1,1), (0,0))
    
    df = data.DataReader(stock,'yahoo')
    ax1.plot_date(df.index, df.Close, '-')
    
    plt.plot([],[], linewidth=5, label='loss', color='r',alpha=0.5)
    plt.plot([],[], linewidth=5, label='gain', color='g',alpha=0.5)
    
    plt.fill_between(df.index, df.Close, df.Close[0],where=(df.Close > df.Close[0]), facecolor='g', alpha=0.5)
    plt.fill_between(df.index, df.Close, df.Close[0],where=(df.Close < df.Close[0]), facecolor='r', alpha=0.5)
    #plt.fill_between(df.index, df.Close, 36, facecolor='m', alpha=0.5)
    
    plt.xticks(rotation=45) # rotating date labels
    plt.xticks(color='r') 
    
    plt.yticks([0,10,20,30,40,50,60])
    plt.yticks(color='m')
    
    plt.grid(True, color='g', linewidth=1) # adds grid
    
    
    plt.xlabel('Date', color='b')
    plt.ylabel('Price', color='r')
    plt.title(stock)
    plt.legend()
    plt.show()

graph_data('EBAY')


# In[ ]:




