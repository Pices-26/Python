#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt

plt.plot([5,2,1],[7,5,3])


# In[2]:


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


# In[3]:


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


# In[4]:


#HISTOGRAM
population_ages = [22,65,32,65,23,76,43,23,77,54,23,12,99] #13

#ids = [x for x in range(len(population_ages))]
bins = [0 ,10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

plt.hist(population_ages, bins, histtype='bar', rwidth=0.8)

plt.xlabel('x')
plt.ylabel('y')

plt.show()


# In[5]:


#SCATTER GRAPHS

x = [1,2,3,4,5,6,7,8,9]
y = [5,3,7,2,8,4,1,8,4]

plt.scatter(x,y, label='scatt', color='r', marker='*', s=100) #MORE MARKERS : https://matplotlib.org/3.1.0/api/markers_api.html 

plt.xlabel('x')
plt.ylabel('y')
plt.legend()

plt.show()


# In[6]:


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


# In[7]:


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


# In[8]:


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


# In[9]:


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


# In[10]:


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


# In[11]:


#handling unix time
import datetime as dt
dateconv = np.vectorize(dt.datetime.fromtimestamp)
date = dateconv(date)


# In[12]:


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


# In[13]:


#SPINES AND HORIZONTAL LINES
import pandas_datareader as data

def graph_data(stock):
    
    fig = plt.figure()
    ax1 = plt.subplot2grid((1,1), (0,0))
    
    df = data.DataReader(stock,'yahoo')
    ax1.plot_date(df.index, df.Close, '-')
    
    ax1.axhline(df.Close[0], color='k', linewidth=2)#draws a horizontal line
    
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
    
    #spines - outlines of the graph
    ax1.spines['left'].set_color('c')
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    ax1.spines['bottom'].set_color('r')
    
    ax1.spines['left'].set_linewidth(5)
    
    ax1.tick_params(axis='x',colors='#f06215') #bottom labels become orange
    
    plt.xlabel('Date', color='b')
    plt.ylabel('Price', color='r')
    plt.title(stock)
    plt.legend()
    plt.show()

graph_data('EBAY')


# In[14]:


#STYLES
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
import pandas as pd
from mpl_finance import candlestick_ohlc
from matplotlib import style

style.use('dark_background')
#style.use('fivethirtyeight')
print(plt.style.available) # prints all possible styles

print(plt.__file__) # prints file location

def graph_data(stock):
    
    fig = plt.figure()
    ax1 = plt.subplot2grid((1,1), (0,0))
    
    df = data.DataReader(stock,'yahoo')
    ax1.plot_date(df.index, df.Close, '-')
    
    
    
    plt.xlabel('Date')
    plt.ylabel('Price', color='r')
    plt.title(stock)
    plt.legend()
    plt.show()

graph_data('EBAY')


# In[15]:


#LIVE GRAPHS
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import style

style.use('fivethirtyeight')

fig = plt.figure()
ax1 = fig.add_subplot(1,1,1)

def animate(i):
    graph_data = open('example.txt', 'r').read()
    lines = graph_data.split('\n')
    xs = []
    ys = []
    for line in lines:
        if len(line) > 1:
            x, y = line.split(',')
            xs.append(x)
            ys.append(y)
    ax1.clear()        
    ax1.plot(xs,ys)
    
ani = animation.FuncAnimation(fig, animate, interval = 1000)


# In[16]:


#ANNOTATION AND PLACING TEXT
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
import pandas as pd
from mpl_finance import candlestick_ohlc
from matplotlib import style

style.use('dark_background')

def graph_data(stock):
    
    fig = plt.figure()
    ax1 = plt.subplot2grid((1,1), (0,0))
    
    df = data.DataReader(stock,'yahoo')
    ax1.plot_date(df.index, df.Close, '-')
    
    ax1.grid(True)
    
    font_dict = {'family' : 'serif', 'color': 'pink', 'size':15}
    ax1.text(df.index[1000], df.Close[2000], 'Text Example', fontdict = font_dict)
    
    ax1.annotate('Big news!', (df.index[1000], df.Close[1000]), 
                 xytext=(0.8,0.9), textcoords='axes fraction',
                arrowprops = dict(facecolor = 'grey', color='grey'))
    
    plt.xlabel('Date')
    plt.ylabel('Price', color='r')
    plt.title(stock)
    plt.legend()
    plt.show()

graph_data('EBAY')


# In[17]:


#ANNOTATING LAST PRICE TO EDGE
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
import pandas as pd
from mpl_finance import candlestick_ohlc
from matplotlib import style

style.use('dark_background')

def graph_data(stock):
    
    fig = plt.figure()
    ax1 = plt.subplot2grid((1,1), (0,0))
    
    df = data.DataReader(stock,'yahoo')
    ax1.plot_date(df.index, df.Close, '-')
    
    ax1.grid(True)
    
    
    ax1.annotate('Big news!', (df.index[1000], df.Close[1000]), 
                 xytext=(0.8,0.9), textcoords='axes fraction',
                arrowprops = dict(facecolor = 'grey', color='grey'))
    
    #properties of the box
    bbox_props = dict(boxstyle='round', fc='k',ec='w', lw=1)
    
    #LAST PRICE ANNOTATION
    ax1.annotate(str(df.Close[-1]), (df.index[-1], df.Close[-1]),
                 xytext = (df.index[-1] , df.Close[-1]), 
                 bbox = bbox_props)
    
    
    
    plt.xlabel('Date')
    plt.ylabel('Price', color='r')
    plt.title(stock)
    plt.legend()
    plt.show()

graph_data('EBAY')


# In[18]:


#SUBPLOTS
import random

style.use('fivethirtyeight')

fig = plt.figure()

def create_plots():
    xs = []
    ys = []
    
    for i in range(10):
        x = i
        y = random.randrange(10)
        
        xs.append(x)
        ys.append(y)
    return xs, ys


#add subplot syntax
#ax1 = fig.add_subplot(2,1,1) # height, width, plot
#ax2 = fig.add_subplot(2,2,2)
#ax3 = fig.add_subplot(2,1,2)

#x , y = create_plots()
#ax1.plot(x,y)

#x , y = create_plots()
#ax2.plot(x,y)

#x , y = create_plots()
#ax3.plot(x,y)

#subplot to grid
x , y = create_plots()
ax1 = plt.subplot2grid((6,1), (0,0), rowspan=1, colspan=1)
x , y = create_plots()
ax2 = plt.subplot2grid((6,1), (1,0), rowspan=1, colspan=1)
x , y = create_plots()
ax3 = plt.subplot2grid((6,1), (5,0), rowspan=1, colspan=1)



plt.show()


# In[10]:


#3D
get_ipython().run_line_magic('matplotlib', 'widget')
get_ipython().run_line_magic('matplotlib', 'ipympl')
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import numpy as np

fig = plt.figure()
ax1 = fig.add_subplot(111, projection='3d')

x = [1,2,3,4,5,6,7,8,9,10]
y = [5,7,2,8,2,6,3,7,4,8]
z = [1,5,3,2,7,3,1,10,3,5]

ax1.plot(x,y,z)

ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('Z')

plt.show()


# In[12]:


#3D SCATTER PLOT
get_ipython().run_line_magic('matplotlib', 'widget')
get_ipython().run_line_magic('matplotlib', 'ipympl')
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
from matplotlib import style

style.use('ggplot')

fig = plt.figure()
ax1 = fig.add_subplot(111, projection='3d')

x = [1,2,3,4,5,6,7,8,9,10]
y = [5,6,7,8,2,5,6,3,7,2]
z = [1,2,6,3,2,7,3,3,7,2]

x2 = [-1,-2,-3,-4,-5,-6,-7,-8,-9,-10]
y2 = [-5,-6,-7,-8,-2,-5,-6,-3,-7,-2]
z2 = [1,2,6,3,2,7,3,3,7,2]

ax1.scatter(x, y, z, c='g', marker='o')
ax1.scatter(x2, y2, z2, c ='r', marker='o')

ax1.set_xlabel('x axis')
ax1.set_ylabel('y axis')
ax1.set_zlabel('z axis')

plt.show()


# In[ ]:


#3D BAR CHART
get_ipython().run_line_magic('matplotlib', 'widget')
get_ipython().run_line_magic('matplotlib', 'ipympl')
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import style
style.use('ggplot')

fig = plt.figure()
ax1 = fig.add_subplot(111, projection='3d')

x3 = [1,2,3,4,5,6,7,8,9,10]
y3 = [5,6,7,8,2,5,6,3,7,2]
z3 = np.zeros(10)

dx = np.ones(10)
dy = np.ones(10)
dz = [1,2,3,4,5,6,7,8,9,10]

ax1.bar3d(x3, y3, z3, dx, dy, dz)


ax1.set_xlabel('x axis')
ax1.set_ylabel('y axis')
ax1.set_zlabel('z axis')

plt.show()

