import sys

import matplotlib.pyplot as plt
import numpy as np
import re
import pandas as pd

# mean filter
def noise_filter(filter_size,list_a):
    a = int(filter_size/2)
    tmp = []
    length = len(list_a)
    for i in range(0,len(list_a)):
        if (i <= length-a-1 and i >= a):
            average = np.average(list_a[i-a:i+a])
            tmp.append(average)
        else:
            tmp.append(list_a[i])
    return tmp


time = []
RSS = []
rss_list = []
average_rss_list = []


files = ['hand_cover','hand_quick_wave','hand_around']

for j in range(0,3):
    data = pd.read_csv('./recent_files/' + files[j] + '_v1.csv')
    # data = pd.read_csv(files[j] + '.csv')
    x = data['Signal strength (dBm)']
    for i in x:
        i = int(re.sub(r'[a-zA-Z]',"",i))
        if (i > -80):
            i = -80
        if (i < -99):
            i = -99
        RSS.append(i)
    rss_list.append(RSS)
    RSS = []

window_size= 100
for i in rss_list:
    tmp = noise_filter(7,i)
    average_rss_list.append(tmp)

data_set = []
for i in range(0,3):
    gesture = files[i]
    for j in range(50,2000,window_size):
        tmp = average_rss_list[i][j:j+window_size]
        tmp.append(gesture)
        data_set.append(tmp)
# print(data_set)
pd.DataFrame(data_set).to_csv("./data_set.csv")


# draw the plot
draw_size = 2000      
color = ['r--','g--','b--']
time = np.arange(0,draw_size)
for i in range(0,3):
    plt.subplot(2,2,i+1)
    plt.plot(time,average_rss_list[i][10:draw_size + 10],'r--',label=files[i])
    plt.title(files[i])
    plt.ylabel('RSS')

plt.subplot(2,2,4)

for i in range(0,3):
    plt.plot(time,average_rss_list[i][10:draw_size + 10],color[i],label=files[i])
    plt.title('3 in 1')
    plt.legend()
    plt.xlabel('packets number')
    plt.ylabel('RSS')


plt.show()


