import numpy as np
import pandas
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

#set seed for presentation purposes
np.random.seed(123)

# Load the NY dataset
turnstile_weather = pandas.read_csv('D:\\Documents\\Dropbox\\turnstile_weather_v2.csv')

#subset of ENTRIESn_hourly per rain
data_rain = list(turnstile_weather['ENTRIESn_hourly'][turnstile_weather['rain'] == 1])
data_norain = list(turnstile_weather['ENTRIESn_hourly'][turnstile_weather['rain'] == 0])



#plot histogram
plt.figure()  
plt.axis([1, 10e5, 0, 2500])  
plt.hist(data_norain, bins=np.logspace(0,10,100), alpha = 0.6)
plt.axvline(np.mean(data_norain), color='blue', linestyle='dashed', linewidth=2)
plt.hist(data_rain, bins=np.logspace(0,10,100), alpha = 0.6)
plt.axvline(np.mean(data_rain), color='green', linestyle='dashed', linewidth=2)
plt.xscale('log')
plt.xlabel('Number of hourly entries')
plt.ylabel('Frequency')
plt.title('Histogram of hourly subway entries')
blue_patch = mpatches.Patch(color='blue', label='no rain')
green_patch = mpatches.Patch(color='green', label='rain')
plt.legend(handles=[blue_patch, green_patch])
plt.show()

