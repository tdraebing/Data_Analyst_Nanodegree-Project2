import numpy as np
import pandas
import matplotlib.pyplot as plt

#set seed for presentation purposes
np.random.seed(123)

# Load the NY dataset
turnstile_weather = pandas.read_csv('E:\\Dropbox\\turnstile_weather_v2.csv')

#subsetting the data
data = turnstile_weather[['ENTRIESn_hourly','hour']]

#reshaping data
hour_0=data['ENTRIESn_hourly'][data['hour']==0]
hour_4=data['ENTRIESn_hourly'][data['hour']==4]
hour_12=data['ENTRIESn_hourly'][data['hour']==12]
hour_16=data['ENTRIESn_hourly'][data['hour']==16]
hour_20=data['ENTRIESn_hourly'][data['hour']==20]
data2 = [hour_0, hour_4, hour_12, hour_16, hour_20]

#boxplot
plt.boxplot(data2)
plt.yscale('log')
plt.xticks([1,2,3,4,5], [0,4,12,16,20], rotation='horizontal')
plt.ylabel('Number of hourly subway entries')
plt.xlabel('Hour')
plt.title('Subway entries at specific hours')