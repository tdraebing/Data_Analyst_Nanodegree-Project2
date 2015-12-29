import numpy as np
import scipy
import scipy.stats
import pandas

np.random.seed(123)

turnstile_weather = pandas.read_csv('D:\\Documents\\Dropbox\\turnstile_data_master_with_weather.csv')
turnstile_weather = turnstile_weather.dropna(axis = 0, how = 'any')
with_rain = turnstile_weather['ENTRIESn_hourly'][turnstile_weather['rain']==1]
without_rain = turnstile_weather['ENTRIESn_hourly'][turnstile_weather['rain']==0]

with_rain_mean = np.mean(with_rain)
print(with_rain_mean)
without_rain_mean = np.mean(without_rain)
print(without_rain_mean)
print(scipy.stats.shapiro(with_rain))
#(0.5969148874282837, 0.0) --> no normality
print(scipy.stats.shapiro(without_rain))
#(0.5993077754974365, 0.0) --> no normality
print(scipy.stats.mannwhitneyu(without_rain,with_rain)[1]*2)
#(0.0132316025654)
#p < 0.05 --> significant difference between both groups

