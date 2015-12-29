import numpy as np
import statsmodels.api as sm
import pandas
import matplotlib.pyplot as plt
from ggplot import *

#set seed for presentation purposes
np.random.seed(123)

# Load the NY dataset
turnstile_weather = pandas.read_csv('D:\\Documents\\Dropbox\\turnstile_weather_v2.csv')

#Remove duplicate data and confounders of ENTRIESn_hourly
turnstile_weather = turnstile_weather.drop(['datetime','TIMEn','DATEn','EXITSn','ENTRIESn','EXITSn_hourly'],1)

#subset data only keeping a few meaningful variables
turnstile_weather = turnstile_weather.drop(['conds','day_week','station','meanprecipi','meantempi','weather_lon','weather_lat','wspdi','meanwspdi','tempi','rain','meanpressurei','pressurei', 'precipi', 'fog', 'longitude', 'latitude','weekday'],1)
#turnstile_weather = turnstile_weather.drop(['wspdi','tempi','pressurei', 'precipi'],1)

#convert categorical to dummy variables
turnstile_weather = pandas.get_dummies(turnstile_weather)

#Convert all numbers into float for fitting
turnstile_weather = turnstile_weather.iloc[:,0:len(turnstile_weather.columns)].astype(float)

#Creation of dataset and target dataframe
tw_data = turnstile_weather.drop('ENTRIESn_hourly',1)
tw_target = turnstile_weather[['ENTRIESn_hourly']]

#Building Model
tw_data = sm.add_constant(tw_data)
model = sm.OLS(tw_target, tw_data)
results = model.fit()
print(results.summary())
print('Parameters: ', results.params)
print('R2: ', results.rsquared)

#Quality Control of Fit
#Residual Histogram
plt.hist(results.resid, bins = 30)
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.title('Histogram of regression residuals')
plt.show()
#QQplot:
sm.qqplot(results.resid, fit = True, line = '45')
plt.title('QQ-plot')
plt.show()

#Expected vs Predicted
plt.scatter(tw_target, results.predict(tw_data))
plt.plot([-2000,13000],[-2000,13000], color='red', linewidth = 2)
plt.xlabel('Expected')
plt.ylabel('Predicted')
plt.title('Expected vs. Predicted Subway Entries')
plt.show()

#Residuals per data point
plt.plot(results.resid[0:30])
plt.xlabel('Data Point #')
plt.ylabel('Residuals')
plt.title('Residuals per Data Point')
plt.show()