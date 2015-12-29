import numpy as np
import pandas
from sklearn import metrics
from sklearn import linear_model
import matplotlib.pyplot as plt

#set seed for presentation purposes
np.random.seed(678)

# Load the NY dataset
turnstile_weather = pandas.read_csv('D:\\Documents\\Dropbox\\turnstile_weather_v2.csv')

#Remove duplicate data and confounders of ENTRIESn_hourly
turnstile_weather = turnstile_weather.drop(['datetime','TIMEn','DATEn','EXITSn','ENTRIESn','EXITSn_hourly'],1)

#subset data
turnstile_weather = turnstile_weather[:][turnstile_weather['UNIT'] == 'R008']
turnstile_weather = turnstile_weather[:][turnstile_weather['hour'] == 16]
turnstile_weather = turnstile_weather.drop(['UNIT','hour'],1)

#convert categorical to dummy variables
turnstile_weather = pandas.get_dummies(turnstile_weather)

#Convert all numbers into float for fitting
turnstile_weather = turnstile_weather.iloc[:,0:len(turnstile_weather.columns)].astype(float)

#normalizing data
turnstile_weather = (turnstile_weather - turnstile_weather.mean()) / (turnstile_weather.max() - turnstile_weather.min())

#Creation of dataset and target dataframe
tw_data = turnstile_weather[['weekday', 'conds_Light Rain', 'conds_Overcast', 'conds_Rain']]
tw_target = turnstile_weather[['ENTRIESn_hourly']]

print(tw_data.columns)

#Building Model
model = linear_model.SGDRegressor(n_iter=20000)
results = model.fit(tw_data, tw_target)
# The coefficients
print('Coefficients: \n', results.coef_, results.intercept_)
# Explained variance score: 1 is perfect prediction
print(metrics.r2_score(tw_target, results.predict(tw_data)))

#plotting target against predicted
plt.scatter(tw_target, results.predict(tw_data),  color='black')
plt.plot([-1,1], [-1,1], color='blue', linewidth=3)
plt.show()

