import numpy as np
import pandas
from sklearn import  cross_validation, metrics, linear_model
import matplotlib.pyplot as plt

#set seed for presentation purposes
np.random.seed(123)

# Load the NY dataset
turnstile_weather = pandas.read_csv('D:\\Documents\\Dropbox\\turnstile_weather_v2.csv')

#Remove duplicate data and confounders of ENTRIESn_hourly
turnstile_weather = turnstile_weather.drop(['datetime','TIMEn','DATEn','EXITSn','ENTRIESn','EXITSn_hourly'],1)

#creating new dataset containing riderhip values normalized to UNIT and hour 
units = turnstile_weather['UNIT'].unique()[:-2]
hours = turnstile_weather['hour'].unique()
normData = pandas.DataFrame(data = {'normENTRIES' : [],'weekday' : [],'conds_Light Rain' : [],'conds_Overcast' : [],'conds_Rain' : []})

for unit in units:
    for hour in hours:
        sub = turnstile_weather[:][turnstile_weather['UNIT'] == unit]
        sub = sub[:][sub['hour'] == hour]
        sub = sub.drop(['UNIT','hour'],1)
        sub = pandas.get_dummies(sub)
        if 'conds_Light Rain' not in sub.columns:
            sub['conds_Light Rain'] = np.zeros(len(sub['weekday']))
        if 'conds_Rain' not in sub.columns:
            sub['conds_Rain'] = np.zeros(len(sub['weekday']))
        if 'conds_Overcast' not in sub.columns:
            sub['conds_Overcast'] = np.zeros(len(sub['weekday']))
        meanENTRIES = np.nanmean(sub['ENTRIESn_hourly'])
        temp = pandas.DataFrame(data = {'normENTRIES' : sub['ENTRIESn_hourly']/meanENTRIES,'weekday' : sub['weekday'],'conds_Light Rain' : sub['conds_Light Rain'],'conds_Overcast' : sub['conds_Overcast'],'conds_Rain' : sub['conds_Rain']})
        normData = pandas.concat([normData, temp])

#Converting data to float and scaling them
normData = normData.iloc[:,0:len(normData.columns)].astype(float)
normData = (normData - normData.mean()) / (normData.max() - normData.min())
        
#Creation of dataset and target dataframe
tw_data = normData.drop(['normENTRIES'],1)
tw_target = normData[['normENTRIES']]

#Create training and testing set
X_train, X_test, y_train, y_test = cross_validation.train_test_split(tw_data, tw_target, test_size=0.4, random_state=0)


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