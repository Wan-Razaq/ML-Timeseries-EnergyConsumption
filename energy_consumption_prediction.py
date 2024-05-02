#!/usr/bin/env python
# coding: utf-8

# In[62]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
import seaborn as sns


# In[17]:


data_train = pd.read_csv("/Users/wbm/Documents/Skilvul/appliance_data.csv")
data_train.head(50)


# In[9]:


statistics = data_train.describe()

print(statistics)


# In[24]:


# Rename columns
data_train.rename(columns={'Ampere (A)': 'Ampere', 'Voltage (V)': 'Voltage', 'Device ID': 'Device'}, inplace=True)

data_train.head()


# In[11]:


# Find the earliest and latest timestamps
earliest_timestamp = data_train["Timestamp"].min()
latest_timestamp = data_train["Timestamp"].max()

print("Earliest Timestamp:", earliest_timestamp)
print("Latest Timestamp:", latest_timestamp)


# In[14]:


# Size of the dataset
dataset_size = data_train.shape

print("Dataset Size:", dataset_size)


# In[18]:


# Check for missing values
missing_data = data_train.isnull().sum()
print("Missing values in each column:\n", missing_data)


# In[26]:


# Check for anomalies (negative measurement)
anomalies = data_train[(data_train['Ampere'] < 0) | (data_train['Voltage'] < 0)]
print("Anomalous records with negative values:", anomalies)


# In[28]:


#Calculate Power in watts
data_train['Power'] = data_train['Voltage'] * data_train['Ampere']

# Calculate energy in watt-hours (5-minutes interval)
data_train["Energy"] = data_train["Power"] * (5/60)


# In[29]:


print(data_train.head())


# In[30]:


# Plot the energy, too see siklus

plt.figure(figsize=(10,6))
plt.plot(data_train['Timestamp'], data_train['Energy'], marker='o', linestyle='-')
plt.title('Energy Consumption (wh) Over time')
plt.xlabel('Timestamp')
plt.ylabel('Energy (wh)')
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()


# In[57]:


# Extract additional time features

# Convert 'Timestamp' to datetime
data_train['Timestamp'] = pd.to_datetime(data_train['Timestamp'])

# Additional Timeseries
data_train['Hour'] = data_train['Timestamp'].dt.hour
data_train['DayOfWeek'] = data_train['Timestamp'].dt.dayofweek # Monday=0, Sunday=6
data_train['IsWeekend'] = data_train['DayOfWeek'].isin([5,6]).astype(int)
data_train['IsPeakHours'] = data_train['Hour'].apply(lambda x: 1 if 18 <= x <= 22 else 0)


# In[58]:


print(data_train.head(500))


# In[125]:


# Dataframe for ranking the devices energy consumption
energy_per_hour_per_device = data_train[['Timestamp', 'Device', 'Energy']].copy()

# Save the the dataframe to local device
energy_per_hour_per_device.to_csv('/Users/wbm/Documents/Skilvul/Web App Energy Consumption/energy_per_hour_per_device.csv', index=True)


# In[60]:


# Aggregate energy consumption per hour
energy_per_hour = data_train.groupby(data_train['Timestamp'].dt.floor('H')).agg({'Energy': 'sum'}).reset_index()
print(energy_per_hour)


# In[123]:


# Save the all dataframe to local device
energy_per_hour.to_csv('/Users/wbm/Documents/Skilvul/Web App Energy Consumption/energy_per_hour.csv', index=True)


# In[67]:


#plotting the energy consumption per hout

plt.figure(figsize=(15, 7))

sns.barplot(x='Timestamp', y='Energy', data=energy_per_hour, color='blue')
plt.title('Energy Consumption per Hour')
plt.xlabel('Hour')
plt.ylabel('Energy (Wh)')
plt.xticks(rotation=45)

# Customizes the x-ticks to show fewer labels to prevent overlap
tick_labels = plt.gca().get_xticklabels()
# Assuming you want to show a label every 6 hours
tick_labels = [label if i % 6 == 0 else "" for i, label in enumerate(tick_labels)]
plt.gca().set_xticklabels(tick_labels)

plt.grid(True)
plt.tight_layout()
plt.show()


# In[74]:


# Is the data stationary
# Dicky-Fuller Test

from statsmodels.tsa.stattools import adfuller

# Extract energy values from the DataFrame
energy_values = energy_per_hour['Energy']

# Perform the Dickey-Fuller test
adf, pvalue, usedlag_, nobs_, critical_values, icbest = adfuller(energy_values)

# Print the results
print(f'ADF Statistic: {adf}')
print(f'p-value: {pvalue}')
print('Critical Values:')
for key, value in critical_values.items():
    print(f'   {key}: {value}')


# In[90]:


# Extract and Plot Trend, Seasonal and residuals
from statsmodels.tsa.seasonal import seasonal_decompose

# Perform seasonal decomposition
decomposed = seasonal_decompose(energy_per_hour['Energy'], model='additive')

# Plot the original data
plt.figure(figsize=(12, 8))
plt.subplot(411)
plt.plot(energy_per_hour.index, energy_per_hour['Energy'], label='Original')
plt.legend(loc='upper left')

# Plot the trend component
plt.subplot(412)
plt.plot(energy_per_hour.index, decomposed.trend, label='Trend')
plt.legend(loc='upper left')

# Plot the seasonal component
plt.subplot(413)
plt.plot(energy_per_hour.index, decomposed.seasonal, label='Seasonal')
plt.legend(loc='upper left')

# Plot the residual component
plt.subplot(414)
plt.plot(energy_per_hour.index, decomposed.resid, label='Residual')
plt.legend(loc='upper left')

plt.tight_layout()
plt.show()


# In[92]:


from pmdarima.arima import auto_arima

arima_model = auto_arima(energy_per_hour['Energy'], start_p=1, d=1, start_q=1,
                        max_p=5, max_q=5, max_d=5, m=12,
                        start_P=0, D=1, start_Q=0, max_D=5, max_Q=5,
                        seasonal = True,
                        trace = True,
                        error_action = 'ignore',
                        suppress_warnings= True,
                        stepwise = True, n_fits=50)


# In[93]:


print(arima_model.summary())


# In[95]:


# Split data into train and test
size = int(len(energy_per_hour) * 0.66)
X_train, X_test = energy_per_hour[0:size], energy_per_hour[size:len(energy_per_hour)]


# In[99]:


# Fit a SARIMAX(3, 1, 0)x(2, 1, 0, 12) on the training dataset
from statsmodels.tsa.statespace.sarimax import SARIMAX

X_train.index.freq = X_train.index.inferred_freq

model = SARIMAX(X_train['Energy'],
               order = (3, 1, 1),
               seasonal_order = (2, 1, 0, 12))

result = model.fit()
result.summary()


# In[108]:


# Train prediction
start_index = 0
end_index = len(X_train)-1
train_prediction = result.predict(start_index, end_index)

# Prediction
start_index = len(X_train)
end_index = len(energy_per_hour)-1
prediction = result.predict(start_index, end_index).rename('Predicted Hourly Energy Consumption') 


# In[105]:


print(X_train)


# In[106]:


print(train_prediction)


# In[109]:


# Plot predictions and actual values
prediction.plot(legend=True)
X_test['Energy'].plot(legend=True)


# In[126]:


import math
from sklearn.metrics import mean_squared_error

# calculate root mean squared error
trainScore = math.sqrt(mean_squared_error(X_train, train_prediction))
print('Train Score: %.2f RMSE' % (trainScore))

testScore = math.sqrt(mean_squared_error(X_test, prediction))
print('Test Score: %.2f RMSE' % (testScore))


# In[117]:


# Forecast for the next 1 week
forecast = result.predict(start=len(energy_per_hour),
                          end=(len(energy_per_hour) - 1) + 14 * 24).rename('Forecast')

plt.figure(figsize=(12,8))
plt.plot(X_train, label='Training', color='green')
plt.plot(X_test, label='Test', color='yellow')
plt.plot(forecast, label='Forecast', color='cyan')
plt.legend(loc='upper left')
plt.show()


# In[119]:


# Save the all dataframe to local device
X_train.to_csv('/Users/wbm/Documents/Skilvul/Web App Energy Consumption/X_train.csv', index=True)
X_test.to_csv('/Users/wbm/Documents/Skilvul/Web App Energy Consumption/X_test.csv', index=True)
forecast.to_csv('/Users/wbm/Documents/Skilvul/Web App Energy Consumption/forecast.csv', index=True)



# In[120]:


print(X_train.head())


# In[121]:


print(X_test.head())


# In[122]:


print(forecast.head())


# In[ ]:




