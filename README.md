# ML-Timeseries-EnergyConsumption

Project Domain

This project is about energy management and optimization field that focuses on the need for efficient energy use across modern appliances. By gathering and analyzing energy usage data, the project tries to tackle the challenges associated with excessive energy consumption and aims to promote sustainable energy habits. 

Business Understanding

The main challenges addressed by this project include the lack of real-time insights into energy consumption patterns and the lack of existing solutions to dynamically predict and manage energy use. To achieve these objectives, I propose implementing a predictive model for forecasting based on historical data, which will facilitate proactive energy management. 

Data Understanding

The dataset utilized in this project is structured in a .csv format, containing key measurements from various appliances. It encompasses columns labeled 'Voltage', 'Ampere', 'Timestamp', and 'Device ID', with devices categorized into TV, AC, Refrigerator, PC, and Lamp. Data collection is conducted at 5-minute intervals which start from 2024-03-01 00:00:00 to 2024-03-07 23:55:00

Data Preparation

1	Renaming Columns: Initially, columns such as 'Ampere (A)' and 'Voltage (V)' are renamed to 'Ampere' and 'Voltage' for consistency and ease of reference in the code. Simplifying column names reduces the risk of errors in scripting and improves the readability of the code.
2	Timestamp Conversion: The 'Timestamp' column is converted from a string format to a datetime object. This transformation is important for time series analysis, as it enables the use of datetime-specific functions in pandas, such as resampling and time-based indexing. 
3	Feature Engineering:
•	Power Calculation: A new column, 'Power', is computed as the product of 'Voltage' and 'Ampere'. This step is important for understanding the energy dynamics since power measurement is a direct indicator of how much energy an appliance consumes per unit time.
•	Energy Calculation: Energy consumption over a 5-minute interval is calculated by multiplying the 'Power' by the time interval (in hours). This column ('Energy') quantifies the total energy used by an appliance.
4	Handling Missing Data: Any missing values in the dataset are identified and addressed. In this project, the code checks for missing data points and applies imputation techniques if necessary. The handling of missing data prevents the model from being biased or inaccurate due to incomplete information.
5	Removing Anomalies: The dataset is scrutinized for any anomalies, such as negative values in the 'Ampere' and 'Voltage' columns, which are physically implausible. Records with such values are removed to ensure the quality and reliability of the dataset. 
6	Feature Extraction
•	Time Features: Additional features like 'Hour', 'DayOfWeek', and 'IsWeekend' are derived from the 'Timestamp' to capture potential cyclic patterns in energy consumption. These features can help the model recognize and adjust to regular temporal variations in energy use.
•	Peak Hours Identification: A binary feature 'IsPeakHours' is created to indicate whether the measurement was taken during peak energy usage hours (defined as between 18:00 and 22:00). This feature is expected to assist in distinguishing periods of high energy demand
7	Stationarity Check Using Dickey-Fuller Test
To ensure the reliability of time-series models like ARIMA and SARIMAX, it's essential to verify that the data does not exhibit any unit root, meaning it should be stationary. The Dickey-Fuller test is employed to test the null hypothesis that the time series is non-stationary due to the presence of a unit root.  Given that the p-value is significantly low (less than 0.05), we reject the null hypothesis, concluding that the data is stationary.

Data Modelling
Algorithm Selection and Implementation:
ARIMA (AutoRegressive Integrated Moving Average):
Stages and Parameters:
The ARIMA model is first configured with initial parameters (p, d, q) that are identified based on the autocorrelation and partial autocorrelation plots. These parameters help the model understand the level, trend, and seasonality in the data.

SARIMAX (Seasonal AutoRegressive Integrated Moving-Average with eXogenous factors):
Stages and Parameters:
•	Similar to ARIMA, SARIMAX extends the capabilities by incorporating auside variables and seasonal components into the model.

I choose SARIMAX as the best solution for this project for several reasons:
Superior Performance: It outperforms the ARIMA model in terms of accuracy due to its ability to include seasonal patterns and external influences effectively.
Adaptability: It offers better adaptability to the nuances of energy consumption data which typically exhibit pronounced seasonal variations and are influenced by various external factors.

Evaluation

In this phase, the performance of the developed models is evaluated using the Root Mean Square Error (RMSE), which is a standard metric for assessing the accuracy of a model's predictions. The RMSE for the training dataset is 12.93. A score of 12.93 suggests that, on average, the model's predictions deviate from the actual values by approximately 12.93 units. The RMSE for the test dataset is 10.41. A test score of 10.41, which is lower than the training score, indicates that the model is generalizing well and not overfitting to the training data.

