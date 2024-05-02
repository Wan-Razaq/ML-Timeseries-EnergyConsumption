import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np

def load_data():
    # Load data and prepare for plotting
    data = pd.read_csv('energy_per_hour_per_device.csv')
    data['Timestamp'] = pd.to_datetime(data['Timestamp'])
    # Aggregate data by device and timestamp
    aggregated_data = data.groupby(['Device', pd.Grouper(key='Timestamp', freq='D')])['Energy'].sum().reset_index()
    aggregated_data = aggregated_data.sort_values(by=['Timestamp', 'Energy'], ascending=[True, False])

    # Load prediction data
    train = pd.read_csv('X_train.csv', index_col='Timestamp', parse_dates=True)
    test = pd.read_csv('X_test.csv', index_col='Timestamp', parse_dates=True)
    forecast = pd.read_csv('forecast.csv', index_col='Timestamp', parse_dates=True, names=['Timestamp', 'Forecast'], header=0)
    return aggregated_data, train, test, forecast

device_energy, train_data, test_data, forecast_data = load_data()

# Prepare data for plotting
combined_data = pd.concat([train_data, test_data])
combined_data['Type'] = 'Actual'
forecast_data['Type'] = 'Forecast'
all_data = pd.concat([combined_data, forecast_data.rename(columns={'Forecast': 'Energy'})])

# Layout for prediction plot
col1, col2 = st.columns([1, 8])
with col1:
    start_date, end_date = st.date_input("Date range", [all_data.index.min(), all_data.index.max()])
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date) + pd.Timedelta(days=1)

filtered_data = all_data[(all_data.index >= start_date) & (all_data.index < end_date)]
with col2:
    st.title('Energy Consumption Prediction')
    # Create a scatter plot for actual data and forecasts
    fig_prediction = px.scatter(filtered_data, x=filtered_data.index, y='Energy', color='Type',
                                labels={"Energy": "Energy (Wh)", "Type": "Data Type", "index": "Timestamp"},
                                title='Energy Consumption: Actual vs Forecast')
    fig_prediction.update_traces(mode='markers')
    
    # Add a trend line using a rolling mean
    rolling_window = 7  # Change based on your data
    filtered_data['Trend'] = filtered_data['Energy'].rolling(window=rolling_window, min_periods=1, center=True).mean()
    fig_prediction.add_scatter(x=filtered_data.index, y=filtered_data['Trend'], mode='lines', name='Trend', line=dict(color='black'))
    
    st.plotly_chart(fig_prediction, use_container_width=True)

# Layout for device energy graph
col1, col2 = st.columns([1, 8])

# Define custom colors for devices
device_colors = {
    'AC': 'blue',         # Blue for AC
    'PC': 'red',          # Red for PC
    'Refrigerator': 'green', # Green for Refrigerator
    'TV': 'grey',         # Grey for TV
    'Lamp': 'yellow'      # Yellow for Lamp
}

with col1:
    # Ensure the selected date is within the available range
    if not device_energy.empty:
        min_date = device_energy['Timestamp'].min()
        max_date = device_energy['Timestamp'].max()
        selected_date = st.date_input("End Date", value=max_date, min_value=min_date, max_value=max_date)
        filtered_device_energy = device_energy[device_energy['Timestamp'] == pd.to_datetime(selected_date)]
    else:
        filtered_device_energy = pd.DataFrame()  # Empty DataFrame if no data available

filtered_device_energy['Formatted Energy'] = filtered_device_energy['Energy'].apply(lambda x: f"{x:.1f}")

with col2:
    st.title('Ranking of Devices by Energy Consumption')
    fig_device = px.bar(filtered_device_energy, x='Energy', y='Device', title='Ranking of Devices by Energy Consumption',
                        orientation='h', text='Formatted Energy', color='Device', color_discrete_map=device_colors)
    fig_device.update_traces(hovertemplate="<b>%{y}</b><br>Total Energy Consumption: %{x} Wh")
    fig_device.update_layout(xaxis_title='Total Energy Consumption (Wh)', yaxis_title='Device', showlegend=False)
    st.plotly_chart(fig_device, use_container_width=True)
