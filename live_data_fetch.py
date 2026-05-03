#!/usr/bin/env python
# coding: utf-8

# In[4]:


import requests
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

# Fetch live data
url = "https://api.open-meteo.com/v1/forecast?latitude=12.97&longitude=77.59&hourly=temperature_2m"
response = requests.get(url)
data = response.json()

time = data["hourly"]["time"]
temp = data["hourly"]["temperature_2m"]

df = pd.DataFrame({"Datetime": time, "Consumption": temp})

# Convert datetime
df["Datetime"] = pd.to_datetime(df["Datetime"])

df.set_index("Datetime", inplace=True)
df = df.asfreq("h")

# Forecast
model = ARIMA(df["Consumption"], order=(2, 1, 2))
model_fit = model.fit()

forecast = model_fit.forecast(steps=24)

# Save
df.reset_index(inplace=True)
df.to_csv(
    r"C:\Users\shrey\OneDrive\Desktop\Energy_Consumption_optimization\live_data.csv",
    index=False,
)

# Create forecast DataFrame with proper datetime index
forecast_df = pd.DataFrame(
    {
        "Datetime": pd.date_range(start=df["Datetime"].iloc[-1], periods=24, freq="h"),
        "Forecast": forecast.values,
    }
)

forecast_df.to_csv(
    r"C:\Users\shrey\OneDrive\Desktop\Energy_Consumption_optimization\forecast.csv",
    index=False,
)
print("Saved here:", r"C:\Users\shrey\OneDrive\Desktop\Energy_Consumption_optimization")

print("Data updated successfully!")


# In[5]:


# In[ ]:
