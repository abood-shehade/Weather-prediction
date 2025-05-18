import joblib
import torch 
import pandas as pd
import numpy as np
from metpy.calc import wind_components
from metpy.units import units
import openmeteo_requests
import requests_cache
from math import radians, sin, cos, sqrt, atan2
from torch.utils.data import Dataset, DataLoader
from transformers import PatchTSTForPrediction, PatchTSTConfig
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib 
import matplotlib.pyplot as plt
from darts import TimeSeries
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import os
import datetime as dt
import pytz
from retry_requests import retry
from datetime import timedelta, datetime

import torch.nn as nn
from tensorflow.keras.models import load_model
from tsfm_public import (
    TimeSeriesForecastingPipeline,
    TinyTimeMixerForPrediction,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device

def get_ordered_hourly_variables(params):
    return params.get("hourly", [])

#setting up API client
cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
openmeteo = openmeteo_requests.Client(session=retry_session)

def get_reading(url, latitude, longtitute, hourly, past_days, future_days):
    params = {
        "latitude": latitude,
        "longitude": longtitute,
        "hourly": hourly,
        "past_days": past_days,
        "forecast_days": future_days
    }

    response = openmeteo.weather_api(url, params=params)[0]
    hourly = response.Hourly()

    # Convert from UTC to UTC+3
    start_time = pd.to_datetime(hourly.Time(), unit="s", utc=True) + pd.Timedelta(hours=3)
    end_time = pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True) + pd.Timedelta(hours=3)

    hourly_data = {
        "date": pd.date_range(
            start=start_time,
            end=end_time,
            freq=pd.Timedelta(seconds=hourly.Interval()),
            inclusive="left"
        )
    }

    for i, var_name in enumerate(get_ordered_hourly_variables(params)):
        hourly_data[var_name] = hourly.Variables(i).ValuesAsNumpy()

    hourly_df = pd.DataFrame(data=hourly_data)
    hourly_df['Date/Time'] = pd.to_datetime(hourly_df['date'], errors='coerce')
    hourly_df.set_index('Date/Time', inplace=True)
    hourly_df.drop(columns=['date'], inplace=True)


    hourly_df.rename(columns={
        "dew_point_2m": "Air Dew Point",
        "temperature_2m": "Air Temperature (OC)",
        "relative_humidity_2m": "Humidity %",
        "pressure_msl": "Atmospheric Pressure",
        "wind_speed_10m": "Wind Speed (MPS)",
        "wind_direction_10m": "Wind Direction (Degrees)"
    }, inplace=True)

    hourly_df.index = hourly_df.index.tz_convert(None)

    u, v = wind_components(
        hourly_df['Wind Speed (MPS)'].values * units('m/s'),
        hourly_df['Wind Direction (Degrees)'].values * units.degree
    )

    hourly_df['Wind_U'] = u.magnitude  
    hourly_df['Wind_V'] = v.magnitude
    hourly_df.Wind_U = hourly_df.Wind_U.round(2)
    hourly_df.Wind_V = hourly_df.Wind_V.round(2)

    hourly_df.drop(columns=["Wind Speed (MPS)", "Wind Direction (Degrees)"], inplace=True)
    hourly_df = hourly_df[['Air Dew Point', 'Air Temperature (OC)', 'Humidity %', 'Atmospheric Pressure', 'Wind_U', 'Wind_V']]
    
    return hourly_df

def get_realtime_512hours(api_url, latitude, longitude, variables):
    """
    Fetches the latest 512 hours of weather data up to the current time using get_reading().

    Parameters:
    - api_url: URL of the weather API
    - latitude: float, location latitude
    - longitude: float, location longitude
    - variables: list of weather variables to fetch
    - timezone_offset_hours: int, for formatting timestamps correctly
    - gmt_offset: int, offset from UTC in hours (e.g., 0 for GMT, 3 for GMT+3)

    Returns:
    - DataFrame with last 512 hourly weather records up to current time
    """
    # Calculate the full time window: 23 days past + 1 day future = 24 days
    now_utc = datetime.utcnow()
    start_date = now_utc - timedelta(days=23)
    end_date = now_utc + timedelta(days=1)


    # Call get_reading() with computed dates
    df = get_reading(
        api_url,
        latitude,
        longitude,
        variables,
        23,
        1
    )

    # Ensure datetime index
    df.index = pd.to_datetime(df.index)

    # Filter last 512 hours up to now (adjusting for local time)
    now_local = datetime.utcnow()
    df = df[df.index <= now_local].tail(512)

    return df


def reconstruct_wind_speed_direction(df):
    """
    Given a DataFrame with Wind_U and Wind_V columns,
    returns two new columns: Wind Speed (MPS) and Wind Direction (Degrees)
    """
    df = df.copy()
    
    # Compute wind speed
    df['Wind Speed (MPS)'] = np.sqrt(df['Wind_U']**2 + df['Wind_V']**2)
    
    # Compute wind direction in meteorological convention:
    # direction from which wind is blowing (0 = North, 90 = East, 180 = South, etc.)
    df['Wind Direction (Degrees)'] = (270 - np.degrees(np.arctan2(df['Wind_V'], df['Wind_U']))) % 360

    return df



class RainLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2):
        super(RainLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()  
        )

    def forward(self, x):
        _, (hn, _) = self.lstm(x) 
        return self.fc(hn[-1])  


def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat / 2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2)**2
    return R * 2 * atan2(sqrt(a), sqrt(1 - a))

def get_nearest_station(location):
    names = ['aqaba', 'ghor', 'irbid', 'irwaished', 'maan', 'mafraq', 'amman', 'safawi']
 
    coordinates = [            # (latitude, longitude)
    (29.5500, 35.0000),    # Aqabah
    (31.0333, 35.4667),    # Ghor
    (32.5500, 35.8500),    # Irbid
    (32.5000, 38.2000),    # Irwaished
    (30.1667, 35.7833),    # Maan
    (32.3667, 36.2500),    # Mafraq
    (31.7167, 35.9833),    # Amman
    (32.1608, 37.1539),    # Safawi
    ]
    lat, lon = location
    distances = []
    for stat_lat, stat_lon in coordinates:
        distances.append(haversine(lat, lon, stat_lat, stat_lon))
    closest_idx = int(np.argmin(distances))
    return names[closest_idx]


def predict_rain_probability(location, df_48h, models_dir="Models/Precipitation_Models"):
    
    input_features = [
    'Air Dew Point', 'Air Temperature (OC)', 'Humidity %',
    'Atmospheric Pressure', 'Cloud Cover %', 'Wind_U', 'Wind_V',
    'hour_sin', 'hour_cos', 'day_sin', 'day_cos'
    ]
    
    #Find closest station
    station = get_nearest_station(location)

    #Load model and scaler
    scaler = joblib.load(f"{models_dir}/{station}_scaler.pkl")
    model = RainLSTM(input_size=11)
    model.load_state_dict(torch.load(f"{models_dir}/{station}_precipitation_model.pth", map_location=torch.device('cpu')))
    model.eval()

    #preprocessing 
    df = df_48h.copy()
    df['hour'] = df['time'].dt.hour
    df['day_yr'] = df['time'].dt.dayofyear
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['day_sin'] = np.sin(2 * np.pi * df['day_yr'] / 365)
    df['day_cos'] = np.cos(2 * np.pi * df['day_yr'] / 365)

    df[input_features] = scaler.transform(df[input_features])
    input_data = df[input_features].values.astype(np.float32)

    #Padding just in case
    if len(input_data) < 48:
        pad_len = 48 - len(input_data)
        padding = np.repeat(input_data[-1:], pad_len, axis=0)
        input_data = np.vstack([input_data, padding])

    input_tensor = torch.tensor(input_data).unsqueeze(0)  #shape (1, 48, 11)

    with torch.no_grad():
        prob = model(input_tensor).item()

    return {
        "closest_station": station,
        "precipitation_probability": round(prob, 4)
    }

def predict_cloud_cover(location, df_48h):
    names = ['aqaba', 'ghor', 'irbid', 'irwaished', 'maan', 'mafraq', 'amman', 'safawi']
    coordinates = [
        (29.5500, 35.0000), (31.0333, 35.4667), (32.5500, 35.8500), (32.5000, 38.2000),
        (30.1667, 35.7833), (32.3667, 36.2500), (31.7167, 35.9833), (32.1608, 37.1539),
    ]
    
    station = get_nearest_station(location)
    input_scaler = joblib.load(f"Models/Cloud_Cover_Models/{station}_input_scaler.save")
    target_scaler = joblib.load(f"Models/Cloud_Cover_Models/{station}_target_scaler.save")
    model = load_model(f"Models/Cloud_Cover_Models/{station}_cloud_model.keras")

    #Preprocess input dataframe
    df = df_48h.copy()
   

    df['hour'] = df['time'].dt.hour
    df['month'] = df['time'].dt.month
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

    input_features = ['Air Dew Point', 'Air Temperature (OC)', 'Humidity %',
                      'Atmospheric Pressure', 'Wind_U', 'Wind_V']
    X_input = input_scaler.transform(df[input_features])
    X_input = np.expand_dims(X_input, axis=0)  # Shape: (1, 48, 6)

    y_pred_scaled = model.predict(X_input)
    y_pred = target_scaler.inverse_transform(y_pred_scaled)
    y_pred = np.clip(y_pred, a_min=0, a_max=None)
    return y_pred.flatten()  # Shape: (24,)


def Cloud_and_Precipitation(df_120h, location):
    all_cloud_preds = []
    #Predict cloud first
    for step in [0, 24, 48, 72, 96]:
        df_window = df_120h.iloc[step:step + 48].copy()
        cloud_preds = predict_cloud_cover(location, df_window)  
        all_cloud_preds.extend(cloud_preds)


    assert len(all_cloud_preds) == 120, f"Expected 120 cloud preds, got {len(all_cloud_preds)}"


    df_with_cloud = df_120h.copy()
    df_with_cloud['Cloud Cover %'] = 0.0  # ensure float type
    cloud_col_index = df_with_cloud.columns.get_loc('Cloud Cover %')


    df_with_cloud.iloc[-120:, cloud_col_index] = all_cloud_preds

    #predict rain probs
    rain_probs = [] 

    for step in range(0, 109, 12):  # steps: 0, 12, ..., 72
        df_window = df_with_cloud.iloc[step:step + 48].copy()
        prob = predict_rain_probability(location, df_window)["precipitation_probability"]
        rain_probs.append(prob)

    
    day_night_tuples = [
        (rain_probs[i], rain_probs[i + 1]) for i in range(0, len(rain_probs), 2)
    ]

    return {
        "cloud_cover": all_cloud_preds,           #120 values (5x24h)
        "rain_probabilities": day_night_tuples    #5 tuples (day, night)
    }


def predict_TTM(df_512, station):
    # Constants
    target_columns = ["Air Dew Point", "Air Temperature (OC)", "Humidity %", 
                      "Atmospheric Pressure", "Wind_U", "Wind_V"]
    context_length = 512
    prediction_length = 96
    df_512["Unnamed: 0"] = df_512.index

    # Paths
    BASE_DIR = f'TTM Model/{station.replace(" ", "_")}/eval'
    checkpoint_dir = next(d for d in os.listdir(os.path.join(BASE_DIR, "output")) if d.startswith("checkpoint-"))
    CHECKPOINT_PATH = os.path.join(BASE_DIR, "output", checkpoint_dir)
    PREPROCESSOR_PATH = os.path.join(BASE_DIR, "trained_preprocessor.joblib")

    # Load preprocessor and model
    tsp = joblib.load(PREPROCESSOR_PATH)
    model = TinyTimeMixerForPrediction.from_pretrained(CHECKPOINT_PATH)

    # Create inference pipeline
    inference_pipeline = TimeSeriesForecastingPipeline(
        model=model,
        device="cuda" if torch.cuda.is_available() else "cpu",
        feature_extractor=tsp,
        batch_size=512,
        freq="h"
    )

    # ---- First prediction (96h) ----
    forecast_df1 = inference_pipeline(df_512)
    prediction_columns = [col for col in forecast_df1.columns if col.endswith('_prediction')]
    predictions1 = {col.replace('_prediction', ''): forecast_df1[col].iloc[0] for col in prediction_columns}
    start_time = pd.to_datetime(df_512["Unnamed: 0"].iloc[-1]) + pd.Timedelta(hours=1)
    prediction_index1 = pd.date_range(start=start_time, periods=96, freq="h")
    forecast_output1 = pd.DataFrame(predictions1, index=prediction_index1)

    # ---- Second prediction (24h) ----
    # Extend the input with the first prediction to get the next 24h
    df_extended = pd.concat([df_512[target_columns], forecast_output1], axis=0)
    df_extended = df_extended.iloc[-512:]
    df_extended["Unnamed: 0"] = df_extended.index

    forecast_df2 = inference_pipeline(df_extended)
    predictions2 = {col.replace('_prediction', ''): forecast_df2[col].iloc[0][:24] for col in prediction_columns}
    prediction_index2 = pd.date_range(start=prediction_index1[-1] + pd.Timedelta(hours=1), periods=24, freq="h")
    forecast_output2 = pd.DataFrame(predictions2, index=prediction_index2)

    # Combine both
    final_forecast = pd.concat([forecast_output1, forecast_output2])
    final_forecast.index.name = "datetime"

    return final_forecast


def predict_full_forecast(location: tuple):
    station = get_nearest_station(location)

    # past_data = get_reading("https://api.open-meteo.com/v1/forecast",
    #         location[0], location[1],
    #         ["temperature_2m", "dew_point_2m", "relative_humidity_2m",
    #        "pressure_msl", "wind_speed_10m", "wind_direction_10m"], 4,
    #         0)
    past_data = get_realtime_512hours("https://api.open-meteo.com/v1/forecast",
            location[0], location[1],
            ["temperature_2m", "dew_point_2m", "relative_humidity_2m",
           "pressure_msl", "wind_speed_10m", "wind_direction_10m"])
    # past_data = past_data.tail(72)
    TTM_preds = predict_TTM(past_data, station)
    input_with_time = TTM_preds.copy()
    input_with_time['time'] = TTM_preds.index
    cloud_precipitation = Cloud_and_Precipitation(input_with_time.tail(120),location)

    time_series_outputs = TTM_preds
    time_series_outputs['Cloud Cover %'] = cloud_precipitation['cloud_cover']

    final_df = reconstruct_wind_speed_direction(time_series_outputs)
    final_df.drop(['Wind_U', 'Wind_V'], axis = 1, inplace = True)
    forecast = {
        "time_series" : final_df,
        'rain_probabilities' : cloud_precipitation['rain_probabilities']
    }
    
    return forecast