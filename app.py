import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime, timedelta
from Predict import predict_full_forecast, get_nearest_station
import requests
city_coords = {
    "Aqabah": (29.5500, 35.0000),
    "Ghor": (31.0333, 35.4667),
    "Irbid": (32.5500, 35.8500),
    "Irwaished": (32.5000, 38.2000),
    "Maan": (30.1667, 35.7833),
    "Mafraq": (32.3667, 36.2500),
    "Amman": (31.7167, 35.9833),
    "Safawi": (32.1608, 37.1539)
}
def get_user_location():
    try:
        ip_info = requests.get("https://ipinfo.io/json").json()
        loc = ip_info.get("loc", None)
        if loc:
            lat_str, lon_str = loc.split(",")
            return float(lon_str), float(lat_str)
    except:
        return None
# Helper functions

def classify_condition(cloud_cover, dew_point, temp, humidity, wind_speed, rain_prob):
    conditions = []

    if cloud_cover < 20:
        conditions.append("Sunny")
    elif cloud_cover < 40:
        conditions.append("Partly Cloudy")
    elif cloud_cover < 50:
        conditions.append("Mostly Cloudy")
    else:
        conditions.append("Overcast")

    if humidity > 90 and abs(temp - dew_point) < 2:
        conditions.append("Foggy")

    if rain_prob > 0.2:
        conditions.append("Rainy")

    if wind_speed > 20:
        conditions.append("Windy")

    return ", ".join(conditions)

def summarize_day(df_day, rain_probs):
    day_df = df_day[df_day.index.hour >= 6]
    night_df = pd.concat([df_day[df_day.index.hour < 6], df_day[df_day.index.hour >= 18]])

    def avg_conditions(df):
        return {
            "Cloud Cover %": df["Cloud Cover %"].mean(),
            "Air Temperature (OC)": df["Air Temperature (OC)"].mean(),
            "Humidity %": df["Humidity %"].mean(),
            "Dew Point": df["Air Dew Point"].mean(),
            "Wind Speed": df["Wind Speed (MPS)"].mean()
        }

    day_avg = avg_conditions(day_df)
    night_avg = avg_conditions(night_df)

    day_summary = classify_condition(
        day_avg["Cloud Cover %"], day_avg["Dew Point"], day_avg["Air Temperature (OC)"],
        day_avg["Humidity %"], day_avg["Wind Speed"], rain_probs[0])

    night_summary = classify_condition(
        night_avg["Cloud Cover %"], night_avg["Dew Point"], night_avg["Air Temperature (OC)"],
        night_avg["Humidity %"], night_avg["Wind Speed"], rain_probs[1])

    max_temp = df_day["Air Temperature (OC)"].max()
    min_temp = df_day["Air Temperature (OC)"].min()

    return day_summary, night_summary, max_temp, min_temp

# Streamlit App
st.set_page_config(layout="wide", page_title="5-Day Weather Forecast", page_icon="🌤️")
st.markdown("""
    <style>
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        .stButton>button {
            background-color: #007AFF;
            color: white;
        }
        .metric-label {
            font-weight: bold;
        }
    </style>
""", unsafe_allow_html=True)

st.title(":sun_behind_cloud: 5-Day Weather Forecast")

city_option = st.selectbox("Or select a city manually:", ["Use my current location"] + list(city_coords.keys()))

if city_option != "Use my current location":
    lat, lon = city_coords[city_option]
    coords = (lon, lat)
    st.success(f"Selected City: {city_option}")
else:
    coords = get_user_location()
    if coords:
        lat, lon = coords
        st.success(f"Detected Location: Latitude {lat:.4f}, Longitude {lon:.4f}")
    else:
        st.warning("Could not determine location automatically.")
        lat = st.number_input("Enter Latitude", format="%0.4f")
        lon = st.number_input("Enter Longitude", format="%0.4f")
if coords:
    lat, lon = coords
    st.success(f"Detected Location: Latitude {lat:.4f}, Longitude {lon:.4f}")
else:
    st.warning("Could not determine location automatically.")
    lat = st.number_input("Enter Latitude", format="%0.4f")
    lon = st.number_input("Enter Longitude", format="%0.4f")
station_name = get_nearest_station((coords[1], coords[0]))
st.subheader(f"📍 Forecast for: {station_name}")







if st.button("Get Forecast"):
    forecast = predict_full_forecast((lon, lat))
    df = forecast["time_series"]
    rain_probs = forecast["rain_probabilities"]

    df.index = pd.to_datetime(df.index)
    start_date = df.index[0].date()

    for i in range(5):
        day_date = start_date + timedelta(days=i)
        day_df = df[df.index.date == day_date]

        if day_df.empty:
            continue

        day_summary, night_summary, max_temp, min_temp = summarize_day(day_df, rain_probs[i])

        with st.expander(f"Day {i+1} - {day_date.strftime('%A, %b %d')}: ☀️ {day_summary} (High: {max_temp:.1f}°C) | 🌙 {night_summary} (Low: {min_temp:.1f}°C)"):

            def plot_interactive(df, column, ylabel):
                fig = px.line(
                    df.reset_index(),
                    x="datetime",
                    y=column,
                    labels={"index": "Time", column: ylabel},
                    title=column,
                )
                fig.update_layout(
                    height=250,
                    margin=dict(l=20, r=20, t=30, b=20),
                    template="simple_white",
                    hovermode="x unified",
                    xaxis=dict(fixedrange=True),
                    yaxis=dict(fixedrange=True),
                )
                fig.update_traces(line=dict(color="#007AFF", width=3))
                st.plotly_chart(fig, use_container_width=True)

            plot_interactive(day_df, "Air Temperature (OC)", "°C")
            plot_interactive(day_df, "Humidity %", "%")
            plot_interactive(day_df, "Atmospheric Pressure", "hPa")
            plot_interactive(day_df, "Cloud Cover %", "%")
            plot_interactive(day_df, "Wind Speed (MPS)", "m/s")

            st.dataframe(day_df.style.format("{:.2f}"))