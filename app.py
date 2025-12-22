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
            return float(lat_str), float(lon_str)
    except:
        return None

def classify_condition(cloud_cover, dew_point, temp, humidity, wind_speed, rain_prob, is_day=True):
    if rain_prob > 0.2:
        return "🌧"
    elif humidity > 90 and abs(temp - dew_point) < 2:
        return "⛆"
    elif wind_speed > 25:
        return "💨"
    elif cloud_cover < 20:
        return "☀" if is_day else "☾"
    elif cloud_cover < 40:
        return "🌤"
    elif cloud_cover < 60:
        return "☁"
    else:
        return "☁"

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
        day_avg["Humidity %"], day_avg["Wind Speed"], rain_probs[0], True)

    night_summary = classify_condition(
        night_avg["Cloud Cover %"], night_avg["Dew Point"], night_avg["Air Temperature (OC)"],
        night_avg["Humidity %"], night_avg["Wind Speed"], rain_probs[1], False)

    max_temp = df_day["Air Temperature (OC)"].max()
    min_temp = df_day["Air Temperature (OC)"].min()

    return day_summary, night_summary, max_temp, min_temp

def display_icons_row(df):
    icons = []
    for dt, row in df.iterrows():
        hour = dt.hour
        is_day = 6 <= hour <= 18
        icon = classify_condition(
            row["Cloud Cover %"], row["Air Dew Point"], row["Air Temperature (OC)"],
            row["Humidity %"], row["Wind Speed (MPS)"], 0, is_day
        )
        icons.append((hour, icon))

    icon_html = "<div style='display: flex; justify-content: space-between; font-size: 24px;'>"
    for hour, icon in icons:
        icon_html += f"<div style='flex: 1; text-align: center;'>{icon}</div>"
    icon_html += "</div>"
    st.markdown(icon_html, unsafe_allow_html=True)

st.set_page_config(layout="wide", page_title="Jawwak - Jordan Weather", page_icon="☀️")

st.markdown("""
    <style>
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
    </style>
""", unsafe_allow_html=True)

st.title(":sun_behind_cloud: Jawwak - Jordan Weather")

city_option = st.selectbox("Select location 📍", ["Use my current location"] + list(city_coords.keys()))

if city_option != "Use my current location":
    lat, lon = city_coords[city_option]
    coords = (lat, lon)
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
    station_name = get_nearest_station((coords[0], coords[1]))
    st.subheader(f"📍 Forecast for: {station_name}")

    with st.spinner("Fetching forecast..."):
        forecast = predict_full_forecast((coords[0], coords[1]))
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

            header_icon = day_summary
            with st.expander(f"Day {i+1} - {day_date.strftime('%A, %b %d')}: {header_icon}  High: {max_temp:.1f}°C | {night_summary} Low: {min_temp:.1f}°C", expanded=(i == 0)):
                summary_html = f"""
                <div style='background-color: #f0f2f6; padding: 1rem; border-radius: 12px; margin-bottom: 1rem; font-size: 16px;'>
                    <div style='margin-bottom: 0.5rem;'><strong>☀ Daytime:</strong> {day_summary} &nbsp;&nbsp; <strong>🌧️ Chance of Rain:</strong> {rain_probs[i][0]*100:.0f}%</div>
                    <div style='margin-bottom: 0.5rem;'><strong>☾ Nighttime:</strong> {night_summary} &nbsp;&nbsp; <strong>🌧️ Chance of Rain:</strong> {rain_probs[i][1]*100:.0f}%</div>
                    <div><strong>🌡️ Max:</strong> {max_temp:.1f}°C &nbsp;&nbsp; <strong>❄️ Min:</strong> {min_temp:.1f}°C</div>
                </div>
                """
                st.markdown(summary_html, unsafe_allow_html=True)

                def plot_interactive(df, column, ylabel):
                    fig = px.line(
                        df.reset_index(),
                        x="datetime",
                        y=column,
                        title=column,
                        labels={"index": "Time", column: ylabel},
                        height=250,
                        template="simple_white"
                    )
                    fig.update_traces(line=dict(color="#007AFF", width=3))
                    fig.update_layout(
                        margin=dict(l=20, r=20, t=30, b=20),
                        hovermode="x unified"
                    )
                    st.plotly_chart(fig, use_container_width=True)

                plot_interactive(day_df, "Air Temperature (OC)", "°C")
                display_icons_row(day_df)
                plot_interactive(day_df, "Humidity %", "%")
                plot_interactive(day_df, "Atmospheric Pressure", "hPa")
                plot_interactive(day_df, "Cloud Cover %", "%")
                plot_interactive(day_df, "Wind Speed (MPS)", "m/s")


