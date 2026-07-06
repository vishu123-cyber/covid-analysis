import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------
st.set_page_config(
    page_title="COVID-19 Dashboard",
    layout="wide"
)

st.title("COVID-19 Data Analysis and Visualization Dashboard")
st.write("""
This dashboard analyzes COVID-19 data and provides insights into
confirmed cases, deaths, recoveries, rates, trends, and prediction.
""")

# ---------------------------------------------------
# LOAD DATA
# ---------------------------------------------------
@st.cache_data
def load_data():
    current_dir = os.path.dirname(os.path.abspath(__file__))

    possible_country_paths = [
        os.path.join(current_dir, "data", "country_wise_latest.csv"),
        os.path.join(current_dir, "..", "data", "country_wise_latest.csv"),
        os.path.join(current_dir, "country_wise_latest.csv")
    ]

    possible_time_paths = [
        os.path.join(current_dir, "data", "time_series_covid19_confirmed_global.csv"),
        os.path.join(current_dir, "..", "data", "time_series_covid19_confirmed_global.csv"),
        os.path.join(current_dir, "time_series_covid19_confirmed_global.csv")
    ]

    country_path = None
    time_path = None

    for path in possible_country_paths:
        if os.path.exists(path):
            country_path = path
            break

    for path in possible_time_paths:
        if os.path.exists(path):
            time_path = path
            break

    if country_path is None:
        st.error("country_wise_latest.csv file not found.")
        st.stop()

    if time_path is None:
        st.error("time_series_covid19_confirmed_global.csv file not found.")
        st.stop()

    data = pd.read_csv(country_path)
    ts_data = pd.read_csv(time_path)

    return data, ts_data


data, ts_data = load_data()

# ---------------------------------------------------
# DATA CLEANING
# ---------------------------------------------------
data.replace([np.inf, -np.inf], np.nan, inplace=True)
data.fillna(0, inplace=True)

data["Death Rate"] = np.where(
    data["Confirmed"] > 0,
    (data["Deaths"] / data["Confirmed"]) * 100,
    0
)

data["Recovery Rate"] = np.where(
    data["Confirmed"] > 0,
    (data["Recovered"] / data["Confirmed"]) * 100,
    0
)

data["Active"] = np.where(
    data["Confirmed"] - data["Deaths"] - data["Recovered"] >= 0,
    data["Confirmed"] - data["Deaths"] - data["Recovered"],
    0
)

# ---------------------------------------------------
# SIDEBAR
# ---------------------------------------------------
st.sidebar.header("Filters")

metric = st.sidebar.selectbox(
    "Select Metric",
    ["Confirmed", "Deaths", "Recovered", "Active", "Death Rate", "Recovery Rate"]
)

top_n = st.sidebar.slider("Select Top N Countries", 5, 20, 10)

selected_countries = st.sidebar.multiselect(
    "Select Countries for Comparison",
    options=sorted(data["Country/Region"].unique()),
    default=[]
)

pie_countries = st.sidebar.multiselect(
    "Select 3 or 4 Countries for Pie Chart",
    options=sorted(data["Country/Region"].unique()),
    default=[],
    max_selections=4
)

ma_window = st.sidebar.slider("Moving Average Window (days)", 3, 14, 5)

# ---------------------------------------------------
# DATASET PREVIEW
# ---------------------------------------------------
st.subheader("Dataset Preview")
st.dataframe(data.head())

# ---------------------------------------------------
# GLOBAL STATS
# ---------------------------------------------------
st.subheader("Global COVID-19 Statistics")

total_cases = int(data["Confirmed"].sum())
total_deaths = int(data["Deaths"].sum())
total_recovered = int(data["Recovered"].sum())

col1, col2, col3 = st.columns(3)
col1.metric("Total Confirmed Cases", f"{total_cases:,}")
col2.metric("Total Deaths", f"{total_deaths:,}")
col3.metric("Total Recovered", f"{total_recovered:,}")

# ---------------------------------------------------
# TOP COUNTRIES BAR CHART
# ---------------------------------------------------
st.subheader(f"Top {top_n} Countries by {metric}")

top_countries = data.sort_values(by=metric, ascending=False).head(top_n)
st.dataframe(top_countries[["Country/Region", metric]])

fig1, ax1 = plt.subplots(figsize=(10, 5))
sns.barplot(
    x=metric,
    y="Country/Region",
    hue="Country/Region",
    data=top_countries,
    palette="viridis",
    legend=False,
    ax=ax1
)
ax1.set_title(f"Top {top_n} Countries by {metric}")
ax1.set_xlabel(metric)
ax1.set_ylabel("Country")
plt.tight_layout()
st.pyplot(fig1)

# ---------------------------------------------------
# CORRELATION HEATMAP
# ---------------------------------------------------
st.subheader("Correlation Heatmap")

corr = data[["Confirmed", "Deaths", "Recovered", "Active", "Death Rate", "Recovery Rate"]].corr()

fig2, ax2 = plt.subplots(figsize=(8, 6))
sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax2)
ax2.set_title("Feature Correlation Heatmap")
plt.tight_layout()
st.pyplot(fig2)

# ---------------------------------------------------
# MOVING AVERAGE TREND
# ---------------------------------------------------
st.subheader("Moving Average Trend Analysis")

ts_clean = ts_data.drop(columns=["Province/State", "Country/Region", "Lat", "Long"], errors="ignore")
global_trend = ts_clean.sum()

trend = pd.DataFrame({
    "Date": pd.to_datetime(global_trend.index, errors="coerce"),
    "Confirmed": global_trend.values
})

trend.dropna(inplace=True)
trend["Moving Average"] = trend["Confirmed"].rolling(window=ma_window).mean()

fig3, ax3 = plt.subplots(figsize=(10, 5))
ax3.plot(trend["Date"], trend["Confirmed"], label="Actual Confirmed Cases")
ax3.plot(trend["Date"], trend["Moving Average"], label=f"{ma_window}-Day Moving Average")
ax3.set_title("Global Confirmed Cases Trend")
ax3.set_xlabel("Date")
ax3.set_ylabel("Confirmed Cases")
ax3.legend()
plt.xticks(rotation=45)
plt.tight_layout()
st.pyplot(fig3)

# ---------------------------------------------------
# PIE CHART
# ---------------------------------------------------
st.subheader(f"Distribution of {metric} for Pie Chart Countries")

if len(pie_countries) == 0:
    st.info("Please select 3 or 4 countries from the sidebar for the pie chart.")
elif len(pie_countries) < 3:
    st.warning("Please select at least 3 countries.")
else:
    pie_data = data[data["Country/Region"].isin(pie_countries)][["Country/Region", metric]].copy()
    pie_data = pie_data[pie_data[metric] > 0]

    if pie_data.empty:
        st.warning(f"No non-zero values available for {metric}.")
    else:
        fig4, ax4 = plt.subplots(figsize=(10, 6))

        def show_pct(pct):
            return f"{pct:.1f}%" if pct >= 2 else ""

        wedges, _, autotexts = ax4.pie(
            pie_data[metric],
            labels=None,
            autopct=show_pct,
            startangle=90,
            pctdistance=0.75,
            wedgeprops=dict(width=0.45, edgecolor="white")
        )

        ax4.set_title(f"{metric} Distribution")

        ax4.legend(
            wedges,
            pie_data["Country/Region"],
            title="Countries",
            loc="center left",
            bbox_to_anchor=(1.05, 0.5)
        )

        ax4.set_aspect("equal")
        plt.tight_layout()
        st.pyplot(fig4)

# ---------------------------------------------------
# SELECTED COUNTRIES COMPARISON
# ---------------------------------------------------
st.subheader(f"Selected Countries Comparison by {metric}")

if len(selected_countries) == 0:
    st.info("Please select one or more countries from the sidebar.")
else:
    chart_data = data[data["Country/Region"].isin(selected_countries)].copy()
    chart_data = chart_data.sort_values(by=metric, ascending=False).head(10)

    fig5, ax5 = plt.subplots(figsize=(12, 6))
    sns.barplot(
        x="Country/Region",
        y=metric,
        hue="Country/Region",
        data=chart_data,
        palette="magma",
        legend=False,
        ax=ax5
    )
    ax5.set_title(f"Selected Countries by {metric}")
    ax5.set_xlabel("Country")
    ax5.set_ylabel(metric)
    plt.xticks(rotation=30)
    plt.tight_layout()
    st.pyplot(fig5)

# ---------------------------------------------------
# SCATTER PLOT
# ---------------------------------------------------
st.subheader("Confirmed Cases vs Deaths")

fig6, ax6 = plt.subplots(figsize=(8, 6))
sns.scatterplot(x="Confirmed", y="Deaths", data=data, ax=ax6)
ax6.set_title("Confirmed Cases vs Deaths")
ax6.set_xlabel("Confirmed Cases")
ax6.set_ylabel("Deaths")
plt.tight_layout()
st.pyplot(fig6)

st.write("""
This scatter plot shows the relationship between confirmed cases and deaths.
In general, countries with more confirmed cases tend to have more deaths.
""")

# ---------------------------------------------------
# LINEAR REGRESSION MODEL
# ---------------------------------------------------
st.subheader("Linear Regression Model")

X = data[["Confirmed"]]
y = data["Deaths"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

st.success("Model trained successfully!")

# ---------------------------------------------------
# PREDICTION SECTION
# ---------------------------------------------------
st.subheader("Predict Deaths from Confirmed Cases")

user_input = st.number_input(
    "Enter Confirmed Cases for Prediction",
    min_value=0.0,
    value=100000.0,
    step=1000.0
)

input_data = pd.DataFrame([[user_input]], columns=["Confirmed"])
predicted_deaths = model.predict(input_data)[0]
predicted_deaths = max(0, predicted_deaths)

if predicted_deaths > user_input:
    predicted_deaths = user_input

st.write(f"### Predicted Deaths: {round(predicted_deaths)}")

# ---------------------------------------------------
# REGRESSION LINE PLOT
# ---------------------------------------------------
st.subheader("Regression Line: Confirmed Cases vs Deaths")

fig_reg, ax_reg = plt.subplots(figsize=(8, 6))
ax_reg.scatter(data["Confirmed"], data["Deaths"], alpha=0.6, label="Actual Data")

sorted_data = data.sort_values("Confirmed")
predicted_line = model.predict(sorted_data[["Confirmed"]])

ax_reg.plot(
    sorted_data["Confirmed"],
    predicted_line,
    color="red",
    label="Regression Line"
)

ax_reg.set_title("Confirmed Cases vs Deaths with Regression Line")
ax_reg.set_xlabel("Confirmed Cases")
ax_reg.set_ylabel("Deaths")
ax_reg.legend()
plt.tight_layout()
st.pyplot(fig_reg)

# ---------------------------------------------------
# ACTUAL VS PREDICTED
# ---------------------------------------------------
st.subheader("Actual vs Predicted Deaths")

y_pred = model.predict(X_test)
y_pred = np.maximum(y_pred, 0)

fig7, ax7 = plt.subplots(figsize=(8, 6))
ax7.scatter(y_test, y_pred, alpha=0.7)

# Ideal reference line
line_min = min(y_test.min(), y_pred.min())
line_max = max(y_test.max(), y_pred.max())

ax7.plot(
    [line_min, line_max],
    [line_min, line_max],
    color="red",
    linestyle="--",
    label="Ideal Line"
)

ax7.set_title("Actual vs Predicted Deaths")
ax7.set_xlabel("Actual Deaths")
ax7.set_ylabel("Predicted Deaths")
ax7.legend()
plt.tight_layout()
st.pyplot(fig7)

# ---------------------------------------------------
# MODEL PERFORMANCE
# ---------------------------------------------------
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

st.write(f"**Model R² Score:** {r2:.4f}")
st.write(f"**Mean Absolute Error:** {mae:.2f}")

if r2 < 0.5:
    st.warning("The model accuracy is not very high. It shows general trend but prediction is limited.")
else:
    st.success("The model shows reasonably good predictive performance.")

# ---------------------------------------------------
# FOOTER
# ---------------------------------------------------
st.markdown("---")
st.write("COVID-19 Data Analysis Project using Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, and Streamlit")