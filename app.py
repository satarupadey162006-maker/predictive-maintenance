import pandas as pd
import streamlit as st
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

st.set_page_config(page_title="Predictive Maintenance", layout="wide")
st.title("Predictive Maintenance Dashboard")

features = ["air_temp", "process_temp", "rpm", "torque", "tool_wear"]

@st.cache_data
def load_data():
    df = pd.read_csv("ai4i2020.csv")
    return df.rename(columns={
        "Air temperature [K]": "air_temp",
        "Process temperature [K]": "process_temp",
        "Rotational speed [rpm]": "rpm",
        "Torque [Nm]": "torque",
        "Tool wear [min]": "tool_wear",
        "Machine failure": "failure",
    })

@st.cache_resource
def train(df):
    X_train, X_test, y_train, y_test = train_test_split(
        df[features], df["failure"], test_size=0.2, random_state=42, stratify=df["failure"]
    )
    model = RandomForestClassifier(class_weight="balanced", random_state=42)
    model.fit(X_train, y_train)
    return model, X_test, y_test

df = load_data()
model, X_test, y_test = train(df)
pred = model.predict(X_test)
tn, fp, fn, tp = [int(v) for v in confusion_matrix(y_test, pred).ravel()]

# Cost assumptions the judges can change
st.sidebar.header("Cost assumptions (Rs)")
breakdown = st.sidebar.number_input("Surprise breakdown cost", value=50000, step=1000)
planned = st.sidebar.number_input("Planned repair cost", value=5000, step=500)
false_alarm = st.sidebar.number_input("False alarm inspection cost", value=2000, step=500)

cost_without = (tp + fn) * breakdown
cost_with = tp * planned + fn * breakdown + fp * false_alarm
saving = cost_without - cost_with

c1, c2, c3 = st.columns(3)
c1.metric("Money saved", f"Rs {saving:,}")
c2.metric("Failures caught", f"{tp} of {tp + fn}")
c3.metric("False alarms", fp)

# Machine risk table
risk = X_test.copy()
risk["failure_probability"] = model.predict_proba(X_test)[:, 1].round(2)
risk["status"] = pd.cut(
    risk["failure_probability"], bins=[-0.01, 0.3, 0.6, 1.0],
    labels=["Green", "Amber", "Red"]
)

st.subheader("Highest-risk machines")
st.dataframe(risk.sort_values("failure_probability", ascending=False).head(20))

st.subheader("Why machines fail: torque vs tool wear")
plot_df = X_test.copy()
plot_df["actual_failure"] = y_test.astype(str)
st.plotly_chart(
    px.scatter(plot_df, x="tool_wear", y="torque", color="actual_failure"),
    use_container_width=True,
)