import time
import pandas as pd
import streamlit as st
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, recall_score, precision_score, f1_score

st.set_page_config(page_title="Predictive Maintenance", layout="wide")
st.title("Predictive Maintenance Dashboard")
st.caption("Predict machine failures before they happen and quantify the savings.")

features = ["air_temp", "process_temp", "rpm", "torque", "tool_wear", "temp_diff", "power"]


@st.cache_data
def load_data():
    df = pd.read_csv("ai4i2020.csv")
    df = df.rename(columns={
        "Air temperature [K]": "air_temp",
        "Process temperature [K]": "process_temp",
        "Rotational speed [rpm]": "rpm",
        "Torque [Nm]": "torque",
        "Tool wear [min]": "tool_wear",
        "Machine failure": "failure",
    })
    df["temp_diff"] = df["process_temp"] - df["air_temp"]
    df["power"] = df["torque"] * df["rpm"]
    return df


@st.cache_resource
def train_models(df):
    X_train, X_test, y_train, y_test = train_test_split(
        df[features], df["failure"], test_size=0.2, random_state=42, stratify=df["failure"]
    )
    models = {
        "Random Forest": RandomForestClassifier(class_weight="balanced", random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
        "Logistic Regression": make_pipeline(
            StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=1000)
        ),
    }
    for m in models.values():
        m.fit(X_train, y_train)
    return models, X_test, y_test


df = load_data()
models, X_test, y_test = train_models(df)

# ---------- Sidebar ----------
st.sidebar.header("Model")
model_name = st.sidebar.selectbox("Choose model", list(models.keys()))
model = models[model_name]

st.sidebar.header("Alert sensitivity")
threshold = st.sidebar.slider("Alert threshold", 0.1, 0.9, 0.5, 0.05)

st.sidebar.header("Cost assumptions (Rs)")
breakdown = st.sidebar.number_input("Surprise breakdown cost", value=50000, step=1000)
planned = st.sidebar.number_input("Planned repair cost", value=5000, step=500)
false_alarm = st.sidebar.number_input("False alarm inspection cost", value=2000, step=500)

# ---------- Predictions and savings ----------
probs = model.predict_proba(X_test)[:, 1]
pred = (probs >= threshold).astype(int)
tn, fp, fn, tp = [int(v) for v in confusion_matrix(y_test, pred).ravel()]

cost_without = (tp + fn) * breakdown
cost_with = tp * planned + fn * breakdown + fp * false_alarm
saving = cost_without - cost_with
# Best threshold for maximum savings
best_t, best_saving = None, None
for t in [x / 100 for x in range(10, 91, 5)]:
    p_t = (probs >= t).astype(int)
    a, b, c, d = [int(v) for v in confusion_matrix(y_test, p_t).ravel()]  # tn, fp, fn, tp
    s = (c + d) * breakdown - (d * planned + c * breakdown + b * false_alarm)
    if best_saving is None or s > best_saving:
        best_t, best_saving = t, s

# Scale to a bigger fleet
fleet_size = st.sidebar.number_input("Fleet size (machines)", value=10000, step=1000)
fleet_saving = int(saving / len(X_test) * fleet_size)


def status(p):
    if p >= threshold:
        return "Red"
    if p >= threshold / 2:
        return "Amber"
    return "Green"


def action(p):
    if p >= threshold:
        return "Inspect within 24 hours"
    if p >= threshold / 2:
        return "Monitor closely, recheck next shift"
    return "No action needed"


risk = X_test.copy()
risk["failure_probability"] = probs.round(2)
risk["status"] = risk["failure_probability"].apply(status)

tab1, tab2, tab3, tab4 = st.tabs(
    ["Overview", "Machine detail", "Model insights", "Live simulation"]
)

# ---------- Tab 1: Overview ----------
with tab1:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Money saved", f"Rs {saving:,}")
    c2.metric("Failures caught", f"{tp} of {tp + fn}")
    c3.metric("False alarms", fp)
    c4.metric("Cost reduction", f"{round(saving / cost_without * 100) if cost_without else 0}%")
    st.success(f"Best alert threshold for this model: {best_t:.2f}, saving Rs {best_saving:,}")
    st.info(f"Estimated savings across a fleet of {fleet_size:,} machines: Rs {fleet_saving:,}")

    st.subheader("Highest-risk machines")
    top = risk.sort_values("failure_probability", ascending=False).head(20)
    st.dataframe(top)

    at_risk = risk[risk["status"] == "Red"].sort_values("failure_probability", ascending=False)
    st.download_button(
        "Download at-risk machines (CSV)",
        at_risk.to_csv().encode("utf-8"),
        "at_risk_machines.csv",
        "text/csv",
    )

    st.subheader("Torque vs tool wear")
    plot_df = X_test.copy()
    plot_df["actual_failure"] = y_test.astype(str)
    st.plotly_chart(
        px.scatter(plot_df, x="tool_wear", y="torque", color="actual_failure"),
        use_container_width=True,
    )

# ---------- Tab 2: Machine detail ----------
with tab2:
    st.subheader("Check one machine")
    ids = risk.sort_values("failure_probability", ascending=False).index[:200]
    machine = st.selectbox("Pick a machine ID (highest risk first)", ids)
    p = float(risk.loc[machine, "failure_probability"])
    d1, d2, d3 = st.columns(3)
    d1.metric("Failure probability", f"{p:.0%}")
    d2.metric("Status", status(p))
    d3.metric("Recommended action", action(p))
    st.progress(min(p, 1.0))
    st.write("Sensor readings:")
    st.dataframe(X_test.loc[[machine]])

# ---------- Tab 3: Model insights ----------
with tab3:
    st.subheader("Why machines get flagged (feature importance)")
    if hasattr(model, "feature_importances_"):
        imp = model.feature_importances_
    else:
        imp = abs(model[-1].coef_[0])
    imp_df = pd.DataFrame({"feature": features, "importance": imp}).sort_values("importance")
    st.plotly_chart(
        px.bar(imp_df, x="importance", y="feature", orientation="h"),
        use_container_width=True,
    )

    st.subheader("Confusion matrix")
    cm = confusion_matrix(y_test, pred)
    st.plotly_chart(
        px.imshow(
            cm, text_auto=True,
            x=["Predicted OK", "Predicted failure"],
            y=["Actually OK", "Actually failed"],
        ),
        use_container_width=True,
    )

    st.subheader("Model comparison (at current threshold)")
    rows = []
    for name, m in models.items():
        pr = (m.predict_proba(X_test)[:, 1] >= threshold).astype(int)
        rows.append({
            "Model": name,
            "Recall": round(recall_score(y_test, pr), 2),
            "Precision": round(precision_score(y_test, pr, zero_division=0), 2),
            "F1": round(f1_score(y_test, pr), 2),
        })
    st.dataframe(pd.DataFrame(rows))

# ---------- Tab 4: Live simulation ----------
with tab4:
    st.subheader("Live sensor simulation")
    st.write("Streams 25 machine readings one by one and raises an alert when risk crosses the threshold.")
    if st.button("Start live simulation"):
        box = st.empty()
        log = []
        sample = X_test.sample(25)
        for idx in sample.index:
            p = float(model.predict_proba(X_test.loc[[idx]])[0, 1])
            log.append({
                "machine": idx,
                "failure_probability": round(p, 2),
                "alert": "ALERT" if p >= threshold else "OK",
            })
            with box.container():
                if p >= threshold:
                    st.error(f"ALERT: Machine {idx} has {p:.0%} failure risk. Schedule maintenance.")
                else:
                    st.success(f"Machine {idx} is healthy ({p:.0%} risk).")
                st.dataframe(pd.DataFrame(log).tail(8))
            time.sleep(0.5)