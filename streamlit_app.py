import streamlit as st
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

st.set_page_config(
    page_title="RetainAI – Telecom Customer Churn Prediction System",
    layout="centered"
)

model = joblib.load("model.pkl")
features = joblib.load("features.pkl")

st.title("RetainAI 📉")
st.caption("Predict whether a customer is likely to churn")
st.divider()

st.subheader("🧾 Enter Customer Details")

col1, col2 = st.columns(2)
with col1:
    age = st.number_input("Age", min_value=10, max_value=100, value=30)
    tenure = st.number_input("Tenure (months)", min_value=0, max_value=240, value=12)
    monthly_charges = st.number_input("Monthly Charges", min_value=0.0, value=70.0)

with col2:
    gender = st.selectbox("Gender", ["Male", "Female"])
    contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    tech_support = st.selectbox("Tech Support", ["Yes", "No"])

st.divider()

if st.button("Predict Churn", use_container_width=True):
    input_df = pd.DataFrame([{
        "Age": age,
        "Tenure": tenure,
        "MonthlyCharges": monthly_charges,
        "Gender": gender,
        "ContractType": contract,
        "InternetService": internet,
        "TechSupport": tech_support
    }])

    input_encoded = pd.get_dummies(input_df)
    input_encoded = input_encoded.reindex(columns=features, fill_value=0)

    prediction = model.predict(input_encoded)[0]
    proba = model.predict_proba(input_encoded)[0]
    churn_prob = proba[list(model.classes_).index("Yes")]

    st.subheader("📊 Prediction Result")

    if prediction == "Yes":
        st.error("⚠️ High Risk of Customer Churn")
    else:
        st.success("✅ Customer is Likely to Stay")

    # Gauge chart
    fig, ax = plt.subplots(figsize=(6, 3), subplot_kw=dict(aspect="equal"))
    fig.patch.set_facecolor("#0e1117")
    ax.set_facecolor("#0e1117")

    theta = np.linspace(np.pi, 0, 300)
    r_outer, r_inner = 1.0, 0.6

    for i in range(len(theta) - 1):
        t = i / (len(theta) - 2)
        color = (t, 1 - t, 0.2)
        ax.fill_between(
            [r_inner * np.cos(theta[i]), r_outer * np.cos(theta[i])],
            [r_inner * np.sin(theta[i]), r_outer * np.sin(theta[i])],
            [r_inner * np.cos(theta[i+1]), r_outer * np.cos(theta[i+1])],
            [r_inner * np.sin(theta[i+1]), r_outer * np.sin(theta[i+1])],
            color=color, linewidth=0
        )

    needle_angle = np.pi - churn_prob * np.pi
    ax.plot([0, 0.75 * np.cos(needle_angle)], [0, 0.75 * np.sin(needle_angle)],
            color="white", linewidth=3, solid_capstyle="round")
    ax.add_patch(plt.Circle((0, 0), 0.05, color="white", zorder=5))

    ax.text(0, -0.15, f"{churn_prob:.0%}", ha="center", va="center",
            fontsize=22, fontweight="bold", color="white")
    ax.text(0, -0.32, "Churn Probability", ha="center", va="center",
            fontsize=10, color="#aaaaaa")

    ax.text(-1.0, -0.1, "Low", ha="center", color="#aaaaaa", fontsize=9)
    ax.text(1.0, -0.1, "High", ha="center", color="#aaaaaa", fontsize=9)

    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-0.5, 1.1)
    ax.axis("off")
    st.pyplot(fig)
    plt.close()

    # Feature importance
    st.subheader("🔍 What's Driving This Prediction")

    importances = model.feature_importances_
    feat_series = pd.Series(importances, index=features).sort_values(ascending=True).tail(8)

    fig2, ax2 = plt.subplots(figsize=(6, 4))
    fig2.patch.set_facecolor("#0e1117")
    ax2.set_facecolor("#0e1117")

    colors = ["#e05252" if prediction == "Yes" else "#52b788"] * len(feat_series)
    bars = ax2.barh(feat_series.index, feat_series.values, color=colors, height=0.5)

    ax2.set_xlabel("Importance", color="#aaaaaa")
    ax2.tick_params(colors="#cccccc", labelsize=9)
    ax2.spines[:].set_visible(False)
    ax2.xaxis.label.set_color("#aaaaaa")
    for spine in ax2.spines.values():
        spine.set_visible(False)
    ax2.tick_params(axis="x", colors="#aaaaaa")
    ax2.tick_params(axis="y", colors="#cccccc")

    st.pyplot(fig2)
    plt.close()

    # Summary cards
    st.subheader("📋 Customer Summary")
    c1, c2, c3 = st.columns(3)
    c1.metric("Tenure", f"{tenure} mo")
    c2.metric("Monthly Charges", f"${monthly_charges:.0f}")
    c3.metric("Contract", contract.split()[0])

else:
    st.info("Enter customer details and click **Predict Churn**")