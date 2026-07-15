import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

# Page Configuration
st.set_page_config(page_title="EduPro Analytics Dashboard", layout="wide")

st.title("🎓 EduPro Demand & Revenue Forecaster")
st.markdown("Shift from Reactive Reporting to Proactive Planning with Machine Learning.")

# Load Raw Data (Taki dashboard mein actual values dikhe, scaled nahi)
@st.cache_data
def load_data():
    return pd.read_excel("Copy of EduPro Online Platform.xlsx", sheet_name='Courses')

courses_df = load_data()

# ---------------------------------------------------------
# OUT-OF-THE-BOX IDEA: Price Elasticity Simulator
# ---------------------------------------------------------
st.header("💡 Dynamic Price Simulator (Out-of-the-box Feature)")
st.markdown("Test how changing course prices might affect your overall platform strategy.")

col1, col2 = st.columns(2)

with col1:
    st.info("Adjust the base price to see the simulated impact on Demand.")
    price_change = st.slider("Increase/Decrease Average Course Price (%)", min_value=-50, max_value=50, value=0, step=5)

with col2:
    # Basic economic simulation: Demand usually drops when price increases
    simulated_demand_impact = - (price_change * 0.5) 
    
    st.metric(label="Simulated Impact on Total Enrollments", 
              value=f"{simulated_demand_impact}%", 
              delta=f"{simulated_demand_impact}%",
              delta_color="normal")

st.divider()

# ---------------------------------------------------------
# DASHBOARD TABS
# ---------------------------------------------------------
tab1, tab2 = st.tabs(["📊 Category Level Demand", "🧠 Feature Importance Insights"])

with tab1:
    st.subheader("Current Course Distribution by Category")
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.countplot(data=courses_df, y='CourseCategory', palette='mako', ax=ax)
    ax.set_xlabel("Number of Courses Available")
    ax.set_ylabel("Category")
    st.pyplot(fig)

with tab2:
    st.subheader("What Drives Course Enrollments?")
    st.image("feature_importance.png", caption="Machine Learning Insight: Duration and Price are the biggest drivers of demand.", use_container_width=True)

st.sidebar.success("Dashboard successfully loaded! Use the simulator to plan proactively.")