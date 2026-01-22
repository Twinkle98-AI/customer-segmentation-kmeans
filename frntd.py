import streamlit as st
import numpy as np
import pickle
import pandas as pd
import matplotlib.pyplot as plt

# Load trained model
model = pickle.load(open("kmeans_model.pkl", "rb"))

st.set_page_config(page_title="Customer Segmentation", layout="centered")

st.title("🛍️ Customer Segmentation using K-Means")
st.write("Predict customer group based on Annual Income & Spending Score")

# ---------------- CLUSTER DEFINITIONS ----------------
cluster_info = {
    0: ("Low Value Customers", "Low income & low spending"),
    1: ("Careful Customers", "High income but low spending"),
    2: ("Potential Loyalists", "Low income but high spending"),
    3: ("Regular Customers", "Average income & average spending"),
    4: ("Premium Customers", "High income & high spending")
}

# ---------------- USER INPUT ----------------
st.subheader("Enter Customer Details")
income = st.slider("Annual Income (k$)", 0, 200, 50)
spending = st.slider("Spending Score (1-100)", 1, 100, 50)

# ---------------- PREDICTION ----------------
if st.button("Predict Customer Category"):
    customer = np.array([[income, spending]])
    cluster = model.predict(customer)[0]

    name, meaning = cluster_info.get(cluster, ("Unknown", "Unknown"))

    st.success(f"Cluster {cluster}: {name}")
    st.info(f"Customer Category: {meaning}")

# ---------------- CLUSTER GUIDE ----------------
st.markdown("---")
st.subheader("📌 Customer Segmentation Guide")

for k, v in cluster_info.items():
    st.write(f"**Cluster {k}: {v[0]}** → {v[1]}")

st.markdown("----")
st.caption("ML Model: K-Means | Dataset: Mall Customers")




