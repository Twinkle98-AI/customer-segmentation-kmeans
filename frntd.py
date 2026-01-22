import streamlit as st
import pandas as pd
import pickle
from sklearn.cluster import AgglomerativeClustering

st.set_page_config(page_title="Customer Segmentation", layout="centered")

# ---------------- LOAD DATA ----------------
data = pd.read_csv(r"C:\Users\Twinkele\Downloads\Mall_Customers.csv")
X = data.iloc[:, [3, 4]].values

# ---------------- LOAD MODELS ----------------
kmeans = pickle.load(open("kmeans_model.pkl", "rb"))

hc = AgglomerativeClustering(n_clusters=5, metric="euclidean", linkage="ward")
hc_labels = hc.fit_predict(X)

# ---------------- CLUSTER MEANINGS ----------------
cluster_info = {
    0: ("Low Income, Low Spending", "Budget"),
    1: ("Low Income, High Spending", "Careless"),
    2: ("Average Income, Average Spending", "Regular"),
    3: ("High Income, High Spending", "Premium"),
    4: ("High Income, Low Spending", "Savers")
}

st.title("🛍️ Customer Segmentation")
st.caption("K-Means & Hierarchical Prediction")

st.subheader("Enter Details")

income = st.slider("Income (k$)", 0, 200, 50)
score = st.slider("Spending Score", 1, 100, 50)

if st.button("Predict"):
    k_pred = kmeans.predict([[income, score]])[0]

    # Hierarchical nearest-centroid prediction
    distances = []
    for i in range(5):
        center = X[hc_labels == i].mean(axis=0)
        d = ((center[0]-income)**2 + (center[1]-score)**2)**0.5
        distances.append(d)
    h_pred = distances.index(min(distances))

    col1, col2 = st.columns(2)

    with col1:
        st.success(f"K-Means: {cluster_info[k_pred][1]}")
        st.caption(cluster_info[k_pred][0])

    with col2:
        st.success(f"Hierarchical: {cluster_info[h_pred][1]}")
        st.caption(cluster_info[h_pred][0])

st.markdown("---")
st.caption("📌 Segmentation Guide")
for k, v in cluster_info.items():
    st.write(f"Cluster {k+1}: {v[1]}")









