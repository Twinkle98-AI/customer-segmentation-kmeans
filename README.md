# customer-segmentation-kmeans
🛍️ Customer Segmentation using K-Means & Streamlit
📌 Project Overview

This project uses K-Means Clustering to segment mall customers based on their Annual Income and Spending Score.
The trained model is deployed using an interactive Streamlit web application that predicts the customer segment in real time and visualizes cluster groups.

🔍 Problem Statement

Businesses often treat all customers the same. This leads to ineffective marketing strategies.
This project solves that by grouping customers into meaningful segments so companies can target customers more effectively.

⚙️ Solution Approach

Load and preprocess the dataset

Apply K-Means clustering

Use the Elbow Method to find the optimal number of clusters

Train the model with 5 clusters

Deploy the model using Streamlit

📊 Customer Segments
Cluster	Group Name	Description
0	Low Value Customers	Low income & low spending
1	Careful Customers	High income but low spending
2	Potential Loyalists	Low income but high spending
3	Regular Customers	Average income & spending
4	Premium Customers	High income & high spending
🛠 Tech Stack

Python

Pandas, NumPy

Scikit-learn

Matplotlib

Streamlit

📂 Project Structure
customer_segmentation/
│
├── Mall_Customers.csv
├── kmeans_model.pkl
├── app.py
└── README.md

▶ How to Run the Project
pip install pandas numpy scikit-learn matplotlib streamlit
streamlit run app.py

🎯 Business Impact

Identifies high-value customers

Improves marketing efficiency

Helps design personalized offers

Supports data-driven business decisions

👩‍💻 Author

Haimabati Haripriya
Sahu
Machine Learning Enthusiast
