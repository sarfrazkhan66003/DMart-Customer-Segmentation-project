import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import joblib

# Load dataset
df = pd.read_csv(r"D:\PW Data Science\Project\dmart_transactions.csv")
df["tx_date"] = pd.to_datetime(df["tx_date"])
latest_date = df["tx_date"].max()

# --- Step 1: RFM Features ---
rfm = df.groupby("customer_id").agg({
    "tx_date": lambda x: (latest_date - x.max()).days,
    "transaction_id": "count",
    "amount": "sum"
}).reset_index()
rfm.columns = ["customer_id", "Recency", "Frequency", "Monetary"]

# --- Step 2: Category Share Features (IMPORTANT!) ---
category_sales = df.groupby(["customer_id", "category"])["amount"].sum().unstack(fill_value=0)
category_share = category_sales.div(category_sales.sum(axis=1), axis=0).reset_index()

# Merge with RFM
rfm = rfm.merge(category_share, on="customer_id", how="left")

# --- Step 3: Define consistent features ---
features = ["Recency", "Frequency", "Monetary"] + [col for col in category_share.columns if col != "customer_id"]

# Fill missing with 0
rfm[features] = rfm[features].fillna(0)

# --- Step 4: Scale ---
scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm[features])

# --- Step 5: Train KMeans ---
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
rfm["Cluster"] = kmeans.fit_predict(rfm_scaled)

# --- Step 6: Save everything ---
joblib.dump(kmeans, "dmart_kmeans_model.pkl")
joblib.dump(scaler, "dmart_scaler.pkl")
joblib.dump(features, "dmart_features.pkl")

print("✅ Model trained and saved with correct feature set!")
print(rfm.head())
# # --- Step 7: Save the final dataframe with clusters ---
# rfm.to_csv("dmart_rfm_with_clusters.csv", index=False)
