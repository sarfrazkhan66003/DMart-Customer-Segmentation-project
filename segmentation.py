# train_model.py
import pandas as pd
from sklearn.cluster import KMeans
import joblib

# Dummy dataset (replace with your real data)
data = {
    "CustomerID": [1, 2, 3, 4, 5],
    "Annual_Income": [15000, 35000, 50000, 75000, 120000],
    "Spending_Score": [20, 60, 80, 40, 90]
}
df = pd.DataFrame(data)

# Select features
X = df[['Annual_Income', 'Spending_Score']]

# Train model
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X)

# Save model
joblib.dump(kmeans, "dmart_kmeans_model.pkl")

print("✅ Model trained and saved as dmart_kmeans_model.pkl")
