# =========================
# K-MEANS CLUSTERING
# =========================

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# =========================
# 1. LOAD DATA
# =========================
df = pd.read_csv(
r"C:\Users\SHYMA SUNDARAM\Desktop\heatwave prediction system for chennai\data\clean\heatwave_risk_dataset.csv"
)

print("Dataset shape:", df.shape)

# =========================
# 2. DATETIME PROCESSING
# =========================
df["datetime"] = pd.to_datetime(df["datetime"])
df["month"] = df["datetime"].dt.month
df = df.drop(columns=["datetime"])

# =========================
# 3. SELECT FEATURES
# =========================
features = [
 'feelslikemax', 'temp', 'feelslikemin', 'dew',
 'winddir', 'sealevelpressure', 'month', 'humidity',
 'visibility', 'solarradiation', 'solarenergy',
 'windspeedmean', 'cloudcover', 'uvindex',
 'windspeed'
]

X = df[features]

# 🔥 FIX: REMOVE NaN PROPERLY
X = X.dropna()
df = df.loc[X.index]

print("After removing NaN:", X.shape)

# =========================
# 4. FEATURE SCALING
# =========================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("Feature scaling completed")

# =========================
# 5. APPLY K-MEANS
# =========================
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)

df["Cluster"] = kmeans.fit_predict(X_scaled)

print("Clustering completed")

# =========================
# 6. CHECK DISTRIBUTION
# =========================
print("\nCluster distribution:\n")
print(df["Cluster"].value_counts())

# =========================
# 7. SAVE DATASET
# =========================
df.to_csv(
r"C:\Users\SHYMA SUNDARAM\Desktop\heatwave prediction system for chennai\data\clean\heatwave_clustered_dataset.csv",
index=False
)

print("\nClustered dataset saved successfully!")

# =========================
# 📊 CLUSTER VISUALIZATION (PCA)
# =========================

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Reduce dimensions to 2D
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(8,6))

sns.scatterplot(
    x=X_pca[:, 0],
    y=X_pca[:, 1],
    hue=df["Cluster"],
    palette="viridis",
    s=50
)

plt.title("K-Means Clustering of Heatwave Data (PCA Projection)", fontsize=11)
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")

plt.legend(title="Cluster")

plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
r"C:\Users\SHYMA SUNDARAM\Desktop\heatwave prediction system for chennai\outputs\cluster_pca_plot.png",
dpi=1000
)

plt.show()

# =========================
# 📊 CLUSTER ANALYSIS (MEAN VALUES)
# =========================

# Take important features to analyze
analysis_features = ["temp", "feelslikemax", "humidity"]

cluster_summary = df.groupby("Cluster")[analysis_features].mean()

cluster_summary.plot(
    kind="bar",
    figsize=(8,5),
    colormap="coolwarm"
)

plt.title("Cluster-wise Feature Analysis", fontsize=11)
plt.xlabel("Cluster")
plt.ylabel("Average Value")

plt.xticks(rotation=0)

plt.grid(axis='y', linestyle='--', alpha=0.4)

plt.tight_layout()
plt.savefig(
r"C:\Users\SHYMA SUNDARAM\Desktop\heatwave prediction system for chennai\outputs\cluster_analysis.png",
dpi=1200
)


plt.show()