import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv(
r"C:\Users\SHYMA SUNDARAM\Desktop\heatwave prediction system for chennai\data\clean\heatwave_risk_dataset.csv"
)

# 🔍 Check columns
print(df.columns)

# 🧹 Remove missing values
df = df.dropna()

# Target
y = df["Risk_Level"]
# ⏱️ Datetime processing
df["datetime"] = pd.to_datetime(df["datetime"])

df["year"] = df["datetime"].dt.year
df["month"] = df["datetime"].dt.month
df["day"] = df["datetime"].dt.day

# Drop original datetime
df = df.drop(columns=["datetime"])

# 📊 Features
X = df[[
    "latitude",
    "longitude",

    "year",
    "month",
    "day",

    "temp",
    "feelslikemax",
    "feelslikemin",
    "dew",
    "humidity",
    "precip",
    "precipprob",
    "precipcover",
    "windspeed",
    "windspeedmean",
    "winddir",
    "sealevelpressure",
    "cloudcover",
    "visibility",
    "solarradiation",
    "solarenergy",
    "uvindex"
]]

# 🤖 Train model
model = RandomForestClassifier(random_state=42)
model.fit(X, y)

# 📌 Feature importance
importance = pd.Series(model.feature_importances_, index=X.columns)
importance = importance.sort_values(ascending=False)

print("\nFeature Importance:\n")
print(importance)

# ⭐ Top features
top_features = importance.head(16).index.tolist()

print("\nTop Features:\n", top_features)

# =========================
# 📊 FEATURE IMPORTANCE VISUALIZATION
# =========================
# Take top features (already computed)
top_n = 20
importance_top = importance.head(top_n)

# Plot
plt.figure(figsize=(10,6))

sns.barplot(
    x=importance_top.values,
    y=importance_top.index,
    palette="viridis"
)

plt.title("Top Feature Importance for Heatwave Prediction")
plt.xlabel("Importance Score")
plt.ylabel("Features")

plt.tight_layout()

# Save figure (VERY IMPORTANT for IEEE report)
plt.savefig(
r"C:\Users\SHYMA SUNDARAM\Desktop\heatwave prediction system for chennai\outputs\feature_importance.png",
dpi=1200
)

plt.show()