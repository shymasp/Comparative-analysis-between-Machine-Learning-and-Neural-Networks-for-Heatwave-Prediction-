# Cleaned the merged dataset by removing the duplicates
# Created heatwave labels (numerical) (0- low, 1-medium, 2-high) from Maximum temperature values
# Converted the heatwave labels from numerical to categorical
# Added missing value verification

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
# Load merged dataset
df = pd.read_csv(r"C:/Users/SHYMA SUNDARAM/Desktop/heatwave prediction system for chennai/data/clean/chennai_combined_2010_2023.csv")
print("Original dataset shape:", df.shape)


# 🔍 Check missing values BEFORE cleaning
print("\nMissing values before cleaning:")
print(df.isnull().sum())


# Optional: Show % of missing values
print("\nMissing value percentage:")
print((df.isnull().sum() / len(df)) * 100)


# 🧹 Handle missing values (IMPORTANT for ML)
# Option 1: Drop rows where T2M_MAX is missing (critical column)
df = df.dropna(subset=["feelslikemax"])


# Remove duplicate rows
df = df.drop_duplicates()
print("\nAfter removing duplicates:", df.shape)


# Create Risk_Level column based on T2M_MAX thresholds
def classify_risk(temp):
    
    if temp < 34:
        return 0   # Low
    
    elif 34 <= temp <= 38:
        return 1   # Medium
    
    else:
        return 2   # High


df["Risk_Level"] = df["feelslikemax"].apply(classify_risk)


# Convert numeric risk to categorical labels for visualization
risk_mapping = {
    0: "Low",
    1: "Medium",
    2: "High"
}

df["Risk_Category"] = df["Risk_Level"].map(risk_mapping)


# 🔍 Check missing values AFTER processing
print("\nMissing values after cleaning:")
print(df.isnull().sum())


# Check class distribution
print("\nRisk Category Distribution:")
print(df["Risk_Category"].value_counts())


# Save processed dataset
df.to_csv(
    r"C:\Users\SHYMA SUNDARAM\Desktop\heatwave prediction system for chennai\data\clean\heatwave_risk_dataset.csv",
    index=False
)

print("\nPreprocessing completed and dataset saved.")

import numpy as np

# =========================
# CREATE TEMPERATURE BINS
# =========================

bins = np.arange(20, 50, 2)

labels = [f"{bins[i]}–{bins[i+1]}" for i in range(len(bins)-1)]

df["temp_bins"] = pd.cut(df["feelslikemax"], bins=bins, labels=labels)

# Frequency count
freq = df["temp_bins"].value_counts().sort_index()

## =========================
# 📊 IEEE-STYLE FREQUENCY DISTRIBUTION
# =========================

plt.figure(figsize=(10,5))   # 🔥 smaller + compact

bars = plt.bar(
    freq.index.astype(str),
    freq.values,
    color="#8DB9FF",
    edgecolor="black",
    width=0.7
)

# Add values (proper placement)
for bar in bars:
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width()/2,
        height + (max(freq.values) * 0.01),   # 🔥 dynamic spacing
        f"{int(height)}",
        ha='center',
        va='bottom',
        fontsize=9
    )

plt.title("Frequency Distribution of Temperature Intervals", fontsize=12)
plt.xlabel("Temperature Interval (°C)", fontsize=10)
plt.ylabel("Number of Days", fontsize=10)

plt.xticks(rotation=30)

# Clean grid (IEEE style)
plt.grid(axis='y', linestyle='--', alpha=0.4)

# Remove extra margins
plt.margins(x=0.01)

plt.tight_layout()

plt.savefig(
r"C:\Users\SHYMA SUNDARAM\Desktop\heatwave prediction system for chennai\outputs\risk_distribution.png",
dpi=1200
)
plt.show()

# =========================
# 📈 TIME SERIES ANALYSIS (FIXED)
# =========================

plt.figure(figsize=(10,4))   # 🔥 compact

# ensure datetime
df["datetime"] = pd.to_datetime(df["datetime"])

# sort
df = df.sort_values("datetime")

# 🔥 CREATE rolling feature
df["rolling_risk"] = df["Risk_Level"].rolling(window=30).mean()

# THEN plot
plt.plot(df["datetime"], df["rolling_risk"])

plt.plot(
    df["datetime"],
    df["rolling_risk"],
    color="#1f77b4",   # 🔥 professional blue
    linewidth=1.5
)

plt.title("Temporal Trend of Heatwave Risk", fontsize=11)
plt.xlabel("Year", fontsize=10)
plt.ylabel("Average Risk Level", fontsize=10)

# Clean grid
plt.grid(True, linestyle="--", alpha=0.4)

# Limit y-axis (cleaner look)
plt.ylim(0, 2.5)

# Reduce tick clutter
plt.xticks(fontsize=9)
plt.yticks(fontsize=9)

plt.tight_layout()

plt.savefig(
    r"C:\Users\SHYMA SUNDARAM\Desktop\heatwave prediction system for chennai\outputs\heatwave_time_series.png",
    dpi=300,
    bbox_inches='tight'
)

plt.show()