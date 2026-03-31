import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

from lime.lime_tabular import LimeTabularExplainer


# =========================
# 1. LOAD DATA
# =========================
df = pd.read_csv(
r"C:\Users\SHYMA SUNDARAM\Desktop\heatwave prediction system for chennai\data\clean\heatwave_risk_dataset.csv"
)
df = df.dropna()

print("After removing NaN:", df.shape)

# =========================
# 2. DATETIME PROCESSING
# =========================
df["datetime"] = pd.to_datetime(df["datetime"])
df["month"] = df["datetime"].dt.month
df = df.drop(columns=["datetime"])

# =========================
# 3. FEATURES
# =========================
features = [
 'feelslikemax', 'temp', 'feelslikemin', 'dew',
 'winddir', 'sealevelpressure', 'month', 'humidity',
 'visibility', 'solarradiation', 'solarenergy',
 'windspeedmean', 'cloudcover', 'uvindex',
 'windspeed'
]

X = df[features]
y = df["Risk_Level"]

# =========================
# 4. TRAIN-TEST SPLIT
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =========================
# 5. FEATURE SCALING
# =========================
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# =========================
# 6. TRAIN MODEL
# =========================
model = RandomForestClassifier(random_state=42)
model.fit(X_train_scaled, y_train)

# =========================
# 7. LIME EXPLAINER (FIXED)
# =========================
explainer = LimeTabularExplainer(
    training_data=X_train.values,   # use ORIGINAL data
    feature_names=features,
    class_names=["Low", "Medium", "High"],
    mode="classification",
    discretize_continuous=False    # 🔥 KEY FIX (prevents truncnorm error)
)

# prediction wrapper (VERY IMPORTANT)
def predict_fn(x):
    x_scaled = scaler.transform(x)
    return model.predict_proba(x_scaled)

# =========================
# 8. EXPLAIN ONE SAMPLE
# =========================
sample = X_test.iloc[0].values

exp = explainer.explain_instance(
    sample,
    predict_fn,
    num_features=5
)

# =========================
# 9. OUTPUT
# =========================
print("\nLIME Explanation for Prediction:\n")

for feature, weight in exp.as_list():
    print(feature, ":", weight)

    # =========================
# 📊 LIME VISUALIZATION
# =========================
import matplotlib.pyplot as plt
import os

# Create output folder
output_path = r"C:\Users\SHYMA SUNDARAM\Desktop\heatwave prediction system for chennai\outputs"
os.makedirs(output_path, exist_ok=True)

# Generate LIME plot
fig = exp.as_pyplot_figure()

# 🔥 Change bar colors to blue
for ax in fig.axes:
    for bar in ax.patches:
        bar.set_color("#1f77b4")   # IEEE-style blue

# Title
plt.title("LIME Explanation for Heatwave Prediction", fontsize=11)

plt.tight_layout()

# Save figure
plt.savefig(
    os.path.join(output_path, "lime_explanation.png"),
    dpi=300,   # 🔥 IEEE standard
    bbox_inches='tight'
)

plt.show()