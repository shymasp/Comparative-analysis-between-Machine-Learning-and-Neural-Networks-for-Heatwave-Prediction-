import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

from sklearn.neural_network import MLPClassifier


# =========================
# 1. LOAD DATA
# =========================
df = pd.read_csv(
r"C:\Users\SHYMA SUNDARAM\Desktop\heatwave prediction system for chennai\data\clean\heatwave_clustered_dataset.csv"
)

print("Dataset shape:", df.shape)

# =========================
# 2. REMOVE LEAKAGE (IMPORTANT)
# =========================
if "Risk_Category" in df.columns:
    df = df.drop(columns=["Risk_Category"])

# =========================
# 3. FEATURES (UPDATED)
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

print("Train-test split completed")

# =========================
# 5. FEATURE SCALING (REQUIRED for NN)
# =========================
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Feature scaling completed")

# =========================
# 6. NEURAL NETWORK MODEL
# =========================
nn_model = MLPClassifier(
    hidden_layer_sizes=(64, 32),
    activation='relu',
    solver='adam',
    max_iter=500,
    random_state=42
)

# =========================
# 7. TRAIN MODEL
# =========================
nn_model.fit(X_train_scaled, y_train)

# =========================
# 8. PREDICTIONS
# =========================
predictions = nn_model.predict(X_test_scaled)

# =========================
# 9. EVALUATION
# =========================
accuracy = accuracy_score(y_test, predictions)

print("\nNeural Network Accuracy:", round(accuracy, 4))