import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


# =========================
# 1. LOAD DATA
# =========================
df = pd.read_csv(
r"C:\Users\SHYMA SUNDARAM\Desktop\heatwave prediction system for chennai\data\clean\heatwave_clustered_dataset.csv"
)

print("Dataset shape:", df.shape)

# =========================
# 2. FEATURES (UPDATED)
# =========================
features = [
 'feelslikemax', 'temp', 'feelslikemin', 'dew',
 'winddir', 'sealevelpressure', 'month', 'humidity',
 'visibility', 'solarradiation', 'solarenergy',
 'windspeedmean', 'cloudcover', 'uvindex',
 'windspeed'
]

X = df[features]

# Target
y = df["Risk_Level"]

# =========================
# 3. TRAIN-TEST SPLIT
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Train-test split completed")

# =========================
# 4. FEATURE SCALING
# =========================
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Feature scaling completed")

# =========================
# 5. MODELS
# =========================
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "SVM": SVC(kernel="linear"),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(random_state=42)
}

# =========================
# 6. TRAIN & EVALUATE
# =========================
print("\nModel Performance:\n")

for name, model in models.items():

    # Use scaled data for LR & SVM, original for trees
    if name in ["Logistic Regression", "SVM"]:
        model.fit(X_train_scaled, y_train)
        predictions = model.predict(X_test_scaled)
    else:
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    print(f"{name} Accuracy: {accuracy:.4f}")