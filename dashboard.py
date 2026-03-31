import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from lime.lime_tabular import LimeTabularExplainer

# =========================
# PAGE TITLE
# =========================
st.title("Heatwave Prediction System")
st.write("Predict heatwave risk using meteorological data")

# =========================
# LOAD DATA
# =========================
df = pd.read_csv(
r"C:\Users\SHYMA SUNDARAM\Desktop\heatwave prediction system for chennai\data\clean\heatwave_clustered_dataset.csv"
)

# remove leakage
if "Risk_Category" in df.columns:
    df = df.drop(columns=["Risk_Category"])

# =========================
# FEATURES (FINAL)
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
# TRAIN MODEL
# =========================
X_train, X_test, y_train, y_test = train_test_split(
X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

model = RandomForestClassifier(random_state=42)
model.fit(X_train_scaled, y_train)

# =========================
# TABS
# =========================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
"Prediction",
"Model Comparison",
"Feature Importance",
"Heatwave Analysis",
"Explainable AI"
])

# ------------------------
# TAB 1 : PREDICTION
# ------------------------
with tab1:

    st.sidebar.header("Enter Weather Conditions")

    inputs = []
    for f in features:
        val = st.sidebar.number_input(f, value=float(X[f].mean()))
        inputs.append(val)

    input_df = pd.DataFrame([inputs], columns=features)
    input_scaled = scaler.transform(input_df)

    if st.button("Predict Heatwave Risk"):
        pred = model.predict(input_scaled)[0]

        if pred == 0:
            st.success("Low Heatwave Risk")
        elif pred == 1:
            st.warning("Medium Heatwave Risk")
        else:
            st.error("High Heatwave Risk")

# ------------------------
# TAB 2 : MODEL COMPARISON
# ------------------------
with tab2:

    st.header("Model Comparison")

    model_names = ["LogReg", "SVM", "DT", "RF", "NN"]
    accuracies = [0.98, 0.99, 1.00, 0.99, 0.98]  # replace with your actual results

    fig, ax = plt.subplots()
    sns.barplot(x=model_names, y=accuracies, ax=ax)
    ax.set_title("Model Accuracy Comparison")
    ax.set_ylim(0.9,1.01)

    st.pyplot(fig)

# ------------------------
# TAB 3 : FEATURE IMPORTANCE
# ------------------------
with tab3:

    st.header("Feature Importance")

    importances = model.feature_importances_

    importance_df = pd.DataFrame({
        "Feature": features,
        "Importance": importances
    }).sort_values(by="Importance", ascending=False)

    fig, ax = plt.subplots()
    sns.barplot(x="Importance", y="Feature", data=importance_df, ax=ax)
    st.pyplot(fig)

# ------------------------
# TAB 4 : HEATWAVE ANALYSIS
# ------------------------
with tab4:

    st.header("Heatwave Risk Distribution")

    labels = df["Risk_Level"].map({0:"Low",1:"Medium",2:"High"})

    fig, ax = plt.subplots()
    sns.countplot(x=labels, ax=ax)
    st.pyplot(fig)

    # cluster analysis
    st.header("Cluster Distribution")

    fig2, ax2 = plt.subplots()
    sns.countplot(x=df["Cluster"], ax=ax2)
    st.pyplot(fig2)

# ------------------------
# TAB 5 : LIME
# ------------------------
with tab5:

    st.header("Explainable AI (LIME)")

    explainer = LimeTabularExplainer(
        training_data=X_train.values,
        feature_names=features,
        class_names=["Low","Medium","High"],
        mode="classification",
        discretize_continuous=False
    )

    def predict_fn(x):
        x_scaled = scaler.transform(pd.DataFrame(x, columns=features))
        return model.predict_proba(x_scaled)

    sample = X_test.iloc[0].values

    exp = explainer.explain_instance(
        sample,
        predict_fn,
        num_features=5
    )

    fig = exp.as_pyplot_figure()
    st.pyplot(fig)