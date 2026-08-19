import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳"
)

st.title("💳 Credit Card Fraud Detection")
st.write(
    "This application uses Logistic Regression to detect "
    "potentially fraudulent credit card transactions."
)

uploaded_file = st.file_uploader(
    "Upload creditcard.csv (optional)",
    type="csv"
)

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("Dataset uploaded successfully.")
else:
    st.info("No dataset uploaded. The application is using demonstration data.")

    X_demo, y_demo = make_classification(
        n_samples=5000,
        n_features=30,
        n_informative=12,
        n_redundant=8,
        weights=[0.995, 0.005],
        random_state=42
    )

    feature_names = (
        ["Time"]
        + [f"V{i}" for i in range(1, 29)]
        + ["Amount"]
    )

    df = pd.DataFrame(X_demo, columns=feature_names)
    df["Class"] = y_demo

st.subheader("Dataset Preview")
st.dataframe(df.head())

st.subheader("Class Distribution")
st.bar_chart(df["Class"].value_counts())

X = df.drop("Class", axis=1)
y = df["Class"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

model = make_pipeline(
    StandardScaler(),
    LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42
    )
)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

st.subheader("Model Performance")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.3f}")
col2.metric("Precision", f"{precision_score(y_test, y_pred):.3f}")
col3.metric("Recall", f"{recall_score(y_test, y_pred):.3f}")
col4.metric("F1-score", f"{f1_score(y_test, y_pred):.3f}")

st.subheader("Confusion Matrix")

cm = confusion_matrix(y_test, y_pred)

fig, ax = plt.subplots()
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    ax=ax
)
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")

st.pyplot(fig)
