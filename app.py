import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE

st.set_page_config(page_title="Fraud Detection", layout="wide")

st.title("Fraud Detection using Machine Learning")
st.markdown("Upload your `fraud.csv` file to train and evaluate ML models.")

uploaded_file = st.file_uploader("Upload fraud.csv", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success(f"Dataset loaded! Shape: {df.shape}")
    st.dataframe(df.head())

    target = st.selectbox("Select target column", df.columns)

    if st.button("Train Models"):
        with st.spinner("Training models, please wait..."):

            df = df.dropna()
            le = LabelEncoder()
            for col in df.select_dtypes(include='object').columns:
                df[col] = le.fit_transform(df[col])

            X = df.drop(columns=[target])
            y = df[target]

            smote = SMOTE(random_state=42)
            X_res, y_res = smote.fit_resample(X, y)

            X_train, X_test, y_train, y_test = train_test_split(
                X_res, y_res, test_size=0.2, random_state=42)

            models = {
                "Logistic Regression": LogisticRegression(max_iter=1000),
                "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
            }

            for name, model in models.items():
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                report = classification_report(y_pred, y_test, output_dict=True)

                st.subheader(f"{name}")
                st.dataframe(pd.DataFrame(report).transpose().round(2))

                fig, ax = plt.subplots()
                sns.heatmap(confusion_matrix(y_test, y_pred),
                            annot=True, fmt='d', ax=ax, cmap='Blues')
                ax.set_title(f"Confusion Matrix - {name}")
                st.pyplot(fig)

        st.success("Done!")
else:
    st.info("Please upload fraud.csv to get started.")