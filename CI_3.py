import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

def main():
    st.title('Customer Purchasing Behavior Analysis')

    # File uploader
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        # Read the data
        data = pd.read_csv(uploaded_file)
        
        # Display basic information
        st.header('Dataset Overview')
        st.write(data.head())
        st.write(data.info())

        # Preprocessing
        data['high_spender'] = (data['purchase_amount'] > 300).astype(int)
        data = data.drop(columns=['user_id', 'purchase_amount'])

        # Encode categorical features
        label_encoder = LabelEncoder()
        data['region'] = label_encoder.fit_transform(data['region'])

        # Separate features and target
        X = data.drop(columns=['high_spender'])
        y = data['high_spender']

        # Standardize numerical features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

        # Train Random Forest Classifier
        model = RandomForestClassifier(random_state=42)
        model.fit(X_train, y_train)

        # Make predictions
        y_pred = model.predict(X_test)

        # Calculate metrics
        st.header('Model Performance Metrics')
        col1, col2, col3, col4 = st.columns(4)
        
        col1.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.2f}")
        col2.metric("Precision", f"{precision_score(y_test, y_pred):.2f}")
        col3.metric("Recall", f"{recall_score(y_test, y_pred):.2f}")
        col4.metric("F1 Score", f"{f1_score(y_test, y_pred):.2f}")

        # Confusion Matrix Visualization
        st.header('Confusion Matrix')
        fig, ax = plt.subplots()
        ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=ax, cmap='Blues')
        st.pyplot(fig)

        # Classification Report
        st.header('Classification Report')
        st.text(classification_report(y_test, y_pred))

if __name__ == '__main__':
    main()
