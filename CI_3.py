import sys
import subprocess

def install(package):
    """Install a package using pip"""
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# Try to import required packages, install if not available
try:
    import sklearn
except ImportError:
    st.warning("Installing scikit-learn. This may take a moment...")
    install('scikit-learn')
    import sklearn

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score, 
    classification_report, 
    confusion_matrix
)

def load_data():
    """Load sample customer data"""
    np.random.seed(42)
    data = pd.DataFrame({
        'region': np.random.choice(['North', 'South', 'East', 'West', 'Central'], 100),
        'age': np.random.randint(18, 65, 100),
        'income': np.random.randint(30000, 150000, 100),
        'spending_score': np.random.randint(1, 100, 100),
        'purchase_amount': np.random.randint(100, 500, 100)
    })
    return data

def preprocess_data(data):
    """Preprocess the data for model training"""
    # Create high spender target variable
    data['high_spender'] = (data['purchase_amount'] > 300).astype(int)

    # Encode categorical features
    label_encoder = LabelEncoder()
    data['region_encoded'] = label_encoder.fit_transform(data['region'])

    # Select features
    features = ['region_encoded', 'age', 'income', 'spending_score']
    X = data[features]
    y = data['high_spender']

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, features

def train_and_evaluate_model(X_scaled, y):
    """Train the model and generate evaluation metrics"""
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.3, random_state=42
    )

    # Train Random Forest Classifier
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate metrics
    metrics = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1 Score': f1_score(y_test, y_pred)
    }

    return model, metrics, X_test, y_test, y_pred

def main():
    st.title('Customer Purchasing Behavior Analysis')

    # Load data
    st.header('Data Overview')
    data = load_data()
    st.dataframe(data.head())

    # Preprocess data
    st.header('Data Preprocessing')
    X_scaled, y, features = preprocess_data(data)
    st.write(f"Features used: {features}")

    # Train and evaluate model
    st.header('Model Training and Evaluation')
    try:
        model, metrics, X_test, y_test, y_pred = train_and_evaluate_model(X_scaled, y)

        # Display metrics
        st.subheader('Performance Metrics')
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric('Accuracy', f"{metrics['Accuracy']:.2f}")
        with col2:
            st.metric('Precision', f"{metrics['Precision']:.2f}")
        with col3:
            st.metric('Recall', f"{metrics['Recall']:.2f}")
        with col4:
            st.metric('F1 Score', f"{metrics['F1 Score']:.2f}")

        # Confusion Matrix
        st.subheader('Confusion Matrix')
        conf_matrix = confusion_matrix(y_test, y_pred)
        st.table(conf_matrix)

        # Feature Importance
        st.subheader('Feature Importance')
        feature_importance = pd.DataFrame({
            'Feature': features,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False)
        st.dataframe(feature_importance)

        # Classification Report
        st.subheader('Classification Report')
        report = classification_report(y_test, y_pred, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df)

    except Exception as e:
        st.error(f"An error occurred: {e}")
        st.error("Make sure all required libraries are installed.")

if __name__ == '__main__':
    main()
