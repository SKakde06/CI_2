import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, classification_report, confusion_matrix
)

# Matplotlib import with error handling
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

def display_dataframe_info(df):
    """
    Custom function to display DataFrame info in Streamlit
    """
    # Capture info in a string buffer
    buffer = []
    
    # Columns info
    buffer.append("### Columns Information")
    buffer.append(f"Total Columns: {len(df.columns)}")
    
    # Data types
    buffer.append("\n### Column Data Types")
    for col, dtype in df.dtypes.items():
        buffer.append(f"- {col}: {dtype}")
    
    # Non-null counts
    buffer.append("\n### Non-Null Counts")
    for col in df.columns:
        non_null_count = df[col].count()
        total_count = len(df)
        buffer.append(f"- {col}: {non_null_count} non-null out of {total_count} total")
    
    # Memory usage
    buffer.append(f"\n### Memory Usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    return "\n".join(buffer)

def main():
    st.set_page_config(page_title="Customer Analysis", layout="wide")
    
    st.title('🛍️ Customer Purchasing Behavior Analysis')

    # Sidebar for file upload
    st.sidebar.header('Upload Your Dataset')
    uploaded_file = st.sidebar.file_uploader(
        "Choose a CSV file", 
        type="csv", 
        help="Upload a CSV file with customer purchasing data"
    )
    
    if uploaded_file is not None:
        try:
            # Read the data
            data = pd.read_csv(uploaded_file)
            
            # Dataset Information
            st.header('📊 Dataset Overview')
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader('First Few Rows')
                st.dataframe(data.head())
            
            with col2:
                st.subheader('Dataset Info')
                # Use custom info display function
                st.markdown(display_dataframe_info(data))

            # Preprocessing
            # Verify required columns exist
            required_columns = ['user_id', 'purchase_amount', 'region']
            missing_columns = [col for col in required_columns if col not in data.columns]
            
            if missing_columns:
                st.error(f"Missing required columns: {missing_columns}")
                st.stop()

            # Create high_spender feature
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
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=0.3, random_state=42
            )

            # Train Random Forest Classifier
            model = RandomForestClassifier(random_state=42)
            model.fit(X_train, y_train)

            # Make predictions
            y_pred = model.predict(X_test)

            # Model Performance Metrics
            st.header('📈 Model Performance Metrics')
            
            # Metrics display
            col1, col2, col3, col4 = st.columns(4)
            
            col1.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.2f}")
            col2.metric("Precision", f"{precision_score(y_test, y_pred):.2f}")
            col3.metric("Recall", f"{recall_score(y_test, y_pred):.2f}")
            col4.metric("F1 Score", f"{f1_score(y_test, y_pred):.2f}")

            # Visualization section
            st.header('🔍 Model Visualization')
            
            # Confusion Matrix
            fig, ax = plt.subplots(figsize=(8, 6))
            cm = confusion_matrix(y_test, y_pred)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
            ax.set_title('Confusion Matrix')
            ax.set_xlabel('Predicted Label')
            ax.set_ylabel('True Label')
            st.pyplot(fig)

            # Classification Report
            st.header('📋 Classification Report')
            st.text(classification_report(y_test, y_pred))

        except Exception as e:
            st.error(f"An error occurred: {e}")
            # Optionally, print full traceback for debugging
            import traceback
            st.error(traceback.format_exc())
    else:
        st.info("Please upload a CSV file to begin analysis")

if __name__ == '__main__':
    main()
