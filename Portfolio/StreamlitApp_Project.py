import os, sys, warnings, types
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import posixpath
import joblib
import tarfile
import tempfile
import boto3
import sagemaker
import __main__
from sagemaker.predictor import Predictor
from sagemaker.serializers import JSONSerializer
from sagemaker.deserializers import NumpyDeserializer
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
import shap

# --- 1. VIRTUAL MODULE HACK (Fixes ModuleNotFoundError) ---
m = types.ModuleType('src.Custom_Classes')
sys.modules['src.Custom_Classes'] = m

# --- 2. CUSTOM CLASS DEFINITIONS ---
class DataCleaner(BaseEstimator, TransformerMixin):
    def __init__(self, missing_threshold=0.80):
        self.missing_threshold = missing_threshold
    def fit(self, X, y=None):
        X = pd.DataFrame(X).copy()
        X.columns = X.columns.str.strip().str.lower().str.replace(' ', '_')
        missing_fractions = X.isnull().mean()
        self.drop_list_ = list(missing_fractions[missing_fractions > self.missing_threshold].index)
        return self
    def transform(self, X):
        X = pd.DataFrame(X).copy()
        X.columns = X.columns.str.strip().str.lower().str.replace(' ', '_')
        return X.drop(columns=self.drop_list_, errors='ignore')

class FraudFeatureExtractor(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.amt_median_ = None
    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        if 'transactionamt' in X.columns:
            self.amt_median_ = X['transactionamt'].median()
        return self
    def transform(self, X):
        X = pd.DataFrame(X).copy()
        if 'transactionamt' in X.columns and self.amt_median_ is not None:
            X['hightransactionamt'] = (X['transactionamt'] > self.amt_median_).astype(int)
        return X

# Link classes to the fake module and __main__
m.DataCleaner = DataCleaner
m.FraudFeatureExtractor = FraudFeatureExtractor
__main__.DataCleaner = DataCleaner
__main__.FraudFeatureExtractor = FraudFeatureExtractor

# --- 3. SETUP & DATA LOADING ---
warnings.simplefilter("ignore")
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

# Load Dataset
file_path = os.path.join(project_root, 'Portfolio/X_train.csv')
dataset = pd.read_csv(file_path, index_col=0)
# Clean dataset columns immediately to remove 'Unnamed' garbage
dataset = dataset.loc[:, ~dataset.columns.str.contains('^Unnamed')]

# AWS Secrets
aws_id = st.secrets["aws_credentials"]["AWS_ACCESS_KEY_ID"]
aws_secret = st.secrets["aws_credentials"]["AWS_SECRET_ACCESS_KEY"]
aws_token = st.secrets["aws_credentials"]["AWS_SESSION_TOKEN"]
aws_bucket = st.secrets["aws_credentials"]["AWS_BUCKET"]
aws_endpoint = st.secrets["aws_credentials"]["AWS_ENDPOINT"]

session = boto3.Session(aws_access_key_id=aws_id, aws_secret_access_key=aws_secret,
                        aws_session_token=aws_token, region_name='us-east-1')
sm_session = sagemaker.Session(boto_session=session)

MODEL_INFO = {
    "endpoint"  : aws_endpoint,
    "explainer" : "explainer_pair.shap",
    "pipeline"  : "finalized_fraud_model.tar.gz",
    "inputs"    : ['transactionamt','transactionhour','hightransactionamt','v92']
}

# --- 4. PREDICTION LOGIC ---
def call_model_api(user_inputs_dict):
    predictor = Predictor(endpoint_name=MODEL_INFO["endpoint"], sagemaker_session=sm_session,
                          serializer=JSONSerializer(), deserializer=NumpyDeserializer())
    try:
        # 1. Use a real row as template
        df_template = dataset.iloc[0:1].copy()
        
        # 2. Update user values
        for key, value in user_inputs_dict.items():
            col_match = next((col for col in df_template.columns if col.lower() == key.lower()), None)
            if col_match:
                df_template[col_match] = value

        # 3. CRITICAL: The error said it expects 328. We force exactly 328.
        if df_template.shape[1] > 328:
            df_template = df_template.iloc[:, :328]
        
        # 4. Convert to list of dicts (RECORDS)
        payload = df_template.to_dict(orient='records')
        
        # 5. Invoke
        raw_pred = predictor.predict(payload)
        pred_val = np.array(raw_pred).flatten()[0]
        
        return {0: "Legitimate", 1: "Fraud"}.get(int(float(pred_val)), "Unknown"), 200
    except Exception as e:
        return f"Error: {str(e)}", 500

# --- 5. UI ---
st.title("👨‍💻 Fraud Detection System")

with st.form("pred_form"):
    cols = st.columns(2)
    ui_data = {}
    for i, name in enumerate(MODEL_INFO["inputs"]):
        with cols[i % 2]:
            ui_data[name] = st.number_input(name.upper(), value=0.0)
    submitted = st.form_submit_button("Run Prediction")

if submitted:
    result, status = call_model_api(ui_data)
    if status == 200:
        st.metric("Result", result)
    else:
        st.error(result)
