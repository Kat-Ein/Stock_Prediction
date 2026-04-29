import os, sys, warnings
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
from sagemaker.predictor import Predictor
from sagemaker.serializers import JSONSerializer
from sagemaker.deserializers import NumpyDeserializer
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
import shap
from joblib import load

# Setup & Path Configuration
warnings.simplefilter("ignore")

# --- CUSTOM CLASSES (Fixes ModuleNotFoundError) ---
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
        # Create the feature the model expects
        if 'transactionamt' in X.columns and self.amt_median_ is not None:
            X['hightransactionamt'] = (X['transactionamt'] > self.amt_median_).astype(int)
        return X

# Path Configuration
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

file_path = os.path.join(project_root, 'Portfolio/X_train.csv')
dataset = pd.read_csv(file_path, index_col=0)

# Access secrets
aws_id = st.secrets["aws_credentials"]["AWS_ACCESS_KEY_ID"]
aws_secret = st.secrets["aws_credentials"]["AWS_SECRET_ACCESS_KEY"]
aws_token = st.secrets["aws_credentials"]["AWS_SESSION_TOKEN"]
aws_bucket = st.secrets["aws_credentials"]["AWS_BUCKET"]
aws_endpoint = st.secrets["aws_credentials"]["AWS_ENDPOINT"]

@st.cache_resource 
def get_session(aws_id, aws_secret, aws_token):
    return boto3.Session(aws_access_key_id=aws_id, aws_secret_access_key=aws_secret,
                         aws_session_token=aws_token, region_name='us-east-1')

session = get_session(aws_id, aws_secret, aws_token)
sm_session = sagemaker.Session(boto_session=session)

MODEL_INFO = {
    "endpoint"  : aws_endpoint,
    "explainer" : "explainer_pair.shap",
    "pipeline"  : "finalized_fraud_model.tar.gz",
    "keys"      : ['transactionamt','transactionhour','hightransactionamt','v92'],
    "inputs"    : [{"name": k, "type": "number", "min": -1.0, "max": 1.0, "default": 0.0, "step": 0.01} for k in ['transactionamt','transactionhour','hightransactionamt','v92']]
}

def load_pipeline(_session, bucket, key):
    s3_client = _session.client('s3')
    filename = MODEL_INFO["pipeline"]
    s3_client.download_file(Filename=filename, Bucket=bucket, Key=f"{key}/{filename}")
    with tarfile.open(filename, "r:gz") as tar:
        tar.extractall(path=".")
        joblib_file = [f for f in tar.getnames() if f.endswith('.joblib')][0]
    return joblib.load(joblib_file)

def load_shap_explainer(_session, bucket, key, local_path):
    s3_client = _session.client('s3')
    if not os.path.exists(local_path):
        s3_client.download_file(Filename=local_path, Bucket=bucket, Key=key)
    return joblib.load(local_path) # Changed to joblib for .shap files
'''
def call_model_api(input_dict):
    predictor = Predictor(
        endpoint_name=MODEL_INFO["endpoint"],
        sagemaker_session=sm_session,
        serializer=JSONSerializer(),
        deserializer=NumpyDeserializer()
    )
    try:
        # 1. Convert dictionary to DataFrame
        input_df = pd.DataFrame([input_dict])
        
        # 2. MANUALLY ensure 'productcd_count' exists
        # This is the specific feature the model is complaining about
        if 'productcd_count' not in input_df.columns:
            input_df['productcd_count'] = 0 

        # 3. Force the DataFrame to match the exact columns of your training set
        # 'dataset' is the X_train.csv you loaded at the top
        input_df = input_df.reindex(columns=dataset.columns, fill_value=0)
        
        # 4. Send to SageMaker
        # orient='records' sends the data with feature names included
        raw_pred = predictor.predict(input_df.to_dict(orient='records'))
        
        # 5. Handle result
        pred_val = np.array(raw_pred).flatten()[0]
        mapping = {0: "Legitimate", 1: "Fraud"}
        return mapping.get(int(float(pred_val)), "Unknown"), 200
        
    except Exception as e:
        return f"Error: {str(e)}", 500
'''
# Prediction Logic
def call_model_api(input_dict):
    predictor = Predictor(
        endpoint_name=MODEL_INFO["endpoint"],
        sagemaker_session=sm_session,
        serializer=JSONSerializer(),
        deserializer=NumpyDeserializer()
    )
    try:
        # 1. Convert input to DataFrame
        input_df = pd.DataFrame([input_dict])
        
        # 2. Reindex against your 328-column dataset
        # This aligns features and ensures the count is exactly what dataset.columns has
        input_df = input_df.reindex(columns=dataset.columns, fill_value=0.0)
        
        # 3. Double check the shape before sending
        # If dataset.columns is 328, input_df.shape[1] is now 328
        
        # 4. Send to SageMaker as a list of records
        raw_pred = predictor.predict(input_df.to_dict(orient='records'))
        
        # 5. Extract result
        pred_val = np.array(raw_pred).flatten()[0]
        mapping = {0: "Legitimate", 1: "Fraud"}
        
        return mapping.get(int(float(pred_val)), "Unknown"), 200
        
    except Exception as e:
        return f"Error: {str(e)}", 500

# Local Explainability
def display_explanation(input_dict, session, aws_bucket):
    try:
        explainer_name = MODEL_INFO["explainer"]
        local_shap_path = os.path.join(tempfile.gettempdir(), explainer_name)
        explainer = load_shap_explainer(session, aws_bucket, posixpath.join('explainer', explainer_name), local_shap_path)
        
        best_pipeline = load_pipeline(session, aws_bucket, 'sklearn-pipeline-deployment')
        
        # Transform data for SHAP
        input_df = pd.DataFrame([input_dict])
        input_df = input_df.reindex(columns=dataset.columns, fill_value=0.0)
        
        preprocessing_pipeline = Pipeline([step for step in best_pipeline.steps if step[0] != 'model'])
        input_df_transformed = preprocessing_pipeline.transform(input_df)
        
        # Generate SHAP waterfall
        shap_values = explainer(input_df_transformed)
        st.subheader("🔍 Decision Transparency (SHAP)")
        fig, ax = plt.subplots(figsize=(10, 4))
        shap.plots.waterfall(shap_values[0]) 
        st.pyplot(fig)
    except Exception as e:
        st.warning(f"SHAP Error: {e}")

# Streamlit UI
st.set_page_config(page_title="ML Deployment", layout="wide")
st.title("👨‍💻 ML Deployment")

with st.form("pred_form"):
    st.subheader(f"Inputs")
    cols = st.columns(2)
    user_inputs = {}
    for i, inp in enumerate(MODEL_INFO["inputs"]):
        with cols[i % 2]:
            user_inputs[inp['name']] = st.number_input(
                inp['name'].replace('_', ' ').upper(),
                value=float(inp['default']), step=inp['step']
            )
    submitted = st.form_submit_button("Run Prediction")

if submitted:
    original = dataset.iloc[0:1].to_dict(orient='records')[0]
    original.update(user_inputs)
    res, status = call_model_api(original)
    if status == 200:
        st.metric("Prediction Result", res)
        display_explanation(original, session, aws_bucket)
    else:
        st.error(res)
