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

# --- STAGE 1: VIRTUAL MODULE HACK (Fixes ModuleNotFoundError) ---
# This creates a 'fake' path in memory so the model can find its classes
m = types.ModuleType('src.Custom_Classes')
sys.modules['src.Custom_Classes'] = m

# --- STAGE 2: CLASS DEFINITIONS ---
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

# --- STAGE 3: SETUP & PATHS ---
warnings.simplefilter("ignore")
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

file_path = os.path.join(project_root, 'Portfolio/X_train.csv')
dataset = pd.read_csv(file_path, index_col=0)

# Secrets
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
    "keys"      : ['transactionamt','transactionhour','hightransactionamt','v92'],
    "inputs"    : [{"name": k, "type": "number", "min": -1.0, "max": 1.0, "default": 0.0, "step": 0.01} for k in ['transactionamt','transactionhour','hightransactionamt','v92']]
}

# --- STAGE 4: LOADERS ---
def load_pipeline(_session, bucket, key):
    s3_client = _session.client('s3')
    filename = MODEL_INFO["pipeline"]
    local_path = os.path.join(tempfile.gettempdir(), filename)
    s3_client.download_file(Bucket=bucket, Key=f"{key}/{filename}", Filename=local_path)
    with tarfile.open(local_path, "r:gz") as tar:
        tar.extractall(path=tempfile.gettempdir())
        joblib_file = [f for f in tar.getnames() if f.endswith('.joblib')][0]
        return joblib.load(os.path.join(tempfile.gettempdir(), joblib_file))

def load_shap_explainer(_session, bucket, key, local_path):
    s3_client = _session.client('s3')
    if not os.path.exists(local_path):
        s3_client.download_file(Filename=local_path, Bucket=bucket, Key=key)
    return joblib.load(local_path)

# --- STAGE 5: PREDICTION (The "Template" Fix) ---
def call_model_api(user_inputs_dict):
    predictor = Predictor(endpoint_name=MODEL_INFO["endpoint"], sagemaker_session=sm_session,
                          serializer=JSONSerializer(), deserializer=NumpyDeserializer())
    try:
        # 1. Start with a real row from X_train (Guarantees correct feature count)
        df_template = dataset.loc[:, ~dataset.columns.str.contains('^Unnamed')].iloc[0:1].copy()
        
        # 2. Match user inputs to columns (case-insensitive)
        for key, value in user_inputs_dict.items():
            col_match = next((col for col in df_template.columns if col.lower() == key.lower()), None)
            if col_match:
                df_template[col_match] = value

        # 3. Ensure the model gets exactly 328 features (as requested by your specific error)
        if df_template.shape[1] > 328:
            df_template = df_template.iloc[:, :328]

        # 4. Predict
        raw_pred = predictor.predict(df_template.to_dict(orient='records'))
        pred_val = np.array(raw_pred).flatten()[0]
        return {0: "Legitimate", 1: "Fraud"}.get(int(float(pred_val)), "Unknown"), 200
    except Exception as e:
        return f"Error: {str(e)}", 500

# --- STAGE 6: SHAP EXPLANATION ---
def display_explanation(input_dict, session, aws_bucket):
    try:
        explainer_name = MODEL_INFO["explainer"]
        local_shap_path = os.path.join(tempfile.gettempdir(), explainer_name)
        explainer = load_shap_explainer(session, aws_bucket, posixpath.join('explainer', explainer_name), local_shap_path)
        
        best_pipeline = load_pipeline(session, aws_bucket, 'sklearn-pipeline-deployment')
        preprocessor = Pipeline([step for step in best_pipeline.steps if step[0] not in ['model', 'sampler']])
        
        input_df = pd.DataFrame([input_dict])
        input_df = input_df.reindex(columns=dataset.columns, fill_value=0.0)
        input_df_transformed = preprocessor.transform(input_df)
        
        shap_values = explainer(input_df_transformed)
        st.subheader("🔍 Decision Transparency (SHAP)")
        fig, ax = plt.subplots(figsize=(10, 4))
        shap.plots.waterfall(shap_values[0]) 
        st.pyplot(fig)
    except Exception as e:
        st.warning(f"SHAP explanation failed: {e}")

# --- STAGE 7: UI ---
st.set_page_config(page_title="ML Deployment", layout="wide")
st.title("👨‍💻 Fraud Detection System")

with st.form("pred_form"):
    st.subheader("Transaction Parameters")
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
    res, status = call_model_api(user_inputs)
    if status == 200:
        st.metric("Prediction Result", res)
        display_explanation(user_inputs, session, aws_bucket)
    else:
        st.error(res)
