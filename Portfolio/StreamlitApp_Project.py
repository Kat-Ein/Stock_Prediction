import os, sys, warnings, types
import numpy as np
import pandas as pd
import streamlit as st
import joblib
import tarfile
import tempfile
import boto3
import sagemaker
import __main__
from sagemaker.predictor import Predictor
from sagemaker.serializers import JSONSerializer
from sagemaker.deserializers import NumpyDeserializer
from sklearn.base import BaseEstimator, TransformerMixin

# --- 1. VIRTUAL MODULE HACK (Fixes ModuleNotFoundError for Custom Classes) ---
m = types.ModuleType('src.Custom_Classes')
sys.modules['src.Custom_Classes'] = m

# --- 2. CUSTOM CLASS DEFINITIONS (Must match your training notebook) ---
class DataCleaner(BaseEstimator, TransformerMixin):
    def __init__(self, missing_threshold=0.80):
        self.missing_threshold = missing_threshold
    def fit(self, X, y=None): return self
    def transform(self, X):
        X = pd.DataFrame(X).copy()
        X.columns = X.columns.str.strip().str.lower().str.replace(' ', '_')
        return X

class FraudFeatureExtractor(BaseEstimator, TransformerMixin):
    def __init__(self): self.amt_median_ = None
    def fit(self, X, y=None): return self
    def transform(self, X): return X

# Link classes to the fake module and __main__ for joblib/pickle safety
m.DataCleaner = DataCleaner
m.FraudFeatureExtractor = FraudFeatureExtractor
__main__.DataCleaner = DataCleaner
__main__.FraudFeatureExtractor = FraudFeatureExtractor

# --- 3. CONFIGURATION & DATA ---
st.set_page_config(page_title="Fraud Detection System", layout="wide")
warnings.simplefilter("ignore")

# Setup paths
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
file_path = os.path.join(project_root, 'Portfolio/X_train.csv')

# Load the template (380 columns)
@st.cache_data
def load_template():
    df = pd.read_csv(file_path, index_col=0)
    # Remove index/unnamed columns
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    return df

dataset = load_template()

# AWS Secrets
aws_credentials = st.secrets["aws_credentials"]
session = boto3.Session(
    aws_access_key_id=aws_credentials["AWS_ACCESS_KEY_ID"],
    aws_secret_access_key=aws_credentials["AWS_SECRET_ACCESS_KEY"],
    aws_session_token=aws_credentials["AWS_SESSION_TOKEN"],
    region_name='us-east-1'
)
sm_session = sagemaker.Session(boto_session=session)

# --- 4. PREDICTION LOGIC (FIXED: 328-COLUMN TRUNCATION) ---
def call_model_api(user_inputs):
    predictor = Predictor(
        endpoint_name=aws_credentials["AWS_ENDPOINT"],
        sagemaker_session=sm_session,
        serializer=JSONSerializer(),
        deserializer=NumpyDeserializer()
    )
    try:
        # 1. Grab a template row from your local CSV (380 columns)
        df_full = dataset.iloc[0:1].copy()
        
        # 2. Map user inputs to the template
        for key, val in user_inputs.items():
            if key in df_full.columns:
                df_full[key] = val
        
        # 3. THE FIX: Truncate to exactly 328 columns 
        # This matches the 'SelectKBest' requirement on the AWS side
        df_final = df_full.iloc[:, :328]
        
        # 4. Send to SageMaker as a record
        raw_pred = predictor.predict(df_final.to_dict(orient='records'))
        
        # 5. Process result
        pred_val = np.array(raw_pred).flatten()[0]
        label = "Fraudulent" if int(float(pred_val)) == 1 else "Legitimate"
        return label, 200
    except Exception as e:
        return f"AWS Endpoint Error: {str(e)}", 500

# --- 5. STREAMLIT UI ---
st.title("🛡️ Fraud Detection Deployment")
st.markdown("---")

with st.form("input_form"):
    st.subheader("Transaction Details")
    c1, c2 = st.columns(2)
    
    with c1:
        amt = st.number_input("TRANSACTION AMOUNT", value=100.00)
        hour = st.number_input("TRANSACTION HOUR (0-23)", value=12.0)
    with c2:
        high_amt = st.selectbox("IS HIGH TRANSACTION? (0=No, 1=Yes)", [0, 1])
        v92 = st.number_input("V92 FEATURE VALUE", value=0.0)
    
    submitted = st.form_submit_button("Analyze Transaction")

if submitted:
    # Match the keys exactly to your CSV column names
    user_data = {
        'transactionamt': amt,
        'transactionhour': hour,
        'hightransactionamt': high_amt,
        'v92': v92
    }
    
    with st.spinner("Analyzing transaction on AWS SageMaker..."):
        result, status = call_model_api(user_data)
        
    if status == 200:
        color = "red" if result == "Fraudulent" else "green"
        st.markdown(f"### Result: :{color}[{result}]")
    else:
        st.error(result)
        # Helpful reminder for Lab users
        if "ExpiredToken" in result:
            st.warning("Your AWS Session Token has expired. Please refresh your credentials in Streamlit secrets.")
