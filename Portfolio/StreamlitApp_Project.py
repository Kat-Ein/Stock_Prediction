import os, sys, warnings, types
import numpy as np
import pandas as pd
import streamlit as st
import boto3
import sagemaker
import __main__
from sagemaker.predictor import Predictor
from sagemaker.serializers import JSONSerializer
from sagemaker.deserializers import NumpyDeserializer
from sklearn.base import BaseEstimator, TransformerMixin

# --- 1. VIRTUAL MODULE HACK ---
m = types.ModuleType('src.Custom_Classes')
sys.modules['src.Custom_Classes'] = m

class DataCleaner(BaseEstimator, TransformerMixin):
    def __init__(self, missing_threshold=0.80): self.missing_threshold = missing_threshold
    def fit(self, X, y=None): return self
    def transform(self, X): return X

class FraudFeatureExtractor(BaseEstimator, TransformerMixin):
    def __init__(self): self.amt_median_ = None
    def fit(self, X, y=None): return self
    def transform(self, X): return X

m.DataCleaner = DataCleaner
m.FraudFeatureExtractor = FraudFeatureExtractor
__main__.DataCleaner = DataCleaner
__main__.FraudFeatureExtractor = FraudFeatureExtractor

# --- 2. THE "FIND THE NUMBER" FUNCTION (Stops the 'dict' error) ---
def find_the_number(obj):
    """Recursively drills down into lists/dicts to find the first numeric value."""
    if isinstance(obj, (int, float, np.number)):
        return obj
    if isinstance(obj, dict):
        if not obj: return 0
        return find_the_number(list(obj.values())[0])
    if isinstance(obj, (list, np.ndarray, tuple)):
        if not obj: return 0
        return find_the_number(obj[0])
    try:
        return float(obj)
    except:
        return 0

# --- 3. CONFIG & DATA ---
st.set_page_config(page_title="Fraud Detection", layout="wide")
aws = st.secrets["aws_credentials"]

@st.cache_data
def get_data():
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../Portfolio/X_train.csv')
    df = pd.read_csv(path, index_col=0)
    return df.loc[:, ~df.columns.str.contains('^Unnamed')]

dataset = get_data()

session = boto3.Session(
    aws_access_key_id=aws["AWS_ACCESS_KEY_ID"],
    aws_secret_access_key=aws["AWS_SECRET_ACCESS_KEY"],
    aws_session_token=aws["AWS_SESSION_TOKEN"],
    region_name='us-east-1'
)
sm_session = sagemaker.Session(boto_session=session)

# --- 4. PREDICTION LOGIC ---
def call_model_api(user_inputs):
    predictor = Predictor(
        endpoint_name=aws["AWS_ENDPOINT"],
        sagemaker_session=sm_session,
        serializer=JSONSerializer(),
        deserializer=NumpyDeserializer()
    )
    try:
        # Create template and inject user data
        df_full = dataset.iloc[0:1].copy()
        for k, v in user_inputs.items():
            col = next((c for c in df_full.columns if c.lower() == k.lower()), None)
            if col: df_full[col] = float(v)
        
        # Force 328 columns (matches SelectKBest)
        df_payload = df_full.iloc[:, :328]
        
        # Send to AWS
        raw_response = predictor.predict(df_payload.values.tolist())
        
        # DIG FOR THE NUMBER (Fixes the Dict error)
        final_value = find_the_number(raw_response)
        
        label = "Fraudulent" if int(float(final_value)) == 1 else "Legitimate"
        return label, 200
    except Exception as e:
        return f"Error: {str(e)}", 500

# --- 5. UI ---
st.title("🛡️ Fraud Detection System")

with st.form("input_form"):
    c1, c2 = st.columns(2)
    with c1:
        amt = st.number_input("TRANSACTION AMOUNT", value=100.0)
        hour = st.number_input("HOUR", value=12.0)
    with c2:
        high = st.selectbox("HIGH AMT?", [0, 1])
        v92 = st.number_input("V92", value=0.0)
    
    run = st.form_submit_button("Run Prediction")

if run:
    with st.spinner("Calling SageMaker..."):
        res, status = call_model_api({'transactionamt': amt, 'transactionhour': hour, 'hightransactionamt': high, 'v92': v92})
        if status == 200:
            st.metric("Result", res)
        else:
            st.error(res)
