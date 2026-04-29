import streamlit as st
import pandas as pd
import numpy as np
import boto3
import sagemaker
from sagemaker.predictor import Predictor
from sagemaker.serializers import JSONSerializer
from sagemaker.deserializers import NumpyDeserializer

# 1. SIMPLE AWS SETUP
aws_id = st.secrets["aws_credentials"]["AWS_ACCESS_KEY_ID"]
aws_secret = st.secrets["aws_credentials"]["AWS_SECRET_ACCESS_KEY"]
aws_token = st.secrets["aws_credentials"]["AWS_SESSION_TOKEN"]
aws_endpoint = st.secrets["aws_credentials"]["AWS_ENDPOINT"]

session = boto3.Session(aws_access_key_id=aws_id, aws_secret_access_key=aws_secret, 
                        aws_session_token=aws_token, region_name='us-east-1')
sm_session = sagemaker.Session(boto_session=session)

# 2. UI
st.title("🚨 Emergency Test Mode")
val1 = st.number_input("TRANSACTION AMT", value=100.0)
val2 = st.number_input("TRANSACTION HOUR", value=12.0)
val3 = st.number_input("HIGH TRANS AMT (0 or 1)", value=0.0)
val4 = st.number_input("V92", value=0.0)

if st.button("Test Prediction"):
    predictor = Predictor(endpoint_name=aws_endpoint, sagemaker_session=sm_session,
                          serializer=JSONSerializer(), deserializer=NumpyDeserializer())
    
    # We send JUST the 4 values as a simple list of records
    # If the model crashes here, the problem is the Pipeline on AWS
    test_data = [{"transactionamt": val1, "transactionhour": val2, 
                  "hightransactionamt": val3, "v92": val4}]
    
    try:
        raw_pred = predictor.predict(test_data)
        res = np.array(raw_pred).flatten()[0]
        st.success(f"It worked! Prediction: {res}")
    except Exception as e:
        st.error(f"Endpoint Still Failing: {e}")
        st.info("If this failed, you need to re-deploy the model from your notebook with 'input_shape' fixed.")
