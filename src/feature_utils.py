import numpy as np
import pandas as pd
import datetime
import yfinance as yf
import pandas_datareader.data as web
import requests
#from datetime import datetime, timedelta
import os
import sys

import os
import sys


# ... continue with your script ...

def extract_features():

    return_period = 5
    
    START_DATE = (datetime.date.today() - datetime.timedelta(days=365)).strftime("%Y-%m-%d")
    END_DATE = datetime.date.today().strftime("%Y-%m-%d")
    stk_tickers = ['AAPL', 'PANW', 'UBER']
    ccy_tickers = ['DEXJPUS', 'DEXUSUK']
    idx_tickers = ['SP500', 'DJIA', 'VIXCLS']
    
    stk_data = yf.download(stk_tickers, start=START_DATE, end=END_DATE, auto_adjust=False)
    #stk_data = web.DataReader(stk_tickers, 'yahoo')
    ccy_data = web.DataReader(ccy_tickers, 'fred', start=START_DATE, end=END_DATE)
    idx_data = web.DataReader(idx_tickers, 'fred', start=START_DATE, end=END_DATE)

    Y = np.log(stk_data.loc[:, ('Adj Close', 'AAPL')]).diff(return_period).shift(-return_period)
    Y.name = Y.name[-1]+'_Future'
    
    X1 = np.log(stk_data.loc[:, ('Adj Close', ('PANW', 'UBER'))]).diff(return_period)
    X1.columns = X1.columns.droplevel()
    X2 = np.log(ccy_data).diff(return_period)
    X3 = np.log(idx_data).diff(return_period)

    X = pd.concat([X1, X2, X3], axis=1)
    
    dataset = pd.concat([Y, X], axis=1).dropna().iloc[::return_period, :]
    Y = dataset.loc[:, Y.name]
    X = dataset.loc[:, X.columns]
    dataset.index.name = 'Date'

    target = "AAPL"
    
    # Pull AAPL OHLC data
    open_  = stk_data.loc[:, ("Open", target)]
    high   = stk_data.loc[:, ("High", target)]
    low    = stk_data.loc[:, ("Low", target)]
    close  = stk_data.loc[:, ("Adj Close", target)]
    
    # (1) High - Low (daily range)
    hl_diff = (high - low).rename("HL_diff")
    
    # (2) Open - Close (directional move)
    oc_diff = (open_ - close).rename("OC_diff")
    
    # (3) High - Close (upper wick)
    hc_diff = (high - close).rename("HC_diff")
    
    # (4) Close - Low (lower wick)
    cl_diff = (close - low).rename("CL_diff")
    
    # (5) End of quarter flag
    end_of_quarter = close.index.to_series().dt.is_quarter_end.astype(int).rename("end_of_quarter")
    
    # Combine new features
    X_extra = pd.concat(
        [hl_diff, oc_diff, hc_diff, cl_diff, end_of_quarter],
        axis=1
    )
    
    # Add them to existing predictors
    X = pd.concat([X, X_extra], axis=1)
    
    # Rebuild dataset (drop NaNs and re-sample)
    dataset = pd.concat([Y, X], axis=1).dropna().iloc[::return_period, :]


    
    #dataset.to_csv(r"./test_data.csv")
    features = dataset.sort_index()
    features = features.reset_index(drop=True)
    features = features.iloc[:,1:]
    return features


def get_bitcoin_historical_prices(days = 60):
    
    BASE_URL = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart"
    
    params = {
        'vs_currency': 'usd',
        'days': days,
        'interval': 'daily' # Ensure we get daily granularity
    }
    response = requests.get(BASE_URL, params=params)
    data = response.json()
    prices = data['prices']
    df = pd.DataFrame(prices, columns=['Timestamp', 'Close Price (USD)'])
    df['Date'] = pd.to_datetime(df['Timestamp'], unit='ms').dt.normalize()
    df = df[['Date', 'Close Price (USD)']].set_index('Date')
    return df




