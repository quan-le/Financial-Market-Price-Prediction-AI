import pandas as pd
from pathlib import Path
from curl_cffi import requests
import json
import yfinance as yf

def fetch_yahoo(symbol: str, start="2008-01-01", end = None):
    session = requests.Session(impersonate="chrome")
    df = yf.download(
        symbol, 
        start=start, 
        end=end, 
        interval="1d", 
        session=session,            # Need this to avoid sever block
        keepna=False, timeout = 1)
    #we can add interval variables later if needed
    df.reset_index(inplace=True)
    df['Source'] = 'Yahoo'
    df['Symbol'] = symbol
    #df.to_csv()(f"../raw/yahoo_{symbol}_full.csv", index=False)
    return df
    