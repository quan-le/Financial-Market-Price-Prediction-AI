import pandas as pd
from pathlib import Path
from curl_cffi import requests
import json
def fetch_alpha_vantage(symbol, function="TIME_SERIES_DAILY", api_key="", outputsize="full"):
    url = f"https://www.alphavantage.co/query"
    params ={
        "function" : function,
        "symbol" : symbol,
        "apikey" : api_key,
        "outputsize" : outputsize,
        "datatype" : "json"
    }
    r = requests.get(url, params=params)
    print(r)
    data = r.json().get("Time Series (Daily)", {})
    print(data)
    df = pd.DataFrame(data).T
    df.reset_index(inplace=True)
    df.rename(columns={"index":"Date"}, inplace=True)
    new_columns = {}
    for col in df.columns:
        if col != 'Date' and '.' in col:
            # Splits '1. open' into ['1', ' open'], takes the second part (' open'), 
            # strips whitespace, and capitalizes the first letter ('Open')
            cleaned_name = col.split('. ')[-1].strip().capitalize()
            new_columns[col] = cleaned_name
        
    df.rename(columns=new_columns, inplace=True)
    df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
    df['Source'] = 'AlphaVantage'
    df['Symbol'] = symbol
    df.to_csv("../raw/alpha_BTC_full.csv", index=False)
    return df
#fetch_alpha_vantage("IBM", api_key="SU3Q55SZG98ZHARI").to_csv("../raw/alpha_BTC_full.csv", index=False)