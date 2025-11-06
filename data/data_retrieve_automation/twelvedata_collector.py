import pandas as pd
from pathlib import Path
from curl_cffi import requests
import json
from twelvedata import TDClient

def fetch_twelvedata(symbol, api_key="", interval="1day", outputsize=5000):
    td = TDClient(apikey=api_key)
    data = td.time_series(
        symbol=symbol,
        interval=interval,
        outputsize=outputsize,
    ).as_pandas()
    data.index.name = 'Date'
    data.reset_index(inplace=True)
    data.columns = [col.capitalize() if col != 'Date' else col for col in data.columns]
    data['Source'] = 'TwelveData'
    data['Symbol'] = symbol
    df = data[['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Source', 'Symbol']]
    print(df.info())
    df.to_csv("../raw/twelvedata_BTC_full.csv", index=False)
    return df
'''
data = fetch_twelvedata("BTC", api_key="edbd69cba18945279ad897fb11726802")
data.to_csv("../raw/twelvedata_BTC_full.csv", index=False)
'''