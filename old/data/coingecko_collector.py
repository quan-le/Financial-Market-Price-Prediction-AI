import pandas as pd
from pathlib import Path
from curl_cffi import requests
import json
from pycoingecko import CoinGeckoAPI
from datetime import datetime

def fetch_coingecko(coin_id="bitcoin", days="max", vs_currency="usd"):
    cg = CoinGeckoAPI()        
    #Leaving blank mean we are using demo api key - max 1 year data
    # CoinGeckoAPI(demo_api_key = "")
    # CoinGeckoAPI(api_key = "")
    data = cg.get_coin_ohlc_by_id(id=coin_id, vs_currency=vs_currency, days=days)
    df = pd.DataFrame(data, columns=['Timestamp', 'Open', 'High', 'Low', 'Close'])
    df['Date'] = pd.to_datetime(df['Timestamp'], unit='ms').dt.date
    df = df[['Date', 'Close', 'High', 'Low', 'Open']]
    df['Source'] = 'Coingecko'
    df['Symbol'] = coin_id.upper()
    df.to_csv("../raw/coingecko_bitcoin_max.csv", index=False)
    return df
#fetch_coingecko(coin_id="bitcoin", days="365").to_csv("../raw/coingecko_bitcoin_max.csv", index=False)