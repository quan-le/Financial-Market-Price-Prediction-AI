import pandas as pd
from pathlib import Path
from curl_cffi import requests
import json
import investpy

def fetch_investing_crypto(symbol, country="united states"):
    df = investpy.get_crypto_historical_data(
        crypto=symbol, 
        from_date="01/01/2020", 
        to_date="31/12/2020", 
        interval='daily',)
    df.reset_index(inplace=True)
    df['Source'] = 'Investing.com'
    df['Symbol'] = symbol.upper()
    
    output_filename = f"investing_com_{symbol.upper()}.csv"
    output_dir = Path("../raw") # Define the directory path object
    output_path = output_dir / output_filename # Combine path objects
    df.to_csv(output_path, index=False)
    return df

'''
data = fetch_investing_crypto("ethereum")
print("success")
print(data)
data.to_csv("../raw/investing_com_BTC_full.csv", index=False)
'''
