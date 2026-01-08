import os
import time
import json
import requests
import pandas as pd
from datetime import datetime
import yfinance as yf
from pathlib import Path

RAW_DIR = os.getenv("RAW_DATA_DIR", "../data/raw")
ECON_DIR = os.getenv("ECON_DATA_DIR", "../data/econ")

os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(ECON_DIR, exist_ok=True)


class AlphaVantagePriceFetcher:
    BASE_URL = "https://www.alphavantage.co/query"

    def __init__(self, api_key_env="ALPHAVANTAGE_API_KEY"):
        self.api_key = os.environ.get(api_key_env, None)

    def _load_local_csv(self, symbol):
        csv_path = f"{RAW_DIR}/{symbol}.csv"
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path, parse_dates=["date"])
            return df
        return None

    def _save_csv(self, df: pd.DataFrame, symbol: str):
        df.to_csv(f"{RAW_DIR}/{symbol}.csv", index=False)

    def _fetch_yfinance(self, symbol: str):
        try:
            df = yf.download(symbol, period="max", interval="1d", progress=False)
            if df.empty:
                return None

            df = df.rename(columns={
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Volume": "volume"
            })
            df["date"] = df.index
            df["symbol"] = symbol
            df["source"] = "yfinance"
            df = df.reset_index(drop=True)
            return df
        except:
            return None

    def fetch(self, symbol: str, asset_type: str = "stock"):
        local_df = self._load_local_csv(symbol)
        if local_df is not None:
            return local_df

        df = self._fetch_yfinance(symbol)
        if df is not None:
            self._save_csv(df, symbol)
            return df

        return None


class EconIndicatorsFetcher:
    FRED_URL = "https://api.stlouisfed.org/fred/series/observations"

    def __init__(self, fred_key_env="FRED_API_KEY"):
        self.fred_key = os.environ.get(fred_key_env, None)

    def _local_csv(self, name):
        path = f"{ECON_DIR}/{name}.csv"
        if os.path.exists(path):
            return pd.read_csv(path, parse_dates=["date"])
        return None

    def _save_csv(self, df, name):
        df.to_csv(f"{ECON_DIR}/{name}.csv", index=False)

    def _fetch_fred(self, series_id: str, start_date: str = None):
        if not self.fred_key:
            return None

        params = {
            "series_id": series_id,
            "api_key": self.fred_key,
            "file_type": "json"
        }

        if start_date:
            params["observation_start"] = start_date

        r = requests.get(self.FRED_URL, params=params)
        if r.status_code != 200:
            return None

        data = r.json()
        if "observations" not in data:
            return None

        rows = []
        for obs in data["observations"]:
            date = obs["date"]
            val = obs["value"]
            try:
                val = float(val)
            except:
                continue
            rows.append({"date": pd.to_datetime(date), "value": val})

        df = pd.DataFrame(rows)
        df.sort_values("date", inplace=True)
        return df

    def _ffill_daily(self, df, start, end):
        daily = pd.DataFrame({"date": pd.date_range(start=start, end=end, freq="D")})
        merged = pd.merge(daily, df, on="date", how="left")
        merged["value"] = merged["value"].ffill()
        return merged

    def fetch_indicator(self, name: str, fred_series: str):
        local_df = self._local_csv(name)
        start_date = None

        if local_df is not None and not local_df.empty:
            last_date = local_df["date"].max()
            start_date = (last_date + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

        new_df = self._fetch_fred(fred_series, start_date=start_date)

        if new_df is not None and not new_df.empty:
            if local_df is not None and not local_df.empty:
                combined_df = pd.concat([local_df, new_df], ignore_index=True)
                combined_df = combined_df.drop_duplicates(subset=['date'], keep='last')
            else:
                combined_df = new_df

            combined_df.sort_values("date", inplace=True)
            self._save_csv(combined_df, name)
            return combined_df

        if local_df is not None:
            return local_df

        empty = pd.DataFrame({"date": [], "value": []})
        self._save_csv(empty, name)
        return empty

    def fetch_all(self):
        indicators = {
            "fed_funds_rate": "FEDFUNDS",
            "unemployment_rate": "UNRATE",
            "nonfarm_payrolls": "PAYEMS",
            "cpi": "CPIAUCSL",
            "ppi": "PPIACO",
            "gdp": "GDPC1",
            "retail_sales": "RSXFS",
            "industrial_production": "INDPRO",
            "building_permits": "PERMIT",
            "consumer_confidence": "UMCSENT",
            "trade_balance": "NETEXC",
            "money_supply_m1": "M1SL",
            "money_supply_m2": "M2SL",
            "pce_inflation": "PCEPI",
        }

        out = {}
        for name, series_id in indicators.items():
            df = self.fetch_indicator(name, series_id)
            out[name] = df

        return out

    def generate_daily_econ(self, start="2000-01-01", end=None):
        if end is None:
            end = datetime.today().strftime("%Y-%m-%d")

        raw = self.fetch_all()

        daily_frames = []
        for name, df in raw.items():
            if df.empty:
                continue
            ddf = self._ffill_daily(df, start, end)
            ddf["indicator"] = name
            daily_frames.append(ddf)

        if not daily_frames:
            return None

        long_df = pd.concat(daily_frames, axis=0).sort_values("date")
        wide_df = long_df.pivot(index="date", columns="indicator", values="value").sort_index().ffill()

        out_path = f"{ECON_DIR}/econ_wide_daily.csv"
        wide_df.to_csv(out_path)

        return wide_df
