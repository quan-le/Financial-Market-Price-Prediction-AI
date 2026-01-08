import pandas as pd
import numpy as np
from pathlib import Path
import os


def safe_numeric_series(series):
    return pd.to_numeric(series, errors='coerce').fillna(0.0)


def calculate_ema(df, column='close', span=20):
    if column not in df.columns:
        return pd.Series(np.nan, index=df.index, name=f"{column}ema")
    return safe_numeric_series(df[column]).ewm(span=span, adjust=False).mean()


def calculate_rsi(df, column='close', period=14):
    if column not in df.columns:
        return pd.Series(np.nan, index=df.index, name=f"{column}rsi")
    numeric_series = safe_numeric_series(df[column])
    delta = numeric_series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50.0)


def calculate_macd(df, column='close', fast=12, slow=26):
    if column not in df.columns:
        return pd.Series(np.nan, index=df.index, name=f"{column}macd")
    numeric_series = safe_numeric_series(df[column])
    ema_fast = numeric_series.ewm(span=fast, adjust=False).mean()
    ema_slow = numeric_series.ewm(span=slow, adjust=False).mean()
    return (ema_fast - ema_slow).fillna(0.0)


def calculate_volatility(df, column='close', window=20):
    if column not in df.columns:
        return pd.Series(np.nan, index=df.index, name=f"{column}vol")
    numeric_series = safe_numeric_series(df[column])
    returns = numeric_series.pct_change()
    return returns.rolling(window=window).std().fillna(0.0)


def add_lag_features(df, column='close', lags=[1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144]):
    df = df.copy()
    numeric_close = safe_numeric_series(df[column])
    for lag in lags:
        df[f'closelag{lag}'] = numeric_close.shift(lag).fillna(0.0)
        df[f'retlag{lag}'] = numeric_close.pct_change(lag).fillna(0.0)
    return df


def prepare_merge_df(df, date_col='date'):
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.copy()
    if isinstance(df.index, pd.DatetimeIndex):
        df = df.reset_index()
        df.columns = ['date_temp'] + df.columns[1:].tolist()
        df['date'] = pd.to_datetime(df['date_temp'])
        df = df.drop('date_temp', axis=1)
    else:
        if date_col not in df.columns:
            if hasattr(df.index, 'name') and df.index.name == 'date':
                df = df.reset_index()
            else:
                df['date'] = pd.date_range(start='2000-01-01', periods=len(df), freq='D')
        else:
            df['date'] = pd.to_datetime(df[date_col])
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if 'date' in numeric_cols:
        numeric_cols.remove('date')
    return df[['date'] + numeric_cols]


def engineer_features(price_df, econ_df=None, fed_df=None):
    print("[DEBUG] Starting feature engineering...")
    df = price_df.copy()
    if df.empty:
        raise ValueError("Price dataframe is empty")

    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    print(f"[DEBUG] Price data shape: {df.shape}")

    core_cols = ['open', 'high', 'low', 'close', 'volume']
    for col in core_cols:
        if col in df.columns:
            df[col] = safe_numeric_series(df[col])

    df['ema34'] = calculate_ema(df, 'close', 34)
    df['ema89'] = calculate_ema(df, 'close', 89)
    df['ema200'] = calculate_ema(df, 'close', 200)
    df['rsi14'] = calculate_rsi(df, 'close', 14)
    df['macd'] = calculate_macd(df, 'close')
    df['logreturn'] = np.log(df['close'] / df['close'].shift(1)).fillna(0.0)
    df['vol20'] = calculate_volatility(df, 'close', 20)

    df = add_lag_features(df, 'close')

    df['nextclose'] = df['close'].shift(-1).fillna(df['close'].iloc[-1])

    print(f"[DEBUG] After technical features: {df.shape}")

    if econ_df is not None and not econ_df.empty:
        print("[DEBUG] Merging economic data...")
        econ_prep = prepare_merge_df(econ_df)
        if not econ_prep.empty:
            econ_rename = {
                'fed_funds_rate': 'fedfundsrate',
                'unemployment_rate': 'unemploymentrate',
                'nonfarm_payrolls': 'nonfarmpayrolls',
                'consumer_confidence': 'consumerconfidence',
                'building_permits': 'buildingpermits',
                'retail_sales': 'retailsales',
                'industrial_production': 'industrialproduction',
                'trade_balance': 'tradebalance',
                'money_supply_m1': 'moneysupplym1',
                'money_supply_m2': 'moneysupplym2',
                'pce_inflation': 'pceinflation'
            }
            econ_prep = econ_prep.rename(columns=econ_rename)
            df = pd.merge(df, econ_prep, on='date', how='left')
        print(f"[DEBUG] After econ merge: {df.shape}")

    if fed_df is not None and len(fed_df) > 0 and not fed_df.empty:
        print("[DEBUG] Merging Fed embeddings...")
        fed_prep = prepare_merge_df(fed_df)
        if not fed_prep.empty:
            fed_numeric_cols = [col for col in fed_prep.columns if col.startswith('fed_emb_')]
            if fed_numeric_cols:
                fed_numeric = fed_prep[['date'] + fed_numeric_cols]
                for col in fed_numeric_cols:
                    fed_numeric[col] = safe_numeric_series(fed_numeric[col])

                fed_numeric.columns = ['date'] + [col.replace('fed_emb_', 'fedemb') for col in fed_numeric_cols]
                df = pd.merge(df, fed_numeric, on='date', how='left')
        print(f"[DEBUG] After fed merge: {df.shape}")

    numeric_columns = df.select_dtypes(include=[np.number]).columns
    df[numeric_columns] = df[numeric_columns].ffill().bfill().fillna(0)

    df = df.dropna(thresh=len(df.columns) * 0.5)
    if len(df) < 10:
        raise ValueError(f"Insufficient data after cleaning: {len(df)} rows")

    print(f"[DEBUG] Final features shape: {df.shape}")
    print(f"[DEBUG] Columns sample: {df.columns[:10].tolist()}")
    return df


def prepare_model_input(df, lookback=89):
    observed_cols = [
        'open', 'high', 'low', 'close', 'volume',
        'ema34', 'ema89', 'ema200',
        'rsi14', 'macd', 'logreturn', 'vol20'
    ]
    observed_cols += [c for c in df.columns if c.startswith('closelag') or c.startswith('retlag')]

    known_cols = [
        'buildingpermits', 'consumerconfidence', 'cpi',
        'fedfundsrate', 'gdp', 'industrialproduction',
        'moneysupplym1', 'moneysupplym2',
        'nonfarmpayrolls', 'pceinflation',
        'ppi', 'retailsales', 'tradebalance',
        'unemploymentrate'
    ]
    known_cols += [c for c in df.columns if c.startswith('fedemb')]

    available_observed = [c for c in observed_cols if c in df.columns]
    available_known = [c for c in known_cols if c in df.columns]

    print(f"[DEBUG] Available observed cols: {len(available_observed)}")
    print(f"[DEBUG] Available known cols: {len(available_known)}")

    if len(available_observed) == 0:
        raise ValueError(f"No observed columns found! Expected columns like: {observed_cols[:5]}")

    if len(available_known) == 0:
        raise ValueError(f"No known columns found! Expected columns like: {known_cols[:5]}")

    if len(df) < lookback:
        pad_rows = lookback - len(df)
        pad_df = df.iloc[[-1] * pad_rows]
        df = pd.concat([df, pad_df], ignore_index=True)

    recent_data = df.iloc[-lookback:].copy()

    obs_data = recent_data[available_observed].fillna(0).values.astype(np.float32)
    known_data = recent_data[available_known].fillna(0).values.astype(np.float32)

    expected_n_known = int(os.getenv('MINI_TFT_NKNOWN', '782'))
    if known_data.shape[1] < expected_n_known:
        padding = np.zeros((known_data.shape[0], expected_n_known - known_data.shape[1]), dtype=np.float32)
        known_data = np.concatenate([known_data, padding], axis=1)
        print(f"[DEBUG] Padded known features from {known_data.shape[1] - padding.shape[1]} to {known_data.shape[1]}")

    static = np.array([1.0, 1.0], dtype=np.float32)

    print(f"[DEBUG] Final obs shape: {obs_data.shape}")
    print(f"[DEBUG] Final known shape: {known_data.shape}")

    return obs_data, known_data, static
