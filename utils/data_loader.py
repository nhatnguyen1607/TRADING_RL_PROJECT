import os
import shutil
import tempfile

def _ensure_ascii_cert_bundle():
    try:
        import certifi

        cert_path = certifi.where()
        temp_dir = tempfile.gettempdir()
        ascii_cert = os.path.join(temp_dir, "cacert.pem")
        if not os.path.exists(ascii_cert):
            shutil.copy(cert_path, ascii_cert)

        os.environ.setdefault("REQUESTS_CA_BUNDLE", ascii_cert)
        os.environ.setdefault("CURL_CA_BUNDLE", ascii_cert)
        os.environ.setdefault("SSL_CERT_FILE", ascii_cert)
    except Exception:
        pass

_ensure_ascii_cert_bundle()

import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


DEFAULT_MACRO_TICKERS = {
    "US10Y": "^TNX",
    "US2Y": "^UST2Y",
    "DXY": "DX-Y.NYB",
}

def calculate_technical_indicators(df):
    # 1. Các đường trung bình
    df['SMA_5'] = df['Close'].rolling(window=5).mean()
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    
    # 2. MACD
    ema_12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema_26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema_12 - ema_26
    
    # 3. RSI
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))
    df['RSI'] = df['RSI'].fillna(100)
    
    # ==========================================
    # 🚀 CẢI TIẾN: STATIONARY FEATURES (DỮ LIỆU TĨNH)
    # Thay vì dùng giá tuyệt đối, ta tính tỷ lệ % và khoảng cách
    # ==========================================
    df['Return'] = df['Close'].pct_change() # Tỷ suất sinh lời hằng ngày
    df['Dist_SMA20'] = (df['Close'] / df['SMA_20']) - 1 # Càng dương thì giá càng nằm xa trên SMA20
    df['Dist_SMA5'] = (df['Close'] / df['SMA_5']) - 1
    df['Dist_SMA50'] = (df['Close'] / df['SMA_50']) - 1
    df['Momentum_10'] = df['Close'].pct_change(10)
    df['Momentum_20'] = df['Close'].pct_change(20)
    df['Momentum_60'] = df['Close'].pct_change(60)
    df['Volatility_20'] = df['Return'].rolling(window=20).std()
    df['Raw_Volatility_20'] = df['Volatility_20']
    df['SMA20_Slope'] = df['SMA_20'].pct_change(5)
    df['SMA50_Slope'] = df['SMA_50'].pct_change(10)
    df['Trend_Regime'] = np.where(
        (df['Close'] > df['SMA_20']) & (df['SMA_20'] > df['SMA_50']),
        1.0,
        np.where((df['Close'] < df['SMA_20']) & (df['SMA_20'] < df['SMA_50']), -1.0, 0.0)
    )
    df['Fast_Trend_Regime'] = np.where(
        (df['Close'] > df['SMA_5']) & (df['SMA_5'] > df['SMA_20']),
        1.0,
        np.where((df['Close'] < df['SMA_5']) & (df['SMA_5'] < df['SMA_20']), -1.0, 0.0)
    )
    
    df['Open_rel'] = (df['Open'] / df['Close']) - 1
    df['High_rel'] = (df['High'] / df['Close']) - 1
    df['Low_rel'] = (df['Low'] / df['Close']) - 1
    
    # Khối lượng tương đối so với trung bình 20 ngày
    vol_sma_20 = df['Volume'].rolling(window=20).mean()
    df['Vol_rel'] = (df['Volume'] / (vol_sma_20 + 1e-8)) - 1
    df['VIX_Change'] = df['VIX'].pct_change()
    
    return df


def _download_close_series(ticker, start, end, name):
    try:
        data = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=False)
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        if data.empty or "Close" not in data.columns:
            print(f"Macro source skipped: {name} ({ticker}) returned no Close data.")
            return None
        series = data["Close"].rename(name).astype(float)
        return series
    except Exception as exc:
        print(f"Macro source skipped: {name} ({ticker}) failed with {exc}")
        return None


def load_macro_features(start, end, macro_tickers=None):
    """Load macro regime variables and transform them into stationary features."""
    macro_tickers = macro_tickers or DEFAULT_MACRO_TICKERS
    series = []
    for name, ticker in macro_tickers.items():
        close = _download_close_series(ticker, start, end, name)
        if close is not None:
            series.append(close)

    if not series:
        return pd.DataFrame(), []

    macro = pd.concat(series, axis=1).sort_index().ffill()
    feature_cols = []

    for col in list(macro.columns):
        macro[f"Macro_{col}_Level"] = macro[col]
        macro[f"Macro_{col}_Change"] = macro[col].pct_change()
        macro[f"Macro_{col}_Z20"] = (macro[col] - macro[col].rolling(20).mean()) / (macro[col].rolling(20).std() + 1e-8)
        feature_cols.extend([f"Macro_{col}_Level", f"Macro_{col}_Change", f"Macro_{col}_Z20"])

    if {"US10Y", "US2Y"}.issubset(macro.columns):
        macro["Macro_YieldCurve_10Y2Y"] = macro["US10Y"] - macro["US2Y"]
        macro["Macro_YieldCurve_Inverted"] = (macro["Macro_YieldCurve_10Y2Y"] < 0).astype(float)
        feature_cols.extend(["Macro_YieldCurve_10Y2Y", "Macro_YieldCurve_Inverted"])

    return macro[feature_cols], feature_cols

def load_and_preprocess_data(ticker="SPY", start="2015-01-01", end="2023-01-01", scale=True, include_macro=True):
    print(f"Downloading data for {ticker} and ^VIX from Yahoo Finance...")

    cache_dir = os.path.join(os.getcwd(), "data", "yf_cache")
    os.makedirs(cache_dir, exist_ok=True)
    if hasattr(yf, "set_tz_cache_location"):
        yf.set_tz_cache_location(cache_dir)
    
    df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
        
    vix = yf.download("^VIX", start=start, end=end, progress=False, auto_adjust=False)
    if isinstance(vix.columns, pd.MultiIndex):
        vix.columns = vix.columns.get_level_values(0)

    if df.empty or vix.empty:
        raise ValueError(
            f"Yahoo Finance returned empty data for {ticker} or ^VIX. "
            "Check network access/cache permissions before training."
        )
        
    df['VIX'] = vix['Close']
    df = calculate_technical_indicators(df)
    macro_cols = []
    if include_macro:
        macro_df, macro_cols = load_macro_features(start, end)
        if not macro_df.empty:
            df = df.join(macro_df, how="left").ffill()
    df.dropna(inplace=True)
    
    # 🚀 CHỈ ĐƯA CÁC ĐẶC TRƯNG TĨNH (STATIONARY) VÀO CHO AI HỌC
    # Loại bỏ hoàn toàn giá gốc (Open, High, Low, Close) khỏi mắt mô hình
    feature_cols = [
        'Return', 'Dist_SMA20', 'Dist_SMA50', 
        'Open_rel', 'High_rel', 'Low_rel', 'Vol_rel',
        'RSI', 'MACD', 'VIX', 'VIX_Change',
        'Momentum_10', 'Momentum_20', 'Momentum_60',
        'Volatility_20', 'SMA20_Slope', 'SMA50_Slope',
        'Trend_Regime'
    ] + macro_cols
    
    scaler = None
    if scale:
        scaler = StandardScaler()
        df[feature_cols] = scaler.fit_transform(df[feature_cols])
    
    # Lưu lại danh sách tên cột feature để Môi trường (Env) biết mà lấy
    df.attrs['feature_cols'] = feature_cols 
    
    print("Data processed successfully! State size:", df[feature_cols].shape)
    return df, scaler


def load_multi_asset_data(tickers=None, start="2015-01-01", end="2023-01-01", scale=True, include_macro=True):
    if tickers is None:
        tickers = ["SPY", "SH", "TLT"]

    print(f"Downloading multi-asset data for {', '.join(tickers)} and ^VIX from Yahoo Finance...")

    cache_dir = os.path.join(os.getcwd(), "data", "yf_cache")
    os.makedirs(cache_dir, exist_ok=True)
    if hasattr(yf, "set_tz_cache_location"):
        yf.set_tz_cache_location(cache_dir)

    vix = yf.download("^VIX", start=start, end=end, progress=False, auto_adjust=False)
    if isinstance(vix.columns, pd.MultiIndex):
        vix.columns = vix.columns.get_level_values(0)
    if vix.empty:
        raise ValueError("Yahoo Finance returned empty data for ^VIX.")

    asset_frames = []
    feature_cols = []
    close_cols = []
    macro_df, macro_cols = (load_macro_features(start, end) if include_macro else (pd.DataFrame(), []))

    for ticker in tickers:
        raw = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=False)
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        if raw.empty:
            raise ValueError(f"Yahoo Finance returned empty data for {ticker}.")

        asset_df = raw[["Open", "High", "Low", "Close", "Volume"]].copy()
        asset_df["VIX"] = vix["Close"]
        asset_df = calculate_technical_indicators(asset_df)

        close_col = f"Close_{ticker}"
        asset_df[close_col] = asset_df["Close"]
        close_cols.append(close_col)

        base_features = [
            "Return", "Dist_SMA20", "Dist_SMA50",
            "Open_rel", "High_rel", "Low_rel", "Vol_rel",
            "RSI", "MACD", "VIX", "VIX_Change",
            "Momentum_10", "Momentum_20", "Momentum_60",
            "Volatility_20", "SMA20_Slope", "SMA50_Slope",
            "Trend_Regime",
        ]
        rename_map = {col: f"{ticker}_{col}" for col in base_features}
        renamed_features = list(rename_map.values())
        feature_cols.extend(renamed_features)

        keep_cols = [close_col] + base_features
        asset_frames.append(asset_df[keep_cols].rename(columns=rename_map))

    df = pd.concat(asset_frames, axis=1)
    if include_macro and not macro_df.empty:
        df = df.join(macro_df, how="left").ffill()
        feature_cols.extend(macro_cols)
    df = df.dropna()
    scaler = None
    if scale:
        scaler = StandardScaler()
        df[feature_cols] = scaler.fit_transform(df[feature_cols])

    df.attrs["feature_cols"] = feature_cols
    df.attrs["asset_cols"] = close_cols
    df.attrs["tickers"] = tickers

    print("Multi-asset data processed successfully! State size:", df[feature_cols].shape)
    return df, scaler
