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


def _normalize_ohlcv(df):
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    rename = {col: str(col).strip() for col in df.columns}
    df = df.rename(columns=rename)
    required = ["Open", "High", "Low", "Close", "Volume"]
    if not all(col in df.columns for col in required):
        return pd.DataFrame()
    out = df[required].copy()
    for col in required:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.dropna(subset=["Close"]).sort_index()
    return out


def _read_local_price_file(path):
    if not os.path.exists(path):
        return pd.DataFrame()

    if os.path.basename(path).endswith("_data.csv"):
        df = pd.read_csv(path, sep=";", decimal=",", na_values=["#N/A N/A", "#N/A", "N/A"])
        df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
        df = df.rename(
            columns={
                "PX_OPEN": "Open",
                "PX_HIGH": "High",
                "PX_LOW": "Low",
                "PX_LAST": "Close",
                "PX_VOLUME": "Volume",
            }
        )
    else:
        df = pd.read_csv(path)
        if "Price" in df.columns and "Date" not in df.columns:
            df = pd.read_csv(path, skiprows=[1]).rename(columns={"Price": "Date"})
        if "Date" not in df.columns:
            return pd.DataFrame()
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    df = df.dropna(subset=["Date"]).set_index("Date")
    return _normalize_ohlcv(df)


def _load_local_price_data(ticker, start, end):
    candidates = [
        os.path.join("data", "external", f"{ticker}_price_from_options.csv"),
        os.path.join("data", "external", f"{ticker}_data.csv"),
        os.path.join("data", "external", f"{ticker.lower()}_eod_2015_2023.csv"),
        os.path.join("data", "external", f"{ticker}_eod_2015_2023.csv"),
    ]
    for path in candidates:
        local = _read_local_price_file(path)
        if local.empty:
            continue
        local = local.loc[(local.index >= pd.Timestamp(start)) & (local.index < pd.Timestamp(end))]
        if not local.empty:
            print(f"Using local price data for {ticker}: {path}")
            return local
    return pd.DataFrame()


def _download_or_load_price_data(ticker, start, end):
    try:
        data = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=False)
        data = _normalize_ohlcv(data)
        if not data.empty:
            return data
    except Exception as exc:
        print(f"Yahoo source skipped for {ticker}: {exc}")

    local = _load_local_price_data(ticker, start, end)
    if not local.empty:
        return local
    return pd.DataFrame()


def _vix_or_realized_proxy(vix_df, asset_df, asset_name="SPY"):
    if vix_df is not None and not vix_df.empty and "Close" in vix_df.columns:
        return vix_df["Close"]

    print(
        f"Warning: ^VIX unavailable; using {asset_name} rolling realized volatility "
        "as a local fallback for this run."
    )
    proxy = asset_df["Close"].pct_change(fill_method=None).rolling(20).std() * np.sqrt(252) * 100
    return proxy.bfill().ffill()


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
        macro[f"Macro_{col}_Change"] = macro[col].pct_change()
        macro[f"Macro_{col}_Momentum_20"] = macro[col].pct_change(20)
        macro[f"Macro_{col}_Z20"] = (macro[col] - macro[col].rolling(20).mean()) / (macro[col].rolling(20).std() + 1e-8)
        feature_cols.extend([f"Macro_{col}_Change", f"Macro_{col}_Momentum_20", f"Macro_{col}_Z20"])

    if {"US10Y", "US2Y"}.issubset(macro.columns):
        macro["Macro_YieldCurve_10Y2Y"] = macro["US10Y"] - macro["US2Y"]
        macro["Macro_YieldCurve_10Y2Y_Z20"] = (
            macro["Macro_YieldCurve_10Y2Y"] - macro["Macro_YieldCurve_10Y2Y"].rolling(20).mean()
        ) / (macro["Macro_YieldCurve_10Y2Y"].rolling(20).std() + 1e-8)
        macro["Macro_YieldCurve_Inverted"] = (macro["Macro_YieldCurve_10Y2Y"] < 0).astype(float)
        feature_cols.extend(["Macro_YieldCurve_10Y2Y", "Macro_YieldCurve_10Y2Y_Z20", "Macro_YieldCurve_Inverted"])

    risk_components = []
    if "Macro_US10Y_Momentum_20" in macro.columns:
        risk_components.append((macro["Macro_US10Y_Momentum_20"] > 0.02).astype(float))
    if "Macro_DXY_Momentum_20" in macro.columns:
        risk_components.append((macro["Macro_DXY_Momentum_20"] > 0.015).astype(float))
    if "Macro_YieldCurve_Inverted" in macro.columns:
        risk_components.append(macro["Macro_YieldCurve_Inverted"])
    if "Macro_YieldCurve_10Y2Y_Z20" in macro.columns:
        risk_components.append((macro["Macro_YieldCurve_10Y2Y_Z20"] < -0.75).astype(float))

    helper_cols = []
    if risk_components:
        macro["Macro_Risk_Off_Raw"] = pd.concat(risk_components, axis=1).mean(axis=1)
        helper_cols.append("Macro_Risk_Off_Raw")

    return macro[feature_cols + helper_cols], feature_cols


def load_sentiment_features(sentiment_path):
    """
    Load precomputed daily sentiment features.

    Expected CSV schema:
    - date column named one of: Date, date, datetime
    - one or more numeric sentiment columns, e.g. FinBERT_Positive,
      FinBERT_Negative, FinBERT_Neutral, FinBERT_Score.
    """
    if not sentiment_path:
        return pd.DataFrame(), []
    if not os.path.exists(sentiment_path):
        print(f"Sentiment source skipped: {sentiment_path} does not exist.")
        return pd.DataFrame(), []

    sentiment = pd.read_csv(sentiment_path)
    date_col = next((col for col in ("Date", "date", "datetime") if col in sentiment.columns), None)
    if date_col is None:
        raise ValueError("Sentiment CSV must contain a Date/date/datetime column.")

    sentiment[date_col] = pd.to_datetime(sentiment[date_col])
    sentiment = sentiment.set_index(date_col).sort_index()
    numeric_cols = [col for col in sentiment.columns if pd.api.types.is_numeric_dtype(sentiment[col])]
    feature_cols = []
    for col in numeric_cols:
        out_col = f"Sentiment_{col}"
        sentiment[out_col] = sentiment[col].astype(float)
        feature_cols.append(out_col)

    if not feature_cols:
        return pd.DataFrame(), []
    return sentiment[feature_cols], feature_cols


def load_options_features(options_path):
    """Load precomputed daily options-derived features."""
    if not options_path:
        return pd.DataFrame(), []
    if not os.path.exists(options_path):
        print(f"Options source skipped: {options_path} does not exist.")
        return pd.DataFrame(), []

    options = pd.read_csv(options_path)
    date_col = next((col for col in ("Date", "date", "datetime") if col in options.columns), None)
    if date_col is None:
        raise ValueError("Options CSV must contain a Date/date/datetime column.")

    options[date_col] = pd.to_datetime(options[date_col])
    options = options.set_index(date_col).sort_index()
    feature_cols = [
        col for col in options.columns
        if col.startswith("Options_") and pd.api.types.is_numeric_dtype(options[col])
    ]
    if not feature_cols:
        return pd.DataFrame(), []
    return options[feature_cols], feature_cols


def _join_external_features(df, external_df, feature_cols):
    if external_df.empty:
        return df
    return df.join(external_df, how="left").ffill()

def load_and_preprocess_data(
    ticker="SPY",
    start="2015-01-01",
    end="2023-01-01",
    scale=True,
    include_macro=True,
    sentiment_path=None,
    options_path=None,
):
    print(f"Downloading data for {ticker} and ^VIX from Yahoo Finance...")

    cache_dir = os.path.join(os.getcwd(), "data", "yf_cache")
    os.makedirs(cache_dir, exist_ok=True)
    if hasattr(yf, "set_tz_cache_location"):
        yf.set_tz_cache_location(cache_dir)
    
    df = _download_or_load_price_data(ticker, start, end)
        
    vix = yf.download("^VIX", start=start, end=end, progress=False, auto_adjust=False)
    if isinstance(vix.columns, pd.MultiIndex):
        vix.columns = vix.columns.get_level_values(0)

    if df.empty:
        raise ValueError(
            f"Yahoo Finance returned empty data for {ticker}. "
            "Check network access/cache permissions before training."
        )
        
    df['VIX'] = _vix_or_realized_proxy(vix, df, ticker)
    df = calculate_technical_indicators(df)
    macro_cols = []
    if include_macro:
        macro_df, macro_cols = load_macro_features(start, end)
        if not macro_df.empty:
            df = df.join(macro_df, how="left").ffill()
    sentiment_df, sentiment_cols = load_sentiment_features(sentiment_path)
    if not sentiment_df.empty:
        df = df.join(sentiment_df, how="left").ffill()
    options_df, options_cols = load_options_features(options_path)
    if not options_df.empty:
        df = _join_external_features(df, options_df, options_cols)
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
    ] + macro_cols + sentiment_cols + options_cols
    
    scaler = None
    if scale:
        scaler = StandardScaler()
        df[feature_cols] = scaler.fit_transform(df[feature_cols])
    
    # Lưu lại danh sách tên cột feature để Môi trường (Env) biết mà lấy
    df.attrs['feature_cols'] = feature_cols 
    
    print("Data processed successfully! State size:", df[feature_cols].shape)
    return df, scaler


def load_multi_asset_data(
    tickers=None,
    start="2015-01-01",
    end="2023-01-01",
    scale=True,
    include_macro=True,
    sentiment_path=None,
    options_path=None,
):
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
        vix = None

    asset_frames = []
    feature_cols = []
    close_cols = []
    macro_df, macro_cols = (load_macro_features(start, end) if include_macro else (pd.DataFrame(), []))
    sentiment_df, sentiment_cols = load_sentiment_features(sentiment_path)
    options_df, options_cols = load_options_features(options_path)

    for ticker in tickers:
        raw = _download_or_load_price_data(ticker, start, end)
        if raw.empty:
            raise ValueError(f"Yahoo Finance returned empty data for {ticker}.")

        asset_df = raw[["Open", "High", "Low", "Close", "Volume"]].copy()
        asset_df["VIX"] = _vix_or_realized_proxy(vix, raw, ticker)
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
    if not sentiment_df.empty:
        df = df.join(sentiment_df, how="left").ffill()
        feature_cols.extend(sentiment_cols)
    if not options_df.empty:
        df = _join_external_features(df, options_df, options_cols)
        feature_cols.extend(options_cols)
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
