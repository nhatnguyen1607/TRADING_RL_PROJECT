import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def calculate_technical_indicators(df):
    # 1. Các đường trung bình
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
    df['Dist_SMA50'] = (df['Close'] / df['SMA_50']) - 1
    
    df['Open_rel'] = (df['Open'] / df['Close']) - 1
    df['High_rel'] = (df['High'] / df['Close']) - 1
    df['Low_rel'] = (df['Low'] / df['Close']) - 1
    
    # Khối lượng tương đối so với trung bình 20 ngày
    vol_sma_20 = df['Volume'].rolling(window=20).mean()
    df['Vol_rel'] = (df['Volume'] / (vol_sma_20 + 1e-8)) - 1
    
    return df

def load_and_preprocess_data(ticker="SPY", start="2015-01-01", end="2023-01-01"):
    print(f"Đang tải dữ liệu {ticker} và ^VIX từ Yahoo Finance...")
    
    df = yf.download(ticker, start=start, end=end)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
        
    vix = yf.download("^VIX", start=start, end=end)
    if isinstance(vix.columns, pd.MultiIndex):
        vix.columns = vix.columns.get_level_values(0)
        
    df['VIX'] = vix['Close']
    df = calculate_technical_indicators(df)
    df.dropna(inplace=True)
    
    # 🚀 CHỈ ĐƯA CÁC ĐẶC TRƯNG TĨNH (STATIONARY) VÀO CHO AI HỌC
    # Loại bỏ hoàn toàn giá gốc (Open, High, Low, Close) khỏi mắt mô hình
    feature_cols = [
        'Return', 'Dist_SMA20', 'Dist_SMA50', 
        'Open_rel', 'High_rel', 'Low_rel', 'Vol_rel',
        'RSI', 'MACD', 'VIX'
    ]
    
    scaler = StandardScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols])
    
    # Lưu lại danh sách tên cột feature để Môi trường (Env) biết mà lấy
    df.attrs['feature_cols'] = feature_cols 
    
    print("Xử lý dữ liệu thành công! Kích thước State (Tĩnh):", df[feature_cols].shape)
    return df, scaler