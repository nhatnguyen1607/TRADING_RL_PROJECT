import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


EXTERNAL_DIR = Path("data/external")


def _clean_numeric(series):
    return pd.to_numeric(series, errors="coerce")


def _zscore(series, window=20):
    return (series - series.rolling(window).mean()) / (series.rolling(window).std() + 1e-8)


def _safe_ratio(numerator, denominator):
    return numerator / denominator.replace(0, np.nan)


def _stationary_features(df, prefix):
    out = pd.DataFrame(index=df.index)
    put = df[f"{prefix}_PutVolume"]
    call = df[f"{prefix}_CallVolume"]
    total = put.fillna(0.0) + call.fillna(0.0)
    out[f"Options_{prefix}_PutCallRatio"] = _safe_ratio(put, call)
    out[f"Options_{prefix}_TotalVolume_Z20"] = _zscore(np.log1p(total))
    out[f"Options_{prefix}_PutCallRatio_Z20"] = _zscore(out[f"Options_{prefix}_PutCallRatio"])
    out[f"Options_{prefix}_PutCallRatio_Momentum20"] = out[f"Options_{prefix}_PutCallRatio"].pct_change(
        20,
        fill_method=None,
    )
    return out


def build_spy_features(path, start, end, chunksize=500_000):
    header = pd.read_csv(path, nrows=0)
    rename = {col: col.strip("[]") for col in header.columns}
    needed_names = {
        "QUOTE_DATE",
        "C_VOLUME",
        "P_VOLUME",
        "C_IV",
        "P_IV",
        "DTE",
        "STRIKE_DISTANCE_PCT",
    }
    usecols = [col for col in header.columns if rename[col] in needed_names]
    daily_parts = []
    for chunk in pd.read_csv(path, usecols=usecols, chunksize=chunksize):
        chunk = chunk.rename(columns=rename)
        chunk["Date"] = pd.to_datetime(chunk["QUOTE_DATE"], errors="coerce")
        chunk = chunk[(chunk["Date"] >= start) & (chunk["Date"] <= end)]
        for col in ["C_VOLUME", "P_VOLUME", "C_IV", "P_IV", "DTE", "STRIKE_DISTANCE_PCT"]:
            if col in chunk.columns:
                chunk[col] = _clean_numeric(chunk[col])
        atm = chunk["STRIKE_DISTANCE_PCT"].abs() <= 0.05
        grouped = chunk.groupby("Date").agg(
            SPY_PutVolume=("P_VOLUME", "sum"),
            SPY_CallVolume=("C_VOLUME", "sum"),
            SPY_PutIV_Mean=("P_IV", "mean"),
            SPY_CallIV_Mean=("C_IV", "mean"),
            SPY_ATM_PutVolume=("P_VOLUME", lambda x: x[atm.loc[x.index]].sum()),
            SPY_ATM_CallVolume=("C_VOLUME", lambda x: x[atm.loc[x.index]].sum()),
        )
        daily_parts.append(grouped)
    if not daily_parts:
        return pd.DataFrame()
    daily = pd.concat(daily_parts).groupby(level=0).sum(min_count=1)
    daily["SPY_IVSkew"] = daily["SPY_PutIV_Mean"] - daily["SPY_CallIV_Mean"]
    features = _stationary_features(daily, "SPY")
    features["Options_SPY_IVSkew"] = daily["SPY_IVSkew"]
    features["Options_SPY_IVSkew_Z20"] = _zscore(daily["SPY_IVSkew"])
    features["Options_SPY_ATM_PutCallRatio"] = _safe_ratio(daily["SPY_ATM_PutVolume"], daily["SPY_ATM_CallVolume"])
    features["Options_SPY_ATM_PutCallRatio_Z20"] = _zscore(features["Options_SPY_ATM_PutCallRatio"])
    return features


def build_spy_price_cache(path, start, end, output, chunksize=500_000):
    header = pd.read_csv(path, nrows=0)
    rename = {col: col.strip("[]") for col in header.columns}
    needed_names = {"QUOTE_DATE", "UNDERLYING_LAST", "C_VOLUME", "P_VOLUME"}
    usecols = [col for col in header.columns if rename[col] in needed_names]
    daily_parts = []
    for chunk in pd.read_csv(path, usecols=usecols, chunksize=chunksize):
        chunk = chunk.rename(columns=rename)
        chunk["Date"] = pd.to_datetime(chunk["QUOTE_DATE"], errors="coerce")
        chunk = chunk[(chunk["Date"] >= start) & (chunk["Date"] <= end)]
        if chunk.empty:
            continue
        for col in ["UNDERLYING_LAST", "C_VOLUME", "P_VOLUME"]:
            if col in chunk.columns:
                chunk[col] = _clean_numeric(chunk[col])
        grouped = chunk.groupby("Date").agg(
            Close=("UNDERLYING_LAST", "mean"),
            OptionsCallVolume=("C_VOLUME", "sum"),
            OptionsPutVolume=("P_VOLUME", "sum"),
        )
        daily_parts.append(grouped)

    if not daily_parts:
        return None

    daily = pd.concat(daily_parts).groupby(level=0).agg(
        Close=("Close", "mean"),
        OptionsCallVolume=("OptionsCallVolume", "sum"),
        OptionsPutVolume=("OptionsPutVolume", "sum"),
    )
    daily["Open"] = daily["Close"].shift(1).fillna(daily["Close"])
    daily["High"] = daily[["Open", "Close"]].max(axis=1)
    daily["Low"] = daily[["Open", "Close"]].min(axis=1)
    daily["Volume"] = daily["OptionsCallVolume"].fillna(0.0) + daily["OptionsPutVolume"].fillna(0.0)
    daily = daily[["Open", "High", "Low", "Close", "Volume"]].reset_index()
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    daily.to_csv(output, index=False)
    return output


def _read_bond_etf(path):
    df = pd.read_csv(path, sep=";", decimal=",", na_values=["#N/A N/A", "#N/A", "N/A"])
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
    df = df.dropna(subset=["Date"]).sort_values("Date")
    return df


def build_bond_etf_features(path, ticker, start, end):
    df = _read_bond_etf(path)
    df = df[(df["Date"] >= start) & (df["Date"] <= end)].copy()
    df = df.set_index("Date")
    daily = pd.DataFrame(index=df.index)
    daily[f"{ticker}_PutVolume"] = _clean_numeric(df["VOLUME_TOTAL_PUT"])
    daily[f"{ticker}_CallVolume"] = _clean_numeric(df["VOLUME_TOTAL_CALL"])
    features = _stationary_features(daily, ticker)
    features[f"Options_{ticker}_ShortInterestRatio"] = _clean_numeric(df["SHORT_INT_RATIO"])
    features[f"Options_{ticker}_ShortInterestRatio_Z20"] = _zscore(features[f"Options_{ticker}_ShortInterestRatio"])
    if "AVERAGE_BID_ASK_SPREAD_%" in df.columns:
        features[f"Options_{ticker}_BidAskSpread"] = _clean_numeric(df["AVERAGE_BID_ASK_SPREAD_%"])
        features[f"Options_{ticker}_BidAskSpread_Z20"] = _zscore(features[f"Options_{ticker}_BidAskSpread"])
    return features


def build_options_features(start_date="2015-01-01", end_date="2023-01-01", output=None):
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    frames = []
    spy_path = EXTERNAL_DIR / "spy_eod_total.csv"
    if spy_path.exists():
        frames.append(build_spy_features(spy_path, start, end))
        build_spy_price_cache(spy_path, start, end, EXTERNAL_DIR / "SPY_price_from_options.csv")

    for ticker in ["TLT", "HYG", "LQD", "EMB"]:
        path = EXTERNAL_DIR / f"{ticker}_data.csv"
        if path.exists():
            frames.append(build_bond_etf_features(path, ticker, start, end))

    if not frames:
        raise FileNotFoundError("No supported options source files found in data/external.")

    features = pd.concat(frames, axis=1).sort_index()
    features = features.loc[(features.index >= start) & (features.index <= end)]

    if {"Options_HYG_PutCallRatio_Z20", "Options_LQD_PutCallRatio_Z20"}.issubset(features.columns):
        features["Options_CreditStress_PutCall_Z20"] = features[
            ["Options_HYG_PutCallRatio_Z20", "Options_LQD_PutCallRatio_Z20"]
        ].mean(axis=1)
    if {"Options_TLT_PutCallRatio_Z20", "Options_TLT_ShortInterestRatio_Z20"}.issubset(features.columns):
        features["Options_BondStress_Z20"] = features[
            ["Options_TLT_PutCallRatio_Z20", "Options_TLT_ShortInterestRatio_Z20"]
        ].mean(axis=1)

    features = features.replace([np.inf, -np.inf], np.nan).ffill()
    features = features.reset_index().rename(columns={"index": "Date"})
    output = Path(output or (EXTERNAL_DIR / "options_features_daily.csv"))
    output.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(output, index=False)
    return output, features


def main():
    parser = argparse.ArgumentParser(description="Build daily options-derived features for Trading RL.")
    parser.add_argument("--start-date", default="2015-01-01")
    parser.add_argument("--end-date", default="2023-01-01")
    parser.add_argument("--output", default=str(EXTERNAL_DIR / "options_features_daily.csv"))
    args = parser.parse_args()
    output, features = build_options_features(args.start_date, args.end_date, args.output)
    print(f"Wrote {output}")
    print(f"Rows: {len(features)} | Columns: {len(features.columns)}")
    print(f"Date range: {features['Date'].min()} -> {features['Date'].max()}")


if __name__ == "__main__":
    main()
