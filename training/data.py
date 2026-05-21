import random

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def split_with_attrs(df, split_idx):
    left = df.iloc[:split_idx].copy()
    right = df.iloc[split_idx:].copy()
    copy_attrs(df, left)
    copy_attrs(df, right)
    return left, right


def slice_with_attrs(df, start_idx, end_idx):
    sliced = df.iloc[start_idx:end_idx].copy()
    copy_attrs(df, sliced)
    return sliced


def copy_attrs(source, target):
    for key in ("feature_cols", "asset_cols", "tickers"):
        if key in source.attrs:
            target.attrs[key] = source.attrs[key]


def train_test_scale(df, split_idx):
    feature_cols = df.attrs["feature_cols"]
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()

    scaler = StandardScaler()
    train_df.loc[:, feature_cols] = scaler.fit_transform(train_df[feature_cols])
    test_df.loc[:, feature_cols] = scaler.transform(test_df[feature_cols])
    copy_attrs(df, train_df)
    copy_attrs(df, test_df)
    return train_df, test_df, scaler
