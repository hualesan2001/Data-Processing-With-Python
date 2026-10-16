import numpy as np
import pandas as pd


def get_cross_features(features_df: pd.DataFrame) -> pd.DataFrame:
    """Derive cross-feature ratios (brightness, dynamics, noisiness, tonal complexity, ...) from FMA's per-track audio features."""
    mfcc_std = features_df['mfcc', 'std']
    low_mfcc = mfcc_std.loc[:, '01':'09'].mean(axis=1)
    high_mfcc = mfcc_std.loc[:, '10':'20'].mean(axis=1)

    return pd.DataFrame({
        'brightness_ratio': features_df['spectral_centroid', 'mean', '01'] / features_df['spectral_bandwidth', 'mean', '01'],
        'mfcc_high_low_var_ratio': high_mfcc / low_mfcc,
        'chroma_stability': features_df['chroma_cens', 'std'].mean(axis=1),
        'dynamics': features_df['rmse', 'std', '01'] / features_df['rmse', 'mean', '01'],
        'noisiness': features_df['zcr', 'mean', '01'] * features_df['spectral_rolloff', 'mean', '01'],
        'tonal_complexity': features_df['tonnetz', 'std'].mean(axis=1),
    }, index=features_df.index)
