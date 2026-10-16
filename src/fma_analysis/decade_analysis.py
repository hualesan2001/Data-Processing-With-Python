from dataclasses import dataclass

import numpy as np
import pandas as pd
from kneed import KneeLocator
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from .feature_engineering import get_cross_features


@dataclass
class DecadeResult:
    """PCA + k-means output for a single decade (coordinates, cluster labels, variance ratios, elbow inertias)."""

    decade: int
    X_pca: np.ndarray
    genres: np.ndarray
    k_opt: int
    cluster_labels: np.ndarray
    cluster_centers: np.ndarray
    explained_variance_ratio: np.ndarray
    inertias: list[float]


class DecadeAnalyzer:
    """Bucket fma_small tracks by decade and run PCA + elbow-selected k-means on each."""

    def __init__(self, tracks_df: pd.DataFrame, features_df: pd.DataFrame, min_samples: int = 30):
        """Assemble the feature matrix (raw FMA features + cross-feature derivations) once."""
        self.tracks = tracks_df
        self.min_samples = min_samples
        # Create new cross features + original ones
        cross = get_cross_features(features_df)
        # make the new cross features have the original schema
        cross.columns = pd.MultiIndex.from_tuples([('cross', c, '01') for c in cross.columns])
        self.X_full = pd.concat([features_df, cross], axis=1).astype(float)

    def bucket_by_decade(self) -> dict[int, pd.Index]:
        """Return {decade -> track_ids} for fma_small tracks with valid release dates and enough samples."""
        # filter out track that do not have an album.date_released.
        small = self.tracks['set', 'subset'] == 'small'
        released = self.tracks['album', 'date_released']
        mask = small & released.notna()
        ###
        years = released[mask].dt.year
        decades = (years // 10 * 10).astype(int)
        # group by decade
        buckets = {}
        for decade, group in decades.groupby(decades):
            if len(group) >= self.min_samples:
                buckets[int(decade)] = group.index
        return dict(sorted(buckets.items()))

    def analyze_decade(self, decade: int, track_ids: pd.Index) -> DecadeResult:
        """Run StandardScaler + PCA(2D) + elbow-selected k-means for one decade's track set."""
        # filter out track with non-complete features
        X = self.X_full.loc[track_ids].dropna()
        ids = X.index

        genres = self.tracks.loc[ids, ('track', 'genre_top')].astype(str).to_numpy()
        # standardize every feature
        X_scaled = StandardScaler().fit_transform(X.to_numpy())
        # PCA analysis
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
        # Elbow detection
        ks = list(range(1, 11))
        inertias = [KMeans(n_clusters=k, n_init=10, random_state=0).fit(X_pca).inertia_ for k in ks]
        # Find the elbow, otherwise fallback to 3
        knee = KneeLocator(ks, inertias, curve='convex', direction='decreasing').knee
        k_opt = int(knee) if knee is not None else 3

        km = KMeans(n_clusters=k_opt, n_init=10, random_state=0).fit(X_pca)
        return DecadeResult(
            decade=decade,
            X_pca=X_pca,
            genres=genres,
            k_opt=k_opt,
            cluster_labels=km.labels_,
            cluster_centers=km.cluster_centers_,
            explained_variance_ratio=pca.explained_variance_ratio_,
            inertias=inertias,
        )

    def analyze_all(self) -> dict[int, DecadeResult]:
        """Run :meth:`analyze_decade` for every decade bucket and return {decade -> DecadeResult}."""
        # Run the analysis
        buckets = self.bucket_by_decade()
        return {d: self.analyze_decade(d, ids) for d, ids in tqdm(buckets.items(), desc='Decades')}
