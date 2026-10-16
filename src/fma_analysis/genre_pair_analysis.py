import logging
from dataclasses import dataclass
from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from .feature_engineering import get_cross_features

logger = logging.getLogger(__name__)

PERIODS: list[tuple[int, int]] = [(y, y + 5) for y in range(1980, 2015, 5)]


@dataclass
class GenrePairResult:
    """Output of :meth:`GenrePairAnalyzer.analyze`: PCA-2D points, per (period, genre) centroids, and pair distances."""

    period_starts: list[int]
    genres: list[str]
    X_pca: np.ndarray
    track_periods: np.ndarray
    track_genres: np.ndarray
    centroids: dict[tuple[int, str], np.ndarray | None]
    pair_distances: pd.DataFrame
    explained_variance_ratio: np.ndarray


class GenrePairAnalyzer:
    """Global PCA over an FMA subset, then centroid distances per (5y period, top-genre) pair."""

    def __init__(self, tracks_df: pd.DataFrame, features_df: pd.DataFrame,
                 min_samples: int = 10, subset: str = 'small'):
        """Assemble the feature matrix once; ``subset`` picks one of small / medium / large."""
        self.tracks = tracks_df
        self.features = features_df
        self.min_samples = min_samples
        self.subset = subset
        cross = get_cross_features(features_df)
        cross.columns = pd.MultiIndex.from_tuples([('cross', c, '01') for c in cross.columns])
        self.X_full = pd.concat([features_df, cross], axis=1).astype(float)

    def _select_track_ids(self) -> pd.Index:
        """Return track IDs in the chosen subset with valid release date + genre, falling inside the period range."""
        # `set.subset` is an ordered categorical (small < medium < large).
        # `<= 'medium'` includes small + medium; `<= 'large'` includes everything.
        in_subset = self.tracks['set', 'subset'] <= self.subset
        released = self.tracks['album', 'date_released']
        genre = self.tracks['track', 'genre_top']
        mask = in_subset & released.notna() & genre.notna()
        if not mask.any():
            return pd.Index([])
        years = released[mask].dt.year
        in_range = years.between(PERIODS[0][0], PERIODS[-1][1] - 1, inclusive='both')
        return years[in_range].index

    @staticmethod
    def _to_period_start(year: int) -> int:
        """Return the start year of the 5y bin that contains ``year`` (or -1 if outside the range)."""
        for start, end in PERIODS:
            if start <= year < end:
                return start
        return -1

    def analyze(self) -> GenrePairResult:
        """Run the full pipeline: global PCA, per (period, genre) centroids, pair distances."""
        ids = self._select_track_ids()
        if len(ids) == 0:
            raise RuntimeError("No tracks match fma_small + non-null genre + non-null release date.")

        X = self.X_full.loc[ids].dropna()
        ids = X.index

        years = self.tracks.loc[ids, ('album', 'date_released')].dt.year
        periods = years.map(self._to_period_start).to_numpy()
        genres = self.tracks.loc[ids, ('track', 'genre_top')].astype(str).to_numpy()

        keep = periods != -1
        X = X.iloc[keep]
        periods = periods[keep]
        genres = genres[keep]

        logger.info("Fitting global PCA on %d tracks (after period+feature filters)", len(X))
        X_scaled = StandardScaler().fit_transform(X.to_numpy())
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)

        unique_genres = sorted(set(genres))
        period_starts = [p for p, _ in PERIODS]
        logger.info("Computing centroids over %d periods x %d genres "
                    "(min_samples=%d per cell)",
                    len(period_starts), len(unique_genres), self.min_samples)

        centroids: dict[tuple[int, str], np.ndarray | None] = {}
        for p in period_starts:
            for g in unique_genres:
                mask = (periods == p) & (genres == g)
                if mask.sum() >= self.min_samples:
                    centroids[(p, g)] = X_pca[mask].mean(axis=0)
                else:
                    centroids[(p, g)] = None

        pair_cols = [f'{a}|{b}' for a, b in combinations(unique_genres, 2)]
        dist = pd.DataFrame(index=period_starts, columns=pair_cols, dtype=float)
        for a, b in combinations(unique_genres, 2):
            col = f'{a}|{b}'
            for p in period_starts:
                ca = centroids[(p, a)]
                cb = centroids[(p, b)]
                if ca is not None and cb is not None:
                    dist.loc[p, col] = float(np.linalg.norm(ca - cb))

        return GenrePairResult(
            period_starts=period_starts,
            genres=unique_genres,
            X_pca=X_pca,
            track_periods=periods,
            track_genres=genres,
            centroids=centroids,
            pair_distances=dist,
            explained_variance_ratio=pca.explained_variance_ratio_,
        )


def interesting_pairs(result: GenrePairResult, threshold: float) -> list[tuple[str, str]]:
    """Return pairs whose centroid distance changed by at least ``threshold`` (relative) between first and last observed period."""
    out = []
    for a, b in combinations(result.genres, 2):
        vals = result.pair_distances[f'{a}|{b}'].dropna()
        if len(vals) < 2 or vals.iloc[0] == 0:
            continue
        if abs(vals.iloc[-1] - vals.iloc[0]) / vals.iloc[0] >= threshold:
            out.append((a, b))
    return out
