import unittest

import numpy as np
import pandas as pd

from fma_analysis.genre_pair_analysis import GenrePairResult, interesting_pairs


def make_result(distances: dict[str, list[float]], periods: list[int]) -> GenrePairResult:
    """Build a minimal GenrePairResult sufficient for testing interesting_pairs."""
    df = pd.DataFrame(distances, index=periods, dtype=float)
    genres = sorted({g for col in distances for g in col.split('|')})
    return GenrePairResult(
        period_starts=periods,
        genres=genres,
        X_pca=np.empty((0, 2)),
        track_periods=np.empty(0, dtype=int),
        track_genres=np.empty(0, dtype=object),
        centroids={},
        pair_distances=df,
        explained_variance_ratio=np.array([0.0, 0.0]),
    )


class TestInterestingPairs(unittest.TestCase):

    def test_includes_pair_above_threshold(self):
        # 1.0 -> 2.0 is a 100% change, well above 0.3.
        result = make_result({'Pop|Rock': [1.0, 2.0]}, periods=[2000, 2010])
        self.assertEqual(interesting_pairs(result, threshold=0.3), [('Pop', 'Rock')])

    def test_excludes_pair_below_threshold(self):
        # 1.0 -> 1.1 is only a 10% change.
        result = make_result({'Pop|Rock': [1.0, 1.1]}, periods=[2000, 2010])
        self.assertEqual(interesting_pairs(result, threshold=0.3), [])

    def test_excludes_pair_with_single_observation(self):
        result = make_result({'Pop|Rock': [1.0, np.nan]}, periods=[2000, 2010])
        self.assertEqual(interesting_pairs(result, threshold=0.3), [])

    def test_excludes_pair_starting_at_zero(self):
        # Division-by-zero guard: first value 0 -> skip.
        result = make_result({'Pop|Rock': [0.0, 5.0]}, periods=[2000, 2010])
        self.assertEqual(interesting_pairs(result, threshold=0.3), [])

    def test_includes_decreasing_pair(self):
        # 5.0 -> 1.0 is -80% change; absolute relative change qualifies.
        result = make_result({'Pop|Rock': [5.0, 1.0]}, periods=[2000, 2010])
        self.assertEqual(interesting_pairs(result, threshold=0.3), [('Pop', 'Rock')])


if __name__ == '__main__':
    unittest.main()
