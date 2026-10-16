import os
from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import ConvexHull


class Visualizer:
    """Matplotlib plot helpers for the PCA pipelines (decade grids, elbow curves, genre-pair distances)."""

    @staticmethod
    def plot_pca_results(X_pca, y_encoded, genre_names, explained_variance_ratio, title='PCA Analysis of FMA Features', figsize=(12, 6)):
        """Single 2D PCA scatter colored by encoded genre, with a legend mapping back to genre names."""
        plt.figure(figsize=figsize)
        scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1],
                             c=y_encoded, cmap='RdYlBu', alpha=0.6, edgecolors='w')

        plt.xlabel(f'Principal Component 1 ({explained_variance_ratio[0]:.2%} variance)')
        plt.ylabel(f'Principal Component 2 ({explained_variance_ratio[1]:.2%} variance)')
        plt.title(title)

        handles, labels = scatter.legend_elements()
        plt.legend(handles, genre_names, title="Genres", loc='best')

        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.show()

    @staticmethod
    def _grid_shape(n: int) -> tuple[int, int]:
        """Pick a (rows, cols) subplot grid sized to fit ``n`` panels, capped at 3 columns wide."""
        cols = min(3, n)
        rows = (n + cols - 1) // cols
        return rows, cols

    @staticmethod
    def _save(fig, save_path: str) -> None:
        """Save ``fig`` as both PNG (300 dpi) and SVG alongside ``save_path``."""
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        base, _ = os.path.splitext(save_path)
        fig.savefig(f'{base}.png', dpi=300, bbox_inches='tight')
        fig.savefig(f'{base}.svg', bbox_inches='tight')

    @staticmethod
    def plot_decade_grid(results: dict, save_path: str | None = None):
        """Per-decade PCA scatter grid with convex-hull cluster overlays, colored by top genre."""
        items = sorted(results.items())
        all_genres = sorted({g for _, r in items for g in r.genres})
        cmap = plt.get_cmap('tab10')
        genre_colors = {g: cmap(i % 10) for i, g in enumerate(all_genres)}

        rows, cols = Visualizer._grid_shape(len(items))
        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows), squeeze=False)

        for ax, (decade, r) in zip(axes.flat, items):
            for genre in all_genres:
                mask = r.genres == genre
                if mask.any():
                    ax.scatter(r.X_pca[mask, 0], r.X_pca[mask, 1],
                               color=genre_colors[genre], alpha=0.65, s=22,
                               edgecolors='white', linewidths=0.3)
            for c in range(r.k_opt):
                pts = r.X_pca[r.cluster_labels == c]
                if len(pts) >= 3:
                    hull = ConvexHull(pts)
                    poly = pts[hull.vertices]
                    ax.fill(poly[:, 0], poly[:, 1], color='gray', alpha=0.12)
                    ax.plot(np.append(poly[:, 0], poly[0, 0]),
                            np.append(poly[:, 1], poly[0, 1]),
                            color='black', linewidth=0.8, alpha=0.6)
            ax.scatter(r.cluster_centers[:, 0], r.cluster_centers[:, 1],
                       marker='x', s=90, c='black', linewidths=2)
            v1, v2 = r.explained_variance_ratio
            ax.set_title(f'{decade}s — n={len(r.X_pca)}, k={r.k_opt} '
                         f'(PC1 {v1:.1%}, PC2 {v2:.1%})')
            ax.set_xlabel('PC1')
            ax.set_ylabel('PC2')
            ax.grid(True, linestyle='--', alpha=0.3)

        for ax in axes.flat[len(items):]:
            ax.set_visible(False)

        handles = [plt.Line2D([0], [0], marker='o', linestyle='', markersize=9,
                              markerfacecolor=genre_colors[g], markeredgecolor='white',
                              label=g) for g in all_genres]
        fig.legend(handles=handles, loc='lower center',
                   ncol=min(len(all_genres), 8), bbox_to_anchor=(0.5, -0.02))
        fig.suptitle('PCA + K-means clusters by decade')
        fig.tight_layout(rect=(0, 0.04, 1, 1))

        if save_path:
            Visualizer._save(fig, save_path)
        return fig

    @staticmethod
    def plot_elbow_grid(results: dict, save_path: str | None = None):
        """Per-decade k-means inertia curves with the elbow-selected k highlighted."""
        items = sorted(results.items())
        rows, cols = Visualizer._grid_shape(len(items))
        fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows), squeeze=False)

        for ax, (decade, r) in zip(axes.flat, items):
            ks = list(range(1, len(r.inertias) + 1))
            ax.plot(ks, r.inertias, marker='o')
            ax.axvline(r.k_opt, color='red', linestyle='--', alpha=0.7,
                       label=f'k={r.k_opt}')
            ax.set_title(f'{decade}s')
            ax.set_xlabel('k')
            ax.set_ylabel('inertia')
            ax.legend(loc='upper right')
            ax.grid(True, linestyle='--', alpha=0.3)

        for ax in axes.flat[len(items):]:
            ax.set_visible(False)

        fig.suptitle('Elbow curves per decade')
        fig.tight_layout()

        if save_path:
            Visualizer._save(fig, save_path)
        return fig

    @staticmethod
    def plot_pair_distance_trends(result, save_path: str | None = None):
        """Small-multiples line plot: per anchor genre, distance from that anchor to every other genre over time."""
        genres = result.genres
        n = len(genres)
        cmap = plt.get_cmap('tab10')
        genre_colors = {g: cmap(i % 10) for i, g in enumerate(genres)}

        cols = 4
        rows = (n + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(4.5 * cols, 3.2 * rows),
                                 squeeze=False, sharex=True, sharey=True)
        period_mid = [p + 2.5 for p in result.period_starts]

        for ax, anchor in zip(axes.flat, genres):
            for other in genres:
                if other == anchor:
                    continue
                a, b = sorted([anchor, other])
                col = f'{a}|{b}'
                y = result.pair_distances[col].to_numpy(dtype=float)
                ax.plot(period_mid, y, marker='o', color=genre_colors[other],
                        linewidth=1.5)
            ax.set_title(f'distances from "{anchor}"')
            ax.set_xlabel('period midpoint')
            ax.set_ylabel('centroid distance (PCA-2D)')
            ax.grid(True, linestyle='--', alpha=0.3)

        for ax in axes.flat[n:]:
            ax.set_visible(False)

        handles = [plt.Line2D([0], [0], marker='o', color=genre_colors[g],
                              label=g, linewidth=1.5) for g in genres]
        n_legend_rows = (len(genres) + 7) // 8
        bottom_reserve = 0.04 + 0.025 * n_legend_rows
        fig.legend(handles=handles, loc='lower center',
                   ncol=min(len(genres), 8),
                   bbox_to_anchor=(0.5, 0.0),
                   bbox_transform=fig.transFigure,
                   frameon=False)

        v1, v2 = result.explained_variance_ratio
        fig.suptitle(f'Genre-pair centroid distances over time '
                     f'(PC1 {v1:.1%}, PC2 {v2:.1%})')
        fig.tight_layout(rect=(0, bottom_reserve, 1, 0.97))

        if save_path:
            Visualizer._save(fig, save_path)
        return fig

    @staticmethod
    def plot_pair_time_strip(result, save_path: str | None = None,
                             pairs: list[tuple[str, str]] | None = None):
        """Per-pair time strip: rows = genre pairs, columns = 5y periods, cells show both genres' points + centroids in shared PCA-2D."""
        if pairs is None:
            pairs = list(combinations(result.genres, 2))
        if not pairs:
            return None
        period_starts = result.period_starts
        n_rows = len(pairs)
        n_cols = len(period_starts)

        cmap = plt.get_cmap('tab10')
        genre_colors = {g: cmap(i % 10) for i, g in enumerate(result.genres)}

        x_min, x_max = result.X_pca[:, 0].min(), result.X_pca[:, 0].max()
        y_min, y_max = result.X_pca[:, 1].min(), result.X_pca[:, 1].max()
        pad_x = (x_max - x_min) * 0.05
        pad_y = (y_max - y_min) * 0.05

        fig, axes = plt.subplots(n_rows, n_cols,
                                 figsize=(1.9 * n_cols, 1.6 * n_rows),
                                 squeeze=False)

        for row, (a, b) in enumerate(pairs):
            for col, p in enumerate(period_starts):
                ax = axes[row, col]
                ax.set_xlim(x_min - pad_x, x_max + pad_x)
                ax.set_ylim(y_min - pad_y, y_max + pad_y)
                ax.set_xticks([])
                ax.set_yticks([])
                for genre in (a, b):
                    mask = (result.track_periods == p) & (result.track_genres == genre)
                    if mask.any():
                        ax.scatter(result.X_pca[mask, 0], result.X_pca[mask, 1],
                                   s=4, color=genre_colors[genre], alpha=0.45,
                                   linewidths=0)
                    c = result.centroids.get((p, genre))
                    if c is not None:
                        ax.scatter([c[0]], [c[1]], marker='x', s=70,
                                   c=[genre_colors[genre]], linewidths=2.5)
                if row == 0:
                    ax.set_title(f'{p}–{p + 5}', fontsize=9)
                if col == 0:
                    ax.set_ylabel(f'{a}\nvs\n{b}', fontsize=7, rotation=0,
                                  ha='right', va='center')

        fig.suptitle('Per-pair time strip — points + centroids by 5y period', y=1.0)
        fig.tight_layout(rect=(0, 0, 1, 0.98))

        if save_path:
            Visualizer._save(fig, save_path)
        return fig
