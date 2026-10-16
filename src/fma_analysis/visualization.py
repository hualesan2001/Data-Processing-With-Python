import os
import logging
import warnings
import re

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from wordcloud import WordCloud

from .dataset import FMADataset

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)

logger = logging.getLogger(__name__)


class Visualizer:
    """Title- and genre-level exploration plots (genre trend, vocabulary, duplicate rate, overlap heatmap)."""

    def __init__(self):
        """Trigger FMA download/unzip if needed and prepare lazy loaders for tracks/features."""
        FMADataset()
        self.data_dir = FMADataset.data_dir
        self._tracks = None
        self._features = None

    def load_data(self):
        """Lazy-load and cache tracks.csv + features.csv on first call."""
        if self._tracks is not None and self._features is not None:
            return self._tracks, self._features

        tracks_path = os.path.join(
            self.data_dir,
            "fma_metadata",
            "tracks.csv"
        )

        features_path = os.path.join(
            self.data_dir,
            "fma_metadata",
            "features.csv"
        )

        logger.info("Loading CSV files...")

        self._tracks = pd.read_csv(
            tracks_path,
            header=[0, 1],
            index_col=0
        )

        self._features = pd.read_csv(
            features_path,
            header=[0, 1, 2],
            index_col=0
        )

        logger.info("Data loaded successfully.")
        return self._tracks, self._features
    def plot_genre_trend(self, top_n: int = 8):
        """Graph 1 — Genre Trend."""
        tracks, _ = self.load_data()

        small = tracks[
            tracks[('set', 'subset')] == 'small'
        ].copy()

        df = small[
            [('track', 'genre_top'),
             ('track', 'date_created')]
        ].copy()

        df.columns = ['genre', 'date_created']

        df = df.dropna()

        df['year'] = pd.to_datetime(
            df['date_created'],
            errors='coerce'
        ).dt.year

        df = df.dropna(subset=['year'])

        df['year'] = df['year'].astype(int)

        top_genres = (
            df['genre']
            .value_counts()
            .head(top_n)
            .index
        )

        df = df[df['genre'].isin(top_genres)]

        genre_year = (
            df.groupby(['year', 'genre'])
            .size()
            .reset_index(name='count')
        )

        #Decoration
        palette = sns.color_palette("tab10", n_colors=top_n)

        plt.figure(figsize=(14, 7))

        ax = plt.gca()

        ax.set_facecolor('#0d1117')
        plt.gcf().set_facecolor('#0d1117')

        for i, genre in enumerate(top_genres):

            gd = genre_year[
                genre_year['genre'] == genre
            ]

            ax.plot(
                gd['year'],
                gd['count'],
                marker='o',
                linewidth=2.5,
                markersize=6,
                label=genre,
                color=palette[i]
            )

        ax.set_title(
            "Genre Trends Over Time (FMA Small)",
            fontsize=15,
            color='white',
            pad=15,
            fontweight='bold'
        )

        ax.set_xlabel("Year", color='#aaaaaa')
        ax.set_ylabel("Number of Tracks", color='#aaaaaa')

        ax.tick_params(colors='#aaaaaa')

        for spine in ax.spines.values():
            spine.set_color('#333333')

        ax.grid(
            True,
            linestyle='--',
            alpha=0.3,
            color='#444444'
        )

        ax.legend(
            loc='upper left',
            fontsize=9,
            framealpha=0.3,
            facecolor='#1a1a2e',
            labelcolor='white'
        )

        plt.tight_layout()

    def plot_vocabulary_by_decade(self):
        """Graph 2 — Vocabulary Evolution Across Decades.

        Research Question 1: How has the vocabulary used in song titles changed across decades?
        """
        tracks, _ = self.load_data()

        small = tracks[
            tracks[('set', 'subset')] == 'small'
        ].copy()

        df = pd.DataFrame({
            'title': small[('track', 'title')],
            'date': small[('track', 'date_created')]
        })

        df = df.dropna()

        df['year'] = pd.to_datetime(
            df['date'],
            errors='coerce'
        ).dt.year

        df = df.dropna(subset=['year'])

        df['year'] = df['year'].astype(int)

        df['decade'] = (df['year'] // 10) * 10

        top_decades = (
            df['decade']
            .value_counts()
            .sort_index()
            .tail(2)
            .index
            .tolist()
        )

        n = len(top_decades)
        ncols = 2
        nrows = (n + 1) // ncols

        stopwords = {
            'the', 'a', 'an', 'of', 'in', 'on',
            'and', 'for', 'to', 'is', 'it',
            'at', 'by', 'ft', 'feat'
        }

        #Start subplots
        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(16, 5 * nrows),
            facecolor='#0d1117'
        )

        axes = np.array(axes).flatten()

        for i, decade in enumerate(top_decades):

            subset = df[df['decade'] == decade]
            track_count = len(subset)

            text = " ".join(
                subset['title']
                .astype(str)
                .tolist()
            )

            ax = axes[i]

            try:

                wc = WordCloud(
                    width=600,
                    height=400,
                    background_color='#0d1117',
                    colormap='plasma',
                    stopwords=stopwords,
                    max_words=80
                ).generate(text)

                ax.imshow(wc, interpolation='bilinear')

            except Exception:

                ax.text(
                    0.5, 0.5,
                    "WordCloud unavailable",
                    ha='center',
                    va='center',
                    color='white'
                )

            ax.set_title(
                f"{decade}s  —  {track_count:,} tracks",
                color='white',
                fontsize=13,
                fontweight='bold'
            )

            ax.text(
                0.5, -0.04,
                "Word size = frequency in song titles",
                transform=ax.transAxes,
                ha='center',
                fontsize=8,
                color='#aaaaaa',
                style='italic'
            )

            ax.axis('off')

        #Visible 2 decades
        for j in range(n, len(axes)):
            axes[j].set_visible(False)

        plt.suptitle(
            "Vocabulary Evolution Across Decades",
            fontsize=18,
            color='white',
            fontweight='bold'
        )

        #Global legend box explaining the chart
        fig.text(
            0.5, -0.01,
            "Each panel shows the most frequent words in song titles for that decade. Larger words appear more often. Common stopwords (the, a, of…) are excluded.\n",
            ha='center',
            fontsize=9,
            color='#cccccc',
            style='italic'
        )

        plt.tight_layout(rect=[0, 0.03, 1, 0.96])


    def plot_title_duplicate_rate(self):
        """Graph 3 — Title Duplication Over Time.

        Research Question 2: How frequently do song titles overlap?
        """
        tracks, _ = self.load_data()

        small = tracks[
            tracks[('set', 'subset')] == 'small'
        ].copy()

        df = pd.DataFrame({
            'title': small[('track', 'title')],
            'date': small[('track', 'date_created')]
        })

        df = df.dropna()

        df['year'] = pd.to_datetime(
            df['date'],
            errors='coerce'
        ).dt.year

        df = df.dropna(subset=['year'])

        df['year'] = df['year'].astype(int)

        df['decade'] = (df['year'] // 10) * 10

        duplicate_rates = []

        decades = sorted(df['decade'].unique())

        for decade in decades:

            sub = df[df['decade'] == decade]

            total_titles = len(sub)

            repeated_titles = (
                sub['title']
                .duplicated()
                .sum()
            )

            rate = repeated_titles / total_titles

            duplicate_rates.append({
                'decade': decade,
                'duplicate_rate': rate
            })

        dup_df = pd.DataFrame(duplicate_rates)

        plt.figure(
            figsize=(12, 6),
            facecolor='#0d1117'
        )

        ax = plt.gca()

        ax.set_facecolor('#0d1117')

        sns.lineplot(
            data=dup_df,
            x='decade',
            y='duplicate_rate',
            marker='o',
            linewidth=3,
            color='#ff4d6d'
        )

        #Annotate each data point with its value
        for _, row in dup_df.iterrows():
            ax.annotate(
                f"{row['duplicate_rate']:.1%}",
                xy=(row['decade'], row['duplicate_rate']),
                xytext=(0, 10),
                textcoords='offset points',
                ha='center',
                fontsize=8,
                color='#ff4d6d'
            )

        # Mark the peak decade
        peak_row = dup_df.loc[dup_df['duplicate_rate'].idxmax()]
        ax.annotate(
            f"Peak: {int(peak_row['decade'])}s",
            xy=(peak_row['decade'], peak_row['duplicate_rate']),
            xytext=(20, -20),
            textcoords='offset points',
            fontsize=9,
            color='white',
            arrowprops=dict(arrowstyle='->', color='white', lw=1.2)
        )

        plt.title(
            "Song Title Duplication Rate Across Decades",
            fontsize=16,
            color='white',
            fontweight='bold'
        )

        plt.xlabel(
            "Decade  (e.g. 2000 = 2000–2009)",
            color='#aaaaaa',
            fontsize=10
        )
        plt.ylabel(
            "Duplicate Rate\n(% of titles repeated within the same decade)",
            color='#aaaaaa',
            fontsize=10
        )

        #Format y-axis as percentage
        ax.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda y, _: f"{y:.0%}")
        )

        plt.xticks(
            dup_df['decade'],
            [f"{int(d)}s" for d in dup_df['decade']],
            color='white'
        )
        plt.yticks(color='white')

        ax.grid(
            linestyle='--',
            alpha=0.3
        )

        #Explanation box
        ax.legend(
            handles=[
                plt.Line2D([0], [0], color='#ff4d6d',
                           linewidth=2.5, marker='o', markersize=6,
                           label='Duplicate rate per decade')
            ],
            loc='upper left',
            fontsize=9,
            framealpha=0.3,
            facecolor='#1a1a2e',
            labelcolor='white'
        )

        plt.tight_layout()



    def plot_genre_overlap_heatmap(self, top_n=6):
        """Graph 4 — Genre Vocabulary Overlap Heatmap.

        Research Question 3: How much overlap exists between genres?
        """
        tracks, _ = self.load_data()

        small = tracks[
            tracks[('set', 'subset')] == 'small'
        ].copy()

        df = pd.DataFrame({
            'genre': small[('track', 'genre_top')],
            'title': small[('track', 'title')]
        })

        df = df.dropna()

        top_genres = (
            df['genre']
            .value_counts()
            .head(top_n)
            .index
        )

        df = df[df['genre'].isin(top_genres)]

        stopwords = {
            'the', 'a', 'an', 'of', 'in', 'on',
            'and', 'for', 'to', 'is', 'it',
            'at', 'by', 'ft', 'feat'
        }

        #Preparing data
        genre_vocab = {}

        for genre in top_genres:

            titles = df[
                df['genre'] == genre
            ]['title']

            text = " ".join(
                titles.astype(str)
            ).lower()

            words = re.findall(r'\b[a-z]+\b', text)

            words = [
                w for w in words
                if w not in stopwords
            ]

            genre_vocab[genre] = set(words)

        matrix = pd.DataFrame(
            index=top_genres,
            columns=top_genres
        )

        #Jaccard similarity
        for g1 in top_genres:

            for g2 in top_genres:

                set1 = genre_vocab[g1]
                set2 = genre_vocab[g2]

                intersection = len(set1 & set2)
                union = len(set1 | set2)

                similarity = intersection / union

                matrix.loc[g1, g2] = similarity

        matrix = matrix.astype(float)

        #Decoration
        plt.figure(
            figsize=(10, 8),
            facecolor='#0d1117'
        )

        ax = plt.gca()
        ax.set_facecolor('#0d1117')
        plt.gcf().set_facecolor('#0d1117')

        sns.heatmap(
            matrix,
            annot=True,
            fmt='.2f',
            cmap='magma',
            linewidths=0.5,
            linecolor='#333333',
            vmin=0,
            vmax=1,
            ax=ax,
            cbar_kws={
                'label': 'Jaccard Similarity\n(0 = no overlap, 1 = identical vocabulary)',
                'shrink': 0.8
            }
        )

        #Style colorbar
        cbar = ax.collections[0].colorbar
        cbar.ax.yaxis.label.set_color('white')
        cbar.ax.yaxis.label.set_fontsize(9)
        cbar.ax.tick_params(colors='white')

        #Style axis labels
        ax.set_xlabel(
            "Genre",
            color='#aaaaaa',
            fontsize=11,
            labelpad=10
        )
        ax.set_ylabel(
            "Genre",
            color='#aaaaaa',
            fontsize=11,
            labelpad=10
        )
        ax.tick_params(colors='white', labelsize=10)

        plt.title(
            "Genre Vocabulary Overlap (Jaccard Similarity)\n"
            "How much do song title words overlap between genres?",
            fontsize=14,
            fontweight='bold',
            color='white',
            pad=15
        )

        #Note
        plt.figtext(
            0.5, -0.02,
            "Jaccard Similarity = shared words ÷ total unique words between two genres. "
            "Diagonal = 1.0 (each genre compared to itself).",
            ha='center',
            fontsize=8,
            color='#aaaaaa',
            style='italic'
        )

        plt.tight_layout()

    def create_visualizations(self, output_dir: str = "output", show: bool = True):
        """Render all four title/genre plots and save each as a PNG under ``output_dir``."""
        os.makedirs(output_dir, exist_ok=True)
        logger.info("Preparing visualizations...")
        plots = [
            ("genre_trend", lambda: self.plot_genre_trend(top_n=8)),
            ("vocabulary_by_decade", self.plot_vocabulary_by_decade),
            ("title_duplicate_rate", self.plot_title_duplicate_rate),
            ("genre_overlap_heatmap", lambda: self.plot_genre_overlap_heatmap(top_n=6)),
        ]
        for name, fn in plots:
            fn()
            path = os.path.join(output_dir, f"{name}.png")
            plt.gcf().savefig(path, dpi=300, bbox_inches='tight', facecolor=plt.gcf().get_facecolor())
            logger.info("Saved %s", path)
        if show:
            logger.info("Opening visualization windows...")
            plt.show()
            plt.close()
