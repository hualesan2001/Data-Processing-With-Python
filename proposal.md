## Project Proposal

## dataset

We will be using the FMA (Free Music Archive) dataset. It contains ~106,000 tracks (we might be only using a subset) with attached metadata (year of release, genre labels, artist information etc.) and extracted features (chrome information, spectographs etc.).

## task

We shall closely inspect the dataset and try to find patterns in the data.

We can include simple bar plots showing prevalence of each genre through time.
We can also try to use clustering (k-means) to group tracks by genre and try to mine interesting characteristics (similarity between genres, genre overlap, drift of artists over time between genres, etc.)

If time permits we can try to use dimensionality reduction (PCA, t-SNE) to visualize the data.

## tools

We will use python with the following libraries:
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- librosa