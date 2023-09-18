# GNN Practice with PyTorch Geometric

## Introduction

### Data: 
[Spotify 1.2M+ Songs](https://www.kaggle.com/datasets/rodolfofigueroa/spotify-12m-songs)

The raw file that was transformed into the 3 pkl files is too large to be uploaded to GitHub, so it's not included in this repository. However, this file is available in the Kaggle link above.

Task:
1. Build a bipartite heterogeneous graph that connect artists and songs before a pivot year (e.g. 2015).
2. Train a GNN model to predict which pairs of artists would collaborate in the future (after the pivot year).
* The model performs node predictions on "Artist pair" nodes - those are nodes that are chosen randomly from the Cartesian product of the set of artists before the pivot year.

## Notebooks order
1. `transform_data.ipynb`
2. `graph_construction.ipynb`
3. `learning_task.ipynb`

For the purpose of understanding the GNN model, it's enough to read the last two notebooks and skip the `transform_data.ipynb` notebook.
By opening and reviewing the 3 files at the `playground.ipynb` notebook - `artists.pkl`, `creations.pkl` and `songs.pkl`, which are the output of the first notebook, you can get a sense of the data structure. 