
from sklearn.cluster import DBSCAN
import pandas as pd
import networkx as nx
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.neighbors import sort_graph_by_row_values
import itertools as it
import numpy as np
from scipy.spatial.distance import pdist,squareform
from scipy.sparse import csr_matrix
from tqdm import tqdm
import time
from sklearn.neighbors import NearestNeighbors
#utils for task 3


def run_dbscan(min_pts_values, eps_values, metric, clustering_data,
               precomputed_distances=None):
    """
    Execute DBSCAN on a combination of parameters and return the results in a DataFrame.
    """
    results = pd.DataFrame([])

    try:
        if precomputed_distances is not None:
            print("using precomputed distances")
            # Use precomputed distances if provided
            dists = csr_matrix(precomputed_distances)
        else:
            print("missing precomputed distances, calculating from scratch")
            print("n_neighbors",min(max(min_pts_values),clustering_data.shape[0]))
            # Calculate the matrix if not provided
            nn = NearestNeighbors(n_neighbors=min(max(min_pts_values),clustering_data.shape[0]), n_jobs=-1)
            nn.fit(clustering_data)
            dists = nn.kneighbors_graph(clustering_data)

        for idx, (eps, met, min_pts) in enumerate(it.product(eps_values, metric, min_pts_values)):
            start = time.time()
            print(f"Running DBSCAN: {idx} - eps={eps}, metric={met}, min_samples={min_pts}")
            try:
                # execute DBSCAN
                dbscan = DBSCAN(eps=eps, min_samples=min_pts, metric='precomputed', n_jobs=-1)
                labels = dbscan.fit_predict(dists)

                # Calculate silhouette score
                if len(set(labels)) > 1 and not (labels == labels[0]).all():
                    silhouette_score_val = silhouette_score(clustering_data, labels)
                else:
                    silhouette_score_val = "all core" if (labels == 1).all() else "all noise"

                end = time.time()
                print(f"DBSCAN done in {end - start:.2f} seconds | Silhouette Score: {silhouette_score_val}")

                # Add results to the dataframe
                new_conf = pd.DataFrame([{
                    'group_index': idx,
                    'eps': eps,
                    'metric': met,
                    'min_samples': min_pts,
                    'silhouette_score': silhouette_score_val,
                    'execution_time(s)': end - start
                }])
                results = pd.concat([results, new_conf])

            except Exception as e:
                print(f"Error with DBSCAN parameters (eps={eps}, min_samples={min_pts}): {e}")

    except Exception as e:
        print(f"General error: {e}")

    return results

def extract_connected_components(core_points,eps):

    # Create a graph to connect core points
    G = nx.Graph()

    # Add core points as nodes
    for i, point in enumerate(core_points):
        G.add_node(i, pos=point)

    # Add edges between neighboring core points
    nbrs = NearestNeighbors(radius=eps).fit(core_points)
    distances, indices = nbrs.radius_neighbors(core_points)

    for i, neighbors in enumerate(indices):
        for j in neighbors:
            if i != j:  # Avoid self-loops
                G.add_edge(i, j)

    # Extract connected components (clusters)
    connected_components = list(nx.connected_components(G))

    return connected_components

def random_sampling_reduce(data,reduction_percent):
    num_samples = data.shape[0]
    reduction_num_samples = int(np.ceil(reduction_percent*num_samples))
    RANDOM_SEED = 42
    np.random.seed(RANDOM_SEED)
    reduction_idx = np.random.choice(range(len(data)),
                                     reduction_num_samples, replace=False)
    return data.iloc[reduction_idx]
