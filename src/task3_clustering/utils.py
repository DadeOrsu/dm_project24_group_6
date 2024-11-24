
from sklearn.cluster import DBSCAN
import pandas as pd
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
def run_dbscan(min_pts_values,eps_values,metric,clustering_data,precomputed_distances=None):
    results=pd.DataFrame([])

    nn=NearestNeighbors(n_neighbors=clustering_data.shape[0]-1,n_jobs=-1).fit(csr_matrix(clustering_data))
    dists=nn.kneighbors_graph(clustering_data)
     
    for idx,(eps,metric,min_pts) in enumerate(it.product(eps_values,metric,min_pts_values)):
        start=time.time()
        print(f"-{idx} - {(eps,metric,min_pts)}")
        labels=DBSCAN(
            eps=eps,
            min_samples=min_pts,
            metric='precomputed',
            n_jobs=-1
        ).fit_predict(dists)
        end=time.time()
        silhouette_score_val=silhouette_score(clustering_data,labels) if not (labels==labels[0]).all() else "all core" if (labels==1).all() else "all noise" 
        print(f"dbscan done, time={end-start} seconds | silhoutte score:{silhouette_score_val}")
        #NOTE: the noisy labels are NOT taken into account 
        new_conf=pd.DataFrame([{
                'group_index':idx,
                'eps':eps,
                'metric':metric,
                'min_samples':min_pts,
                'silhoutte_score':silhouette_score_val,
                'execution_time(s)':end-start
            }])
        results=pd.concat([results,new_conf])
    return results

def random_sampling_reduce(data,reduction_percent):
    num_samples=data.shape[0]

    reduction_num_samples=int(np.ceil(reduction_percent*num_samples))

    RANDOM_SEED=42

    np.random.seed(RANDOM_SEED)

    reduction_idx=np.random.choice(range(len(clustering_data)),reduction_num_samples,replace=False)

    return data.iloc[reduction_idx]