
from sklearn.cluster import DBSCAN
import pandas as pd
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.metrics import silhouette_score
import itertools as it
import numpy as np
#utils for task 3
def run_dbscan(min_pts_values,eps_values,metric,clustering_data,precomputed_distances=None):
    results=pd.DataFrame([{}])
    for idx,(eps,metric,min_pts) in enumerate(it.product(eps_values,metric,min_pts_values)):
        labels=DBSCAN(
            eps=eps,
            min_samples=min_pts,
            metric='euclidean',
            n_jobs=-1
        ).fit_predict(precomputed_distances)
        #NOTE: the noisy labels are NOT taken into account 
        new_conf=pd.DataFrame([{
                'group_index':idx,
                'eps':eps,
                'metric':metric,
                'min_samples':min_pts,
                'silhoutte_score':silhouette_score(clustering_data,labels)
            }])
        results=pd.concat([results,new_conf])
        print("finished\n",results.iloc[-1])
    return results