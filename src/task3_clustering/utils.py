
from sklearn.cluster import DBSCAN
import pandas as pd
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.metrics import silhouette_score
import itertools as it
#utils for task 3
def run_dbscan(min_pts_values,eps_values,metric,clustering_data):
    results=pd.DataFrame()
    for idx,(eps,metric,min_pts) in enumerate(it.product(eps_values,metric,min_pts_values)):
        db_scan=DBSCAN(
            eps=eps,
            min_samples=min_pts,
            metric=metric,
            n_jobs=-1
        ).fit(clustering_data)
        #NOTE: the noisy labels are NOT taken into account 
        results=pd.concat([
            results,
            {
                'index':idx,
                'eps':eps,
                'metric':metric,
                'min_samples':min_pts,
                'silhoutte_score':silhouette_score(clustering_data,db_scan.labels_)
            }
        ])
        print(results[-1])
    return results