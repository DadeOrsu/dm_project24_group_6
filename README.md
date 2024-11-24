# 📊 dm_project24_group_6
Thi is the repository for the group 6 of the Data Mining course 2024/2025 at the University of Pisa. This repository contains the code and the notebooks used for the analysis of the dataset provided for the project.
# 🗃️ Repo outline
The repository is organized as follows:
```bash
📁 Root
├── 📄 environment.yml
├── 📄 project.pdf
├── 📄 README.md
└── 📂 src
    ├── 📄 dataframe_types.py
    ├── 📂 dataset
    ├── 📄 generic_utils.py
    ├── 📂 task1_data_understanding
    │   ├── 📓 cyclist_analysis.ipynb
    │   ├── 📓 data_distribution_refined.ipynb
    │   ├── 📄 dataunderstanding.py
    │   ├── 📓 races_analysis.ipynb
    │   ├── 📄 transformations.py
    │   └── 📄 utils.py
    ├── 📂 task2_data_transformation
    │   ├── 📓 feature engineering_cyclists.ipynb
    │   ├── 📓 feature_engineering.ipynb
    │   ├── 📓 outlier_detection.ipynb
    │   ├── 📓 races_understanding.ipynb
    │   └── 📄 utils.py
    └── 📂 task3_clustering
        ├── 📓 dbscan.ipynb
        ├── 📓 hierarchical.ipynb
        ├── 📓 kmeans_clustering.ipynb
        ├── 📄 transformations.py
        └── 📄 utils.py
```
## 🔍 Description
- `environment.yml`: file containing the environment used for the project.
- `project.pdf`: file containing the instructions for the project.
- `README.md`: file containing the description of the repository.
- `src`: contains folders with scripts and notebooks used for the analysis
    - `dataframe_types.py`: file containing the definition of the dataframes used for the project.
    - `generic_utils.py`: file containing utility functions used for the project.
    - `dataset`: folder containing the dataset used for the project.
    - `task1_data_understanding`: folder of files for the first task of the project
        - `cyclist_analysis.ipynb`: notebook containing the analysis of the cyclists' dataframe
        - `data_distribution_refined.ipynb`: notebook containing the analysis of the distribution of the data 
        - `dataunderstanding.py`: utility functions for the data understanding task
        - `races_analysis.ipynb`: notebook containing the analysis of the races' dataframe
        - `transformations.py`: utility functions for normalizing the data.
        - `utils.py`: utility functions for the data understanding task
    - `task2_data_transformation`: folder of files for the second task of the project
        - `feature engineering_cyclists.ipynb`: notebook containing the feature engineering and newer understanding of the cyclists' dataframe.
        - `feature_engineering.ipynb`: notebook containing the feature engineering of the races' dataframe
        - `outlier_detection.ipynb`: notebook containing the outlier detection of the data both for the cyclists and races.
        - `races_understanding.ipynb`: notebook containing the data understanding of the new features of the races.
        - `utils.py`: utility functions for the data transformation task
    - `task3_clustering`: folder of files for the third task of the project
        - `dbscan.ipynb`: notebook containing the DBSCAN clustering of the data
        - `hierarchical.ipynb`: notebook containing the hierarchical clustering of the data
        - `kmeans_clustering.ipynb`: notebook containing the KMeans clustering of the data
        - `transformations.py`: utility functions for the normalization task
        - `utils.py`: utility functions for the clustering task

The repository will be updated as the project progresses.
# Resources 

[download dataset](http://didawiki.cli.di.unipi.it/lib/exe/fetch.php/magistraleinformatica/dmi/dataset.tar)

[download project instructions](http://didawiki.cli.di.unipi.it/lib/exe/fetch.php/magistraleinformatica/dmi/project.pdf)