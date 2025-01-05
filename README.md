# 📊 dm_project24_group_6
This is the repository for the group 6 of the Data Mining course 2024/2025 at the University of Pisa. This repository contains the code and the notebooks used for the analysis of the dataset provided for the project.
You can read our final report here: [Final Report](https://github.com/DadeOrsu/dm_project24_group_6/blob/main/report.pdf)
# 👨‍💻 candidates
- [Mirko Michele D'Angelo](https://github.com/mirdan08)
- [Filippo Morelli](https://github.com/ff-falco)
- [Davide Orsucci](https://github.com/DadeOrsu)

# 🗃️ Repo outline
The repository is organized as follows:
```bash
📂 .
├── 🛠️ environment.yml
├── 📄 project.pdf
├── 📘 README.md
└── 📂 src
    ├── 📁 dataset
    ├── 🐍 generic_utils.py
    ├── 📂 task1_data_understanding
    │   ├── 📒 cyclist_analysis.ipynb
    │   ├── 📒 data_distribution_refined.ipynb
    │   ├── 🐍 dataunderstanding.py
    │   ├── 📒 races_analysis.ipynb
    │   ├── 🐍 transformations.py
    │   └── 🐍 utils.py
    ├── 📂 task2_data_transformation
    │   ├── 📒 feature engineering_cyclists.ipynb
    │   ├── 📒 feature_engineering.ipynb
    │   ├── 📒 outlier_detection.ipynb
    │   ├── 📒 races_understanding.ipynb
    │   └── 🐍 utils.py
    ├── 📂 task3_clustering
    │   ├── 📒 dbscan.ipynb
    │   ├── 📒 hierarchical.ipynb
    │   ├── 📒 kmeans_clustering.ipynb
    │   ├── 📒 optics.ipynb
    │   ├── 🐍 transformations.py
    │   └── 🐍 utils.py
    ├── 📂 task4_prediction
    │   ├── 📒 decisione_trees_classification .ipynb
    │   ├── 📒 knn_classification.ipynb
    │   ├── 🗂️ params_dt
    │   ├── 🗂️ params_knn
    │   ├── 🗂️ params_ripper
    │   ├── 📒 ripper_classification.ipynb
    │   ├── 📒 bagging_classification.ipynb
    │   ├── 📒 decision_trees_classification .ipynb
    │   ├── 📒 nn_classification.ipynb
    │   ├── 📒 ripper_classification.ipynb
    │   ├── 📒 boosting.ipynb  
    │   └──  🐍 preprocessing.py
    └── 📂 task5_xai
        ├── 📒 bagging_explanation.ipynb
        ├── 🐍 preprocessing.py
        ├── 🐍 transformations.py
        └── 📒 xgbc_explanation.ipynb
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
    -  `task4_prediction`: folder of files for the fourth task of the project
        - `decision_trees_classification .ipynb`: notebook containing the decision tree classification of the data
        - `knn_classification.ipynb`: notebook containing the KNN classification of the data
        - `params_dt`: folder containing the parameters for the decision tree classification
        - `params_knn`: folder containing the parameters for the KNN classification
        - `params_ripper`: folder containing the parameters for the RIPPER classification
        - `ripper_classification.ipynb`: notebook containing the RIPPER classification of the data
        - `bagging_classification.ipynb`: notebook containing the bagging classification of the data
        - `decisione_trees_classification .ipynb`: notebook containing the decision tree classification of the data
        - `nn_classification.ipynb`: notebook containing the neural network classification of the data
        - `ripper_classification.ipynb`: notebook containing the RIPPER classification of the data
        - `boosting.ipynb`: notebook containing the boosting classification of the data
        - `preprocessing.py`: utility functions for the preprocessing of the data
   - `task5_xai`: folder of files for the fifth task of the project
        - `bagging_explanation.ipynb`: notebook containing the explanation of the bagging classification
        - `preprocessing.py`: utility functions for the preprocessing of the data
        - `transformations.py`: utility functions for the normalization task
        - `xgbc_explanation.ipynb`: notebook containing the explanation of the XGBoost classification


# Resources 

[Download Dataset](http://didawiki.cli.di.unipi.it/lib/exe/fetch.php/magistraleinformatica/dmi/dataset.tar)

[Project Instructions](http://didawiki.cli.di.unipi.it/lib/exe/fetch.php/magistraleinformatica/dmi/project.pdf)