import pandas as pd
from os import path


def get_train_test_data():
    # Load the data
    races_final_path = path.join('..', 'dataset', 'engineered_races.csv')
    cyclists_final_path = path.join('..', 'dataset',
                                    'cyclists_final_enhanced.csv')

    cyclists_data = pd.read_csv(cyclists_final_path)
    races_data = pd.read_csv(races_final_path)
    # Merge the two datasets
    cyclists_data.rename(columns={'name': 'cyclist'}, inplace=True)

    merged_data = races_data.merge(cyclists_data, left_on='cyclist',
                                   right_on='_url', how='inner')
    # Create the target variable
    merged_data['top_20'] = merged_data['position'].apply(lambda x: 1 if x <= 20 else 0)

    merged_data['date'] = pd.to_datetime(merged_data['date'])
    # Create the feature set
    columns_to_keep = [
        'bmi', 'career_points','career_duration(days)','debut_year',
        'difficulty_score', 'competitive_age', 'is_tarmac', 'points',
        'climbing_efficiency', 'startlist_quality', 'top_20'
    ]
    # Split the data into train and test sets based on the date
    train_set = merged_data[merged_data['date'] < '2022-01-01']
    test_set = merged_data[merged_data['date'] >= '2022-01-01']

    train_set = train_set[columns_to_keep]
    test_set = test_set[columns_to_keep]

    X_train = train_set.drop(columns=['top_20'])
    y_train = train_set['top_20']

    X_test = test_set.drop(columns=['top_20'])
    y_test = test_set['top_20']
    # Return the train and test sets
    return X_train, y_train, X_test, y_test, columns_to_keep
