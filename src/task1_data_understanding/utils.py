# utils for task 1
# from procyclingstats import Stage

"""import random
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
from fp.fp import FreeProxy
import json
"""
import re
import matplotlib.pyplot as plt
import missingno as msno
import numpy as np
import unicodedata
import time
from tqdm import tqdm
from functools import partial


RACES_DTYPES = {
            '_url': 'str',
            'name': 'str',
            'points': 'Int64',
            'uci_points': 'Int64',
            'length': 'Int64',
            'climb_total': 'Int64',
            'profile': 'str',
            'starlist_quality': 'Int64',
            'average_temperature': 'float64',
            'position': 'Int64',
            'cyclist': 'str',
            'is_tarmac': 'bool',
            'is_cobbled': 'bool',
            'is_gravel': 'bool',
            'cyclist_team': 'str',
            'date': 'datetime64'
}


def plot_missing_values_barplot(missing_values_df):
    """
    Plots a bar plot of the percentage of missing values per feature.

    Parameters:
    ----------
    missing_values_df : pd.DataFrame
        A DataFrame containing the features and their corresponding missing value percentages.
        The DataFrame should have a column named 'missing values %' which contains the percentage
        of missing values for each feature.

    Returns:
    -------
    None
    """
    values = missing_values_df[missing_values_df['missing values %'] > 0]['missing values %']
    values['all other features'] = 0
    bars = values.sort_values(ascending=False).plot(kind='bar', figsize=(15, 10), ylim=[0, 100])
    plt.title('missing values percentage')
    plt.xlabel('features')
    plt.ylabel('missing values %')
    plt.xticks(rotation=0, ha='center')
    # adding percentage values above bars
    for bar in bars.patches:
        h = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, h, str(h), ha='center', va='bottom')


def plot_msno_matrix(missing_values_df):
    """
    Plots a matrix visualization of missing values using missingno's matrix plot.

    Parameters:
    ----------
    missing_values_df : pd.DataFrame
        A DataFrame that contains the data, including the 'date' column to visualize missing data
        distribution over time.

    Returns:
    -------
    None
    """
    msno.matrix(
            missing_values_df.set_index('date').sort_index(),
            sparkline=False,
            figsize=(10, 30)
        )
    years = list(missing_values_df['date'].dt.year.sort_values().unique())
    years

    locs, labels = plt.yticks()

    plt.title('Missing values distribution across years')

    y_start = locs[0]
    y_end = locs[1]

    y_ticks = np.linspace(y_start, y_end, len(years))

    plt.yticks(ticks=y_ticks, labels=years)


def plot_races_mv(races_df, url_df, mv_cols):
    """
    Plots a stacked horizontal bar chart showing the missing values for selected columns per race.

    Parameters:
    ----------
    races_df : pd.DataFrame
        The DataFrame containing race data.
    
    url_df : pd.DataFrame
        A DataFrame containing URL and race names for mapping purposes.
    
    mv_cols : list of str
        A list of column names that represent the fields to check for missing values.
    
    Returns:
    -------
    None
    """
    races_mv_df = races_df
    races_mv_df['url_name'] = url_df['name']
    races_mv_df = races_mv_df.groupby('url_name').apply(lambda x: x.isnull().sum())
    races_mv_ord = races_mv_df.sum(axis=1).sort_values().index
    races_mv_df[mv_cols].reindex(races_mv_ord).plot(kind='barh', stacked=True, figsize=(20, 10), title='missing values per race', xlabel='missing value counts', ylabel='race name', use_index=True)


def find_outliers(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = data[(data[column] < lower_bound) | (data[column] > upper_bound)]
    return outliers[[column]]


def normalize_text(text):
    # remove all accents,diacritic characters etc.etc.
    text = unicodedata.normalize('NFKD', text).encode('ASCII', 'ignore').decode('utf-8')
    # remove non alphanumeric value from the text
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return text.lower()


"""
def fetch_from_procyclingstas(races_df, races_path, delay_seconds=5, num_proxies=100, num_workers=200):
    races_url = list(races_df['_url'].unique())
    session = requests.Session()
    fp = FreeProxy(rand=True)
    user_agents = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/117.0.5938.132 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_5_2) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.6 Safari/605.1.15",
        "Mozilla/5.0 (Linux; Android 13; Pixel 7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/117.0.5938.132 Mobile Safari/537.36",
        "Mozilla/5.0 (Linux; Android 12; SM-G998B) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/117.0.5938.132 Mobile Safari/537.36",
        "Mozilla/5.0 (Windows NT 11.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/117.0.5938.132 Safari/537.36",
        "Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/117.0.5938.132 Safari/537.36",
        "Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Mobile/15E148 Safari/604.1",
        "Mozilla/5.0 (iPad; CPU OS 16_6 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.6 Mobile/15E148 Safari/604.1",
        "Mozilla/5.0 (Linux; Android 10; Nexus 5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/117.0.5938.132 Mobile Safari/537.36",
        "Mozilla/5.0 (Windows NT 6.1; WOW64; rv:78.0) Gecko/20100101 Firefox/78.0"
    ]
    proxies = []
    with ThreadPoolExecutor(max_workers=num_workers) as tpe:
        proxies = list(tpe.map(lambda _: fp.get(), range(num_proxies)))

    def fetch_race(url, delay):
        session.headers.update({
            'User-Agent': random.choice(user_agents)
        })
        proxy = random.choice(proxies)
        session.proxies = {
            'http': proxy,
            'https': proxy
        }
        stage = Stage("race/"+url)
        time.sleep(delay)
        return stage
    results = []
    with ThreadPoolExecutor(max_workers=num_workers) as tpe:
        futures = {tpe.submit(fetch_race, url, delay_seconds): url for url in races_url}
        pbar = tqdm(total=len(futures), desc="loading data from procyclingstats")
        for future in as_completed(futures):
            try:
                pbar.update(1)
                race = future.result()
                new_row = race.parse()
                new_row['url'] = futures[future]
                results.append(new_row)
            except Exception as e:
                print(e)
                pass
    with open(races_path, 'w') as f:
        json.dump(results, f, indent=6)
"""


def map_place_to_point(stages_df, geolocator):
    places = set(stages_df['arrival'])
    places.update(set(stages_df['departure']))
    places.remove('')
    geocode = partial(geolocator.geocode, language="en")
    places_info = []
    for place in tqdm(places, total=len(places)):
        places_info.append(geocode(place))
        time.sleep(1)
    places_info = [info for info in places_info if info is not None]
    return places_info

def stages_order(stage):
    is_alpha=stage[-1].isalpha()
    v= stage if not is_alpha else stage[:-1]
    ord_val=int(v)*10
    if is_alpha:
        c=stage[-1]
        ord_val+=1 if c=='a' else 2
    return ord_val