#utils for task 1
import matplotlib.pyplot as plt
import missingno as msno
import numpy as np


RACES_DTYPES={
            '_url':'str',
            'name':'str',
            'points':'Int64',
            'uci_points':'Int64',
            'length':'Int64',
            'climb_total':'Int64',
            'profile':'str',
            'starlist_quality':'Int64',
            'average_temperature':'float64',
            'position':'Int64',
            'cyclist':'str',
            'is_tarmac':'bool',
            'is_cobbled':'bool',
            'is_gravel':'bool',
            'cyclist_team':'str',
            'date':'datetime64'
}

def plot_missing_values_barplot(missing_values_df):
    values=missing_values_df[missing_values_df['missing values %']>0]['missing values %']
    values['all other features']=0
    bars=values.sort_values(ascending=False).plot(kind='bar',figsize=(15,10),ylim=[0,100])
    plt.title('missing values percentage')
    plt.xlabel('features')
    plt.ylabel('missing values %')
    plt.xticks(rotation=0, ha='center')
    # adding percentage values above bars
    for bar in bars.patches:
        h=bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, 
                h, 
                str(h), 
                ha='center', va='bottom')

def plot_msno_matrix(missing_values_df):
    msno.matrix(
            missing_values_df.set_index('date').sort_index(),
            sparkline=False,
            figsize=(10,30)
        )
    years=list(missing_values_df['date'].dt.year.sort_values().unique())
    years

    locs,labels=plt.yticks()

    plt.title('Missing values distribution across years')

    y_start=locs[0]
    y_end=locs[1]

    y_ticks=np.linspace(y_start,y_end,len(years))

    plt.yticks(ticks=y_ticks,labels=years)

def plot_races_mv(races_df,url_df):
    races_mv_df=races_df
    races_mv_df['url_name']=url_df['name']
    races_mv_df=races_mv_df.groupby('url_name').apply(lambda x: x.isnull().sum())
    races_mv_ord=races_mv_df.sum(axis=1).sort_values().index
    races_mv_df[mv_cols].reindex(races_mv_ord).plot(kind='barh',stacked=True,figsize=(20,10),title='missing values per race',xlabel='missing value counts',ylabel='race name',use_index=True)