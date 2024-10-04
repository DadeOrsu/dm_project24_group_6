#utils for task 1
import matplotlib.pyplot as plt
def plot_missing_values_barplot(missing_values_df):
    values=missing_values_df[missing_values_df['missing values %']>0]['missing values %']
    values['all other values']=0
    bars=values.sort_values(ascending=False).plot(kind='bar',figsize=(15,10),ylim=[0,100])
    plt.title('missing values %')
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