import sklearn
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import seaborn as sns
from scipy import spatial
from datetime import datetime as dt
import pandas as pd
import datetime
import holoviews as hv
import matplotlib.pyplot as plt
from bokeh.plotting import show


hv.extension('bokeh')


def similarity_heatmap(embs: np.array):
    print("Temporal similarity plot:")
    similarity = sklearn.metrics.pairwise.cosine_similarity(embs, embs)
    heatmap = sns.heatmap(similarity, vmin=0, vmax=1)
    heatmap.figure.savefig('similarity_heatmap.svg', format='svg')
    return heatmap


def plot_trend_over_time(embs: np.array, google_trends_df: pd.DataFrame, anomalies: list):
    dist = np.array([spatial.distance.cosine(embs[i - 1], embs[i]) for i in range(1, len(embs))])
    scaler = MinMaxScaler()
    dist = scaler.fit_transform(dist.reshape(1, -1).T).squeeze()

    google_trends_df['google'] = scaler.fit_transform(google_trends_df['google'].values.reshape(-1, 1))
    google_trends_df['graphert_delta'] = dist

    googletrends = hv.Curve((google_trends_df['Day'], google_trends_df['google']), 'Time', 'Score',
                            label='Google Trends')
    graphert_delta = hv.Curve((google_trends_df['Day'], google_trends_df['graphert_delta']), 'Time', 'Score',
                            label='TAAGNN')
    scat_anoms = hv.Scatter(data=google_trends_df[google_trends_df['Day'].isin(anomalies)],
                            vdims=['graphert_delta']).opts(
        width=700, height=500, size=10, fill_color="red", fontsize={'labels': 24, 'xticks': 16, 'yticks': 16})
    googletrends.opts(color='#FFA500', line_dash='dashed')
    graphert_delta.opts(color='#00BFFF')
    overlay = (graphert_delta * googletrends * scat_anoms)
    overlay.opts(legend_position='bottom_right')
    return overlay


if __name__ == '__main__':
    dataset_name = 'game_of_thrones'
    model_path = f'datasets/{dataset_name}/models/mlm_and_temporal_model'
    # get temporal embeddings by the last layer
    temporal_embeddings = get_temporal_embeddings(model_path)

    google_trends_df = pd.read_csv(f"data/{dataset_name}/google_trends.csv",
                                   parse_dates=['Day'], date_parser=
                                   lambda x: dt.strptime(x, "%d-%m-%y"))

    anoms = [datetime.date(2017, 7, 17), datetime.date(2017, 7, 24),
             datetime.date(2017, 7, 31),
             datetime.date(2017, 8, 7), datetime.date(2017, 8, 14), datetime.date(2017, 8, 21),
             datetime.date(2017, 8, 28)]

    heatmap = similarity_heatmap(temporal_embeddings)
    plt.show()
    trend_plot = plot_trend_over_time(embs=temporal_embeddings,
                                      google_trends_df=google_trends_df,
                                      anomalies=anoms)
    show(hv.render(trend_plot))
