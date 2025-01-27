import pandas as pd
import numpy as np
from sklearn import cluster
import plotly.graph_objects as go

data = pd.read_csv("CC GENERAL.csv")

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

print(data.head())

pd.reset_option('display.max_rows')
pd.reset_option('display.max_columns')

print(data.isnull().sum())

data = data.dropna()

clustering_data = data[["BALANCE", "PURCHASES", "CREDIT_LIMIT"]]
from sklearn.preprocessing import MinMaxScaler

for i in clustering_data.columns:
    MinMaxScaler(i)

from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=6)
clusters = kmeans.fit_predict(clustering_data)
data["CREDIT_CARD_SEGMENTS"] = clusters

data["CREDIT_CARD_SEGMENTS"] = data["CREDIT_CARD_SEGMENTS"].map({0: "Cluster 1", 1:
    "Cluster 2", 2: "Cluster 3", 3: "Cluster 4", 4: "Cluster 5", 5: "Cluster 6", 6: "Cluster 7" , 7:
    "Cluster 8" , 8: "Cluster 9", 9: "Cluster 10"})
print(data["CREDIT_CARD_SEGMENTS"].head(10))
variance = kmeans.inertia_
print(f"Total Within-Cluster Sum of Squares (Variance): {variance:.4f}")

PLOT = go.Figure()
for i in list(data["CREDIT_CARD_SEGMENTS"].unique()):
    PLOT.add_trace(go.Scatter3d(x=data[data["CREDIT_CARD_SEGMENTS"] == i]['BALANCE'],
                                y=data[data["CREDIT_CARD_SEGMENTS"] == i]['PURCHASES'],
                                z=data[data["CREDIT_CARD_SEGMENTS"] == i]['CREDIT_LIMIT'],
                                mode='markers', marker_size=6, marker_line_width=1,
                                name=str(i)))
PLOT.update_traces(hovertemplate='BALANCE: %{x} <br>PURCHASES %{y} <br>DCREDIT_LIMIT: %{z}')

PLOT.update_layout(width=800, height=800, autosize=True, showlegend=True,
                   scene=dict(xaxis=dict(title='BALANCE', titlefont_color='black'),
                              yaxis=dict(title='PURCHASES', titlefont_color='black'),
                              zaxis=dict(title='CREDIT_LIMIT', titlefont_color='black')),
                   font=dict(family="Gilroy", color='black', size=12))
PLOT.show()