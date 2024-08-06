import numpy as np
import pandas as pd
import networkx as nx
import plotly.graph_objects as go
from matplotlib import pyplot as plt

from layered_network_graph import LayeredNetworkGraph

dataf = pd.read_csv("Datasets/OECD_Original/ICIO2023_1996.csv", index_col=0)
data2 = pd.read_csv("Datasets/OECD_Original/ICIO2023_1996.csv", index_col=0)

# only use the relevant data (REDUCED DATASET)
num_of_entries = len(dataf.index) - 2
dataf = dataf.iloc[:num_of_entries-1, :num_of_entries]
data2 = data2.iloc[:num_of_entries-1, :num_of_entries]
# dataf = dataf.iloc[:100, :100]
# data2 = data2.iloc[:100, :100]


G = nx.from_pandas_adjacency(dataf)
# G2 = nx.from_pandas_adjacency(data2)
# nx.draw_networkx(G)

print(G.number_of_edges(), G.number_of_nodes()) # E: 3982729   N: 3465

raise "asd"
nx.Graph()

node_labels = [dataf.columns]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
LayeredNetworkGraph([G, G2], ax=ax, layout=nx.fruchterman_reingold_layout)
ax.set_axis_off()
plt.show()
