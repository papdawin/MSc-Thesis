import numpy as np
import pandas as pd
import networkx as nx
import plotly.graph_objects as go
from matplotlib import pyplot as plt
from networkx.drawing.nx_agraph import graphviz_layout

dataf = pd.read_csv("../Datasets/OECD_Original/ICIO2023_2020.csv", index_col=0)

num_of_entries = len(dataf.index) - 2
dataf = dataf.iloc[:num_of_entries-1, :num_of_entries]

selected_rows = dataf[dataf.index.str.contains('BRN')]

# print(selected_rows)
print(selected_rows.sum().sort_values(ascending=False).head(20))
# print(dataf)

