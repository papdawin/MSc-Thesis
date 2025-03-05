import pandas as pd
import networkx as nx
import plotly.express as px

from .operations import transform_to_df_and_order

seed = 123


def louvain(graph):
    louvain_values = nx.community.louvain_communities(graph, seed=seed)
    return louvain_values


def corep(graph):
    periphery_values = nx.periphery(graph)
    return periphery_values

