import pandas as pd
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection

def extract_sector(col_name, part):
    return col_name.split('_')[part]


def remove_non_weighed_edges(graph):
    edges_to_remove = [(u, v) for u, v, w in graph.edges(data='weight') if w == 0]
    graph.remove_edges_from(edges_to_remove)
    return graph


def aggregate_by_sector(df):
    df_aggregated = df.groupby(['Exporter_sector', 'Importer_sector']).agg({'Value (million USD)': 'sum'})
    df_aggregated = df_aggregated.reset_index()
    df_aggregated = df.rename(columns={'Value (million USD)': 'weight'})
    graph = nx.from_pandas_edgelist(df_aggregated, source='Exporter_sector', target='Importer_sector',
                            edge_attr='weight', create_using=nx.DiGraph())

    reduced_graph = remove_non_weighed_edges(graph)
    return reduced_graph


def aggregate_by_country(df):
    df_aggregated = df.groupby(['Exporter_country', 'Importer_country']).agg({'Value (million USD)': 'sum'})
    df_aggregated = df_aggregated.reset_index()
    df_aggregated = df.rename(columns={'Value (million USD)': 'weight'})
    graph = nx.from_pandas_edgelist(df_aggregated, source='Exporter_country', target='Importer_country',
                            edge_attr='weight', create_using=nx.DiGraph())

    reduced_graph = remove_non_weighed_edges(graph)
    return reduced_graph
