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
    df_aggregated = df_aggregated.rename(columns={'Value (million USD)': 'weight'})
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

# Intralayer


def filter_by_sector_intralayer(df, sector):
    df_filtered = df[(df['Exporter_sector'] == sector) | (df['Importer_sector'] == sector)]
    df_filtered = df_filtered.rename(columns={'Value (million USD)': 'weight'})
    graph = nx.from_pandas_edgelist(df_filtered, source='Exporter_sector', target='Importer_sector',
                                    edge_attr='weight', create_using=nx.DiGraph())

    reduced_graph = remove_non_weighed_edges(graph)
    return reduced_graph


def filter_by_country_intralayer(df, country):
    df_filtered = df[(df['Exporter_country'] == country) | (df['Importer_country'] == country)]
    df_filtered = df_filtered.rename(columns={'Value (million USD)': 'weight'})
    graph = nx.from_pandas_edgelist(df_filtered, source='Exporter_country', target='Importer_country',
                                    edge_attr='weight', create_using=nx.DiGraph())

    reduced_graph = remove_non_weighed_edges(graph)
    return reduced_graph

# Interlayer


def filter_by_sector_interlayer(df, sector):
    df_filtered = df[(df['Exporter_sector'] == sector) | (df['Importer_sector'] != sector)]
    df_filtered = df_filtered.rename(columns={'Value (million USD)': 'weight'})
    graph = nx.from_pandas_edgelist(df_filtered, source='Exporter_sector', target='Importer_sector',
                                    edge_attr='weight', create_using=nx.DiGraph())

    reduced_graph = remove_non_weighed_edges(graph)
    return reduced_graph


def filter_by_country_interlayer(df, country):
    df_filtered = df[(df['Exporter_country'] == country) | (df['Importer_country'] != country)]
    df_filtered = df_filtered.rename(columns={'Value (million USD)': 'weight'})
    graph = nx.from_pandas_edgelist(df_filtered, source='Exporter_country', target='Importer_country',
                                    edge_attr='weight', create_using=nx.DiGraph())

    reduced_graph = remove_non_weighed_edges(graph)
    return reduced_graph
