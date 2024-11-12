import pandas as pd
import networkx as nx
import plotly.express as px

from .operations import transform_to_df_and_order

def betweenness_centrality(graph):
    centrality_values = nx.betweenness_centrality(graph)
    transform_to_df_and_order(centrality_values)
    return centrality_values


def closeness_centrality(graph):
    centrality_values = nx.closeness_centrality(graph)
    transform_to_df_and_order(centrality_values)
    return centrality_values


def degree_centrality(graph):
    centrality_values = nx.degree_centrality(graph)
    transform_to_df_and_order(centrality_values)
    return centrality_values


def edge_betweenness_centrality(graph):
    centrality_values = nx.edge_betweenness_centrality(graph)
    transform_to_df_and_order(centrality_values)
    return centrality_values


def pagerank(graph):
    centrality_values = nx.pagerank(graph)
    transform_to_df_and_order(centrality_values)
    return centrality_values


def density(graph):
    network_density = nx.density(graph)
    return network_density


def homophily(graph):
    def homophily_without_ids(G, chars):
        num_same_ties = 0
        num_ties = 0
        for n1, n2 in G.edges():
            if n1 in chars and n2 in chars:
                if G.has_edge(n1, n2):
                    num_ties += 1
                    if chars[n1] == chars[n2]:
                        num_same_ties += 1
        return (num_same_ties / num_ties)

