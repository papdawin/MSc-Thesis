import pandas as pd
import networkx as nx
import plotly.express as px


def remove_loop_edges(graph):
    graph.remove_edges_from(nx.selfloop_edges(graph))
    return graph


def betweenness_centrality(graph):
    graph = remove_loop_edges(graph)
    centrality_values = nx.betweenness_centrality(graph)
    return centrality_values


def closeness_centrality(graph):
    graph = remove_loop_edges(graph)
    centrality_values = nx.closeness_centrality(graph)
    return centrality_values


def degree_centrality(graph):
    graph = remove_loop_edges(graph)
    centrality_values = nx.degree_centrality(graph)
    return centrality_values


def edge_betweenness_centrality(graph):
    graph = remove_loop_edges(graph)
    centrality_values = nx.edge_betweenness_centrality(graph)
    return centrality_values


def pagerank(graph):
    graph = remove_loop_edges(graph)
    centrality_values = nx.pagerank(graph)
    return centrality_values


def density(graph):
    graph = remove_loop_edges(graph)
    network_density = nx.density(graph)
    return network_density
