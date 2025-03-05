import pandas as pd
import networkx as nx
import plotly.express as px


def remove_loop_edges(graph):
    graph.remove_edges_from(nx.selfloop_edges(graph))
    return graph


def louvain(graph):
    graph = remove_loop_edges(graph)
    network_density = nx.community.louvain_communities(graph)
    return network_density
