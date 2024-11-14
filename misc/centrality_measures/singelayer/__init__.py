from .calculate_indicators import *
from .operations import *


def calculate_singe_layer_indicators(df):
    graph = create_graph_from_dataframe(df)

    print(betweenness_centrality(graph))
    print(closeness_centrality(graph))
    print(degree_centrality(graph))
    print(edge_betweenness_centrality(graph))
    print(pagerank(graph))
    print(density(graph))
