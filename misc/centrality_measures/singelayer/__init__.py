from .calculate_indicators import *
from .operations import *
from .visualizations import plot_by_continents


def calculate_singe_layer_indicators(df):
    graph = create_graph_from_dataframe(df)

    print(betweenness_centrality(graph))
    # print(closeness_centrality(graph))
    # print(degree_centrality(graph))
    # print(edge_betweenness_centrality(graph))
    # print(pagerank(graph))
    # print(density(graph))


def compare_degree_centralities(df, df2):
    graph = create_graph_from_dataframe(df)
    graph2 = create_graph_from_dataframe(df2)
    dc = degree_centrality(graph)
    dc2 = degree_centrality(graph2)
    plot_by_continents(dc, dc2, 1995, 2020)