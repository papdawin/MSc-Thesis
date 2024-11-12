from .operations import *
from .calculate_indicators import *


def calculate_multilayer_network_indicators(df, selector='sector'):
    if selector == 'sector':
        print("Calculating indicators for sector-layered network...")
        graph = aggregate_by_sector(df)
    elif selector == 'country':
        print("Calculating indicators for country-layered network...")
        graph = aggregate_by_country(df)
    else:
        raise Exception('Invalid selector')
    print(betweenness_centrality(graph))
    print(closeness_centrality(graph))
    print(degree_centrality(graph))
    print(edge_betweenness_centrality(graph))
    print(pagerank(graph))
    print("\n")

