import matplotlib.pyplot as plt

from .operations import *
from .calculate_indicators import *
from .visualizations import visualize_dataframe
from ... import load_csv


# Works with original data
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


# Works with transformed data
def visualize_multilayer_graph(df):

    sector_map = load_csv("./Datasets/sector_LUT.csv", "\ufeffID", "Code")
    country_map = load_csv("./Datasets/country_LUT.csv", "\ufeffID", "Code")
    # Only consider routes with at least $1M trade value
    df = df[df['Value (million USD)'] > 1]
    # Countries to display
    # EU
    allowed_values = [
        3, 4, 6, 18, 19, 20, 21, 23, 24, 25, 26, 28, 30, 31,
        34, 37, 44, 45, 46, 49, 53, 59, 60, 61, 66, 67, 68
    ]
    df = df[df['Exporter_country'].isin(allowed_values)]
    df = df[df['Importer_country'].isin(allowed_values)]
    # Sectors to display
    allowed_values = [
        1, 2, 35
    ]
    df = df[df['Exporter_sector'].isin(allowed_values)]
    df = df[df['Importer_sector'].isin(allowed_values)]

    visualize_dataframe(df, sector_map, country_map)

