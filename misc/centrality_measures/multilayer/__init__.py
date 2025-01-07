import matplotlib.pyplot as plt

from .operations import *
from .calculate_indicators import *
from .visualizations import visualize_dataframe
from ... import load_csv, load_transformed_data_to_dataframe


# Works with original data
def viz(data):
    fig = px.bar(x=list(data.keys()), y=list(data.values()))
    fig.show()


def calculate_multilayer_network_indicators(df, selector='sector'):
    df = df[df['Value (million USD)'] > 1]
    if selector == 'sector':
        print("Calculating indicators for sector-layered network...")
        graph = aggregate_by_sector(df)
    elif selector == 'country':
        print("Calculating indicators for country-layered network...")
        graph = aggregate_by_country(df)
    else:
        raise Exception('Invalid selector')


    bc = betweenness_centrality(graph)
    print(bc)
    viz(bc)
    # cc = closeness_centrality(graph)
    # viz(cc)
    # dc = degree_centrality(graph)
    # print(dc)
    # viz(dc)
    # return dc
    # # ebc = edge_betweenness_centrality(graph)
    # # viz(ebc)
    # pr = pagerank(graph)
    # viz(pr)
    print("\n")

def get_hc_by_year(year):
    df = load_transformed_data_to_dataframe(f"./Datasets/OECD_Transformed/OECD_{year}.csv")
    df = df[df['Value (million USD)'] > 1]
    graph = aggregate_by_country(df)
    return closeness_centrality(graph)


def get_bc_by_year(year):
    df = load_transformed_data_to_dataframe(f"./Datasets/OECD_Transformed/OECD_{year}.csv")
    df = df[df['Value (million USD)'] > 1]
    graph = aggregate_by_country(df)
    return betweenness_centrality(graph)


def get_pr_by_year(year):
    df = load_transformed_data_to_dataframe(f"./Datasets/OECD_Transformed/OECD_{year}.csv")
    # df = df[df['Value (million USD)'] > 1]
    graph = aggregate_by_country(df)
    return pagerank(graph)

def get_network_country_yearly(df):
    # df = df[df['Value (million USD)'] > 1]
    graph = aggregate_by_country(df)
    dc = degree_centrality(graph)
    return dc

# Works with transformed data
def visualize_multilayer_graph(df, sector_map, country_map):
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

