import matplotlib.pyplot as plt

from .operations import *
from .calculate_indicators import *
from .visualizations import visualize_dataframe
from ... import load_csv, load_transformed_data_to_dataframe


def get_louvain_by_year_multilayer(year):
    df = load_transformed_data_to_dataframe(f"./Datasets/OECD_Transformed/OECD_{year}.csv")
    df = df[df['Value (million USD)'] > 1]
    graph = aggregate_by_country(df)
    return louvain(graph)

def get_graph_by_year_multilayer(year):
    df = load_transformed_data_to_dataframe(f"./Datasets/OECD_Transformed/OECD_{year}.csv")
    df = df[df['Value (million USD)'] > 1]
    graph = aggregate_by_country(df)
    return graph
