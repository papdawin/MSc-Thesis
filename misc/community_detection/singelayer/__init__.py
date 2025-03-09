from .calculate_indicators import *
from .operations import *
from ...localmemory_operations import load_transformed_data_to_dataframe


def get_louvain_by_year(year):
    df = load_transformed_data_to_dataframe(f"./Datasets/OECD_Original/ICIO2023_{year}.csv")
    graph = create_graph_from_dataframe(df)
    return louvain(graph)


def get_corep_by_year(year):
    df = load_transformed_data_to_dataframe(f"./Datasets/OECD_Original/ICIO2023_{year}.csv")
    graph = create_graph_from_dataframe(df)
    graph.remove_nodes_from(list(nx.isolates(graph)))
    print(nx.is_connected(graph))
    return corep(graph)

def get_graph_by_year(year):
    df = load_transformed_data_to_dataframe(f"./Datasets/OECD_Original/ICIO2023_{year}.csv")
    graph = create_graph_from_dataframe(df)
    return graph
