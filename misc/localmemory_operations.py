import pandas as pd
import networkx as nx


def load_database_to_graph(csv_path):
    dataf = pd.read_csv(csv_path, index_col=0)

    num_of_entries = len(dataf.index) - 2
    dataf = dataf.iloc[:num_of_entries - 1, :num_of_entries]

    G = nx.from_pandas_adjacency(dataf)
    return G




