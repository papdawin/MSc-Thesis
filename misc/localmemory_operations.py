import pandas as pd
import networkx as nx
import csv


def load_database_to_graph(csv_path):
    dataf = pd.read_csv(csv_path, index_col=0)

    num_of_entries = len(dataf.index) - 2
    dataf = dataf.iloc[:num_of_entries - 1, :num_of_entries]

    G = nx.from_pandas_adjacency(dataf)
    return G


def load_data_to_dataframe(csv_path):
    dataf = pd.read_csv(csv_path, index_col=0)

    num_of_entries = len(dataf.index) - 2
    dataf = dataf.iloc[:num_of_entries - 1, :num_of_entries]

    return dataf


def load_transformed_data_to_dataframe(csv_path):
    dataf = pd.read_csv(csv_path, index_col=0)
    return dataf


def load_csv(path, key, value):
    fin_map = {}
    with open(path, mode='r', encoding='utf-8') as file:
        reader = csv.DictReader(file, delimiter=';')
        for row in reader:
            fin_map[int(row[key])] = row[value]

    return fin_map
