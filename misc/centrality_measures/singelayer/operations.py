import networkx as nx
import pandas as pd

def create_graph_from_dataframe(df):
    num_of_entries = len(df.index) - 2
    dataf = df.iloc[:num_of_entries - 1, :num_of_entries]

    graph = nx.from_pandas_adjacency(dataf)
    return graph


def transform_to_df_and_order(centrality_values):
    labels, parents, values = [], [], []
    for el in centrality_values:
        labels.append(el[4:])
        parents.append(el[:3])
        values.append(centrality_values[el])

    df = pd.DataFrame({"parents": parents,
                       "labels": labels,
                       "centrality_values": values})

    return df.sort_values(by=['centrality_values'])
