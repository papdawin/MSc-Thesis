from .calculate_indicators import *
from .operations import *
from ...localmemory_operations import load_transformed_data_to_dataframe
import torch
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx


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


def convert_to_pyg_data(graph, country_dict, sector_dict):
    # Process nodes to create features and node-to-index mapping
    country_dict = {v: k for k, v in country_dict.items()}
    sector_dict = {v: k for k, v in sector_dict.items()}
    nodes = list(graph.nodes())
    node_to_idx = {node: idx for idx, node in enumerate(nodes)}

    # Extract country/sector indices using original dictionaries
    x = []
    for node in nodes:
        country_code, sector_code = node.split("_", 1)
        country_idx = country_dict[country_code]  # Directly use original dict
        sector_idx = sector_dict[sector_code]  # No reversal needed
        x.append([country_idx, sector_idx])

    # Convert to tensors
    x = torch.tensor(x, dtype=torch.long)  # Shape: [num_nodes, 2]

    # Process edges and weights
    edge_index, edge_attr = [], []
    for u, v, data in graph.edges(data=True):
        src = node_to_idx[u]
        tgt = node_to_idx[v]
        edge_index.append([src, tgt])
        edge_attr.append(data["weight"])

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous() if edge_index else torch.empty((2, 0),
                                                                                                            dtype=torch.long)
    edge_attr = torch.tensor(edge_attr, dtype=torch.float).flatten()  # 1D tensor

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

# def convert_to_pyg_data(nx_graph, countrymap, sectormap):
#     # Convert to undirected graph
#     graph = nx_graph.to_undirected()
#
#     # Remove self-loops
#     graph.remove_edges_from(nx.selfloop_edges(graph))
#     node_mapping = {node: i for i, node in enumerate(graph.nodes())}
#
#     # Sum duplicate edges
#     new_graph = nx.Graph()
#     for u, v, data in graph.edges(data=True):
#         weight = data.get("weight", 0)  # Default weight is 1.0 if not present
#         u, v = node_mapping[u], node_mapping[v]
#         if new_graph.has_edge(u, v):
#             new_graph[u][v]["weight"] += weight
#         else:
#             new_graph.add_edge(u, v, weight=weight)
#
#     # Extract node features
#     node_features = []
#     for node in nx_graph.nodes():
#         # features = list(nx_graph.nodes[node].values())  # Convert dict to list
#         country_name, sector_name = node.split('_', 1)
#
#         reverse_countrymap = {v: k for k, v in countrymap.items()}
#         reverse_sectormap = {v: k for k, v in sectormap.items()}
#
#         country_code = reverse_countrymap.get(country_name, -1)
#         sector_code = reverse_sectormap.get(sector_name, -1)
#
#         # Concatenate features
#
#         node_features.append([country_code, sector_code])
#     node_features = torch.tensor(node_features, dtype=torch.float)
#
#     # Convert to PyG Data object
#     pyg_data = from_networkx(nx_graph)
#
#     # Add node features to the Data object
#     pyg_data.x = node_features
#
#     pyg_data.edge_index = torch.tensor(list(new_graph.edges), dtype=torch.long).t().contiguous()
#
#     pyg_data.edge_weight = torch.tensor([data["weight"] for _, _, data in new_graph.edges(data=True)], dtype=torch.float)
#
#     return pyg_data
