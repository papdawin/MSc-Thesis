from torch_geometric.utils import to_networkx
from .multilayer import get_louvain_by_year_multilayer, get_graph_by_year_multilayer
from .singelayer import get_louvain_by_year, get_corep_by_year, get_graph_by_year, convert_to_pyg_data
from plotly.subplots import make_subplots
from ..localmemory_operations import load_csv
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.cluster.hierarchy import linkage, dendrogram
import numpy as np
import networkx as nx
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
import pickle
from node2vec import Node2Vec
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import os
from scipy.signal import find_peaks
from scipy.signal import argrelextrema
import random
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from torch_geometric.utils import to_networkx, to_dense_adj
from sklearn.cluster import KMeans

sector_map = load_csv("./Datasets/sector_LUT.csv", "\ufeffID", "Code")
country_map = load_csv("./Datasets/country_LUT.csv", "\ufeffID", "Code")


def eleventh_graph():
    lv = get_louvain_by_year(1995)
    for community in lv:
        print(len(community), community)

    entries = []
    for group_idx, data_set in enumerate(lv):
        # if len(data_set) > 1:
        for code in data_set:
            prefix, suffix = code.split('_', 1)
            entries.append({'Prefix': prefix, 'Suffix': suffix, 'Group': group_idx})

    df = pd.DataFrame(entries)

    table = df.pivot_table(index='Suffix', columns='Prefix', values='Group', aggfunc='first')

    linkage_matrix = linkage(table.fillna(0), method='ward')

    plt.figure(figsize=(12, 8))
    g = sns.clustermap(table, cmap='coolwarm', linewidths=0.5, linecolor='gray', method='ward',
                       figsize=(12, 8))
    plt.title('Clustering Heatmap of Country Codes and Categories')
    plt.show()


def twelfth_graph():
    lv = get_louvain_by_year_multilayer(1995)
    print(lv)


def thirteenth_graph():
    cp = get_corep_by_year(1998)

    entries = []
    for code in cp:
        prefix, suffix = code.split('_', 1)
        entries.append({'Prefix': prefix, 'Suffix': suffix, 'Group': 1})

    df = pd.DataFrame(entries)

    table = df.pivot_table(index='Suffix', columns='Prefix', values='Group', aggfunc='first')
    table = table.fillna(0)
    row_linkage = linkage(table, method='ward')
    col_linkage = linkage(table.T, method='ward')

    sns.clustermap(table, cmap='coolwarm', linewidths=0.5, linecolor='gray',
                   row_linkage=row_linkage, col_linkage=col_linkage, figsize=(12, 8))
    plt.show()


def jaccard_similarity(set1, set2):
    intersection = set1 & set2
    union = set1 | set2
    return len(intersection) / len(union) if len(union) > 0 else 0


def average_best_jaccard(communities1, communities2):
    similarities = []

    for group1 in communities1:
        best_similarity = max(jaccard_similarity(group1, group2) for group2 in communities2)
        similarities.append(best_similarity)

    for group2 in communities2:
        best_similarity = max(jaccard_similarity(group2, group1) for group1 in communities1)
        similarities.append(best_similarity)

    return sum(similarities) / len(similarities) if similarities else 0


def fourteenth_graph():
    years = list(range(1995, 2021))
    louvain_data = {year: get_louvain_by_year(year) for year in years}

    similarity_matrix = np.zeros((len(years), len(years)))

    for i, year1 in enumerate(years):
        print("processing year:", year1)
        lv1 = louvain_data[year1]
        for j in range(i + 1, len(years)):
            if i == j:
                similarity_matrix[i, j] = 1.0
            else:
                year2 = years[j]
                lv2 = louvain_data[year2]
                similarity = average_best_jaccard(lv1, lv2)
                similarity_matrix[i, j] = similarity
                similarity_matrix[j, i] = similarity

    similarity_matrix[similarity_matrix == 0.0] = 1.0
    plt.figure(figsize=(12, 8))
    sns.heatmap(similarity_matrix, xticklabels=years, yticklabels=years, cmap="coolwarm")
    plt.title("Yearly Louvain Jaccard Similarity")
    plt.xlabel("Year")
    plt.ylabel("Year")
    plt.show()


def fifteenth_graph():
    years = list(range(1995, 2021))
    louvain_data = {year: get_louvain_by_year(year) for year in years}

    jaccard_scores = []
    for i in range(len(years) - 1):
        y1, y2 = years[i], years[i + 1]
        jaccard = average_best_jaccard(louvain_data[y1], louvain_data[y2])
        jaccard_scores.append((y1, y2, jaccard))

    jaccard_values = [score[2] for score in jaccard_scores]
    mean_jaccard = np.mean(jaccard_values)
    std_jaccard = np.std(jaccard_values)
    threshold = mean_jaccard - 1.5 * std_jaccard  # Shock threshold (2σ below mean)

    shocks = [(y1, y2, j) for y1, y2, j in jaccard_scores if j < threshold]

    # Step 4: Visualizing results
    plt.figure(figsize=(10, 5))
    plt.plot([f"{y1}-{y2}" for y1, y2, _ in jaccard_scores], jaccard_values, marker='o', linestyle='-')
    plt.axhline(threshold, color='r', linestyle='dashed', label="Shock Threshold")
    plt.xlabel("Year Pair")
    plt.ylabel("Jaccard Similarity")
    plt.title("Jaccard Similarity of Trade Network Communities Over Time")
    plt.legend()
    plt.xticks(rotation=45)
    plt.show()

    # Print detected shocks
    print("\n🚨 Detected Shocks:")
    for y1, y2, j in shocks:
        print(f"Significant structural change detected: {y1} → {y2} (Jaccard: {j:.3f})")


class Autoencoder(nn.Module):
    def __init__(self, input_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 4),
            nn.ReLU(),
            nn.Linear(4, 1),
            # nn.ReLU(),
            # nn.Linear(8, 4)
        )
        self.decoder = nn.Sequential(
            # nn.Linear(4, 8),
            # nn.ReLU(),
            nn.Linear(1, 4),
            nn.ReLU(),
            nn.Linear(4, input_dim)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


def extract_features(graph):
    graph.remove_edges_from(nx.selfloop_edges(graph))

    degrees = dict(graph.degree())
    strengths = dict(graph.degree(weight='weight'))
    clustering = nx.clustering(graph)
    betweenness = nx.betweenness_centrality(graph)
    pagerank = nx.pagerank(graph, alpha=0.85)
    eigenvector = nx.eigenvector_centrality(graph, max_iter=1000)
    k_core = nx.core_number(graph)

    feature_matrix = np.array([
        [
            degrees[node], strengths[node], clustering[node], betweenness[node],
            pagerank[node], eigenvector[node], k_core[node]
        ]
        for node in graph.nodes()
    ])
    return feature_matrix


def sixteenth_graph():
    # seed = 52
    # random.seed(seed)
    # np.random.seed(seed)
    # torch.manual_seed(seed)
    years = list(range(1995, 2021))
    # trade_graphs = {}
    # for year in years:
    #     trade_graphs[year] = get_graph_by_year(year)
    #
    # features_per_year = {year: extract_features(graph) for year, graph in trade_graphs.items()}

    # with open("stuff.pkl", "wb") as f:
    #     pickle.dump(features_per_year, f)
    with open("person.pkl", "rb") as f:
        features_per_year = pickle.load(f)

    scaler = StandardScaler()
    all_features = np.vstack(list(features_per_year.values()))
    scaler.fit(all_features)
    for year in features_per_year:
        features_per_year[year] = scaler.transform(features_per_year[year])

    input_dim = 7
    autoencoder = Autoencoder(input_dim)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(autoencoder.parameters(), lr=0.001)

    X_train = torch.tensor(np.vstack(list(features_per_year.values())), dtype=torch.float32)

    epochs = 500
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = autoencoder(X_train)
        loss = criterion(outputs, X_train)
        loss.backward()
        optimizer.step()
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Loss = {loss.item()}")

    reconstruction_errors = {}
    with torch.no_grad():
        for year in years:
            X_test = torch.tensor(features_per_year[year], dtype=torch.float32)
            reconstructed = autoencoder(X_test)
            error = torch.mean((X_test - reconstructed) ** 2, axis=1).numpy()
            reconstruction_errors[year] = np.mean(error)

    error_values = np.array(list(reconstruction_errors.values()))
    mean_error = np.mean(error_values)
    std_error = np.std(error_values)
    threshold = mean_error + 1.5 * std_error  # Define anomaly threshold

    shocks = [year for year in years if reconstruction_errors[year] > threshold]
    peaks = argrelextrema(error_values, np.greater)[0]

    plt.figure(figsize=(10, 5))
    plt.plot(years, error_values, marker="o", label="Reconstruction Error")
    plt.scatter(np.array(years)[peaks], error_values[peaks], color="red", label="Local Maxima", zorder=3)
    # plt.axhline(y=threshold, color="gray", linestyle="--", label="Anomaly Threshold")
    plt.xlabel("Year")
    plt.ylabel("Reconstruction Error")
    plt.title("Autoencoder-Based Shock Detection with Local Maxima")
    plt.legend()
    plt.show()


def predict_commercial_block_changes(graph_dict, dimensions=64, walk_length=30, num_walks=200, p=1, q=1, num_clusters=5,
                                     embedding_dir="embeddings"):
    cluster_results = {}
    os.makedirs(embedding_dir, exist_ok=True)

    all_embeddings = []
    all_nodes = []
    year_markers = []

    for year, G in graph_dict.items():
        print(f"Processing year {year}...")
        embedding_file = os.path.join(embedding_dir, f"embeddings_{year}.npy")

        if os.path.exists(embedding_file):
            print(f"Loading existing embeddings for year {year}...")
            embeddings = np.load(embedding_file)
        else:
            print("Generating new embeddings...")
            node2vec = Node2Vec(G, dimensions=dimensions, walk_length=walk_length, num_walks=num_walks, p=p, q=q)
            model = node2vec.fit(window=10, min_count=1, batch_words=4)

            nodes = list(G.nodes)
            embeddings = np.array([model.wv[str(node)] for node in nodes])

            np.save(embedding_file, embeddings)
            print(f"Embeddings saved to {embedding_file}")

        nodes = list(G.nodes)
        filtered_indices = [i for i, node in enumerate(nodes) if str(node) in ["31", "59", "19", "66"]] # V4-ek

        cluster_results[year] = {nodes[i]: year for i in filtered_indices}

        all_embeddings.append(embeddings[filtered_indices])
        all_nodes.extend([nodes[i] for i in filtered_indices])
        year_markers.extend([year] * len(filtered_indices))

    all_embeddings = np.vstack(all_embeddings)

    tsne = TSNE(n_components=2, random_state=42, perplexity=5)
    reduced_embeddings = tsne.fit_transform(all_embeddings)

    plt.figure(figsize=(12, 10))

    scatter = plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c=year_markers, cmap='plasma', alpha=0.7)
    plt.colorbar(scatter, label='Year')

    year_markers = np.array(year_markers)
    for year in np.unique(year_markers):
        year_indices = np.where(year_markers == year)[0]
        year_points = reduced_embeddings[year_indices]

        for i in range(len(year_points)):
            for j in range(i + 1, len(year_points)):
                plt.plot([year_points[i, 0], year_points[j, 0]],
                         [year_points[i, 1], year_points[j, 1]],
                         color='gray', alpha=0.3, linewidth=0.5)

    for i, node in enumerate(all_nodes):
        translated_name = country_map[int(str(node))]
        plt.annotate(translated_name, (reduced_embeddings[i, 0], reduced_embeddings[i, 1]), fontsize=8, alpha=0.7)

    plt.title('Commercial Block Changes')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.show()

    return cluster_results


def seventeenth_graph():
    years = list(range(2002, 2017, 3))
    trade_graphs = {year: get_graph_by_year_multilayer(year) for year in years}
    predict_commercial_block_changes(trade_graphs)


def eighteenth_graph():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    years = list(range(1995, 2021))
    trade_graphs = {}
    #
    # # Load and preprocess graphs
    # for year in years:
    #     graph = get_graph_by_year(year)
    #     pyg_data = convert_to_pyg_data(graph, country_map, sector_map)
    #     trade_graphs[year] = pyg_data.to(device)
    #
    # # Train model
    # input_dim = trade_graphs[years[0]].x.shape[1]
    # model = train_model(trade_graphs, years, device, input_dim)

    model_path = "gnn_model.pth"
    # torch.save(model.state_dict(), model_path)

    # input_dim = trade_graphs[years[0]].x.shape[1]
    model = GNN(79, 46).to(device)
    model.load_state_dict(torch.load(model_path))
    print("Loaded", model_path)

    for year in years:
        graph = get_graph_by_year(year)

        pyg_data = convert_to_pyg_data(graph, country_map, sector_map)
        trade_graphs[year] = pyg_data.to(device)

    labels = ["China", "Japan", "Korea", "Taiwan", "United States"]
    # First part country, second sector eg: 1*1 ARG_A01_02
    selected_indices = [13 * 17, 39 * 17, 42 * 17, 72 * 17, 74 * 17]
    visualize_embeddings(model, trade_graphs, years, device, labels, selected_indices)


class GNN(torch.nn.Module):
    def __init__(self, num_countries, num_sectors, embed_dim=16, hidden_dim=64):
        super().__init__()
        # Embedding layers for categorical features
        self.country_embed = torch.nn.Embedding(num_countries, embed_dim)
        self.sector_embed = torch.nn.Embedding(num_sectors, embed_dim)

        # GCN layers (input_dim = embed_dim * 2)
        self.conv1 = GCNConv(embed_dim * 2, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, embed_dim)

    def forward(self, data):
        # Get embeddings for country and sector indices
        country_emb = self.country_embed(data.x[:, 0])  # Country indices
        sector_emb = self.sector_embed(data.x[:, 1])  # Sector indices

        # Concatenate embeddings
        x = torch.cat([country_emb, sector_emb], dim=1)

        # GCN operations
        x = self.conv1(x, data.edge_index, data.edge_attr)
        x = F.relu(x)
        x = self.conv2(x, data.edge_index, data.edge_attr)

        return x


def train_model(trade_graphs, years, device, input_dim, hidden_dim=64, embed_dim=32, num_epochs=100):
    model = GNN(79, 46).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for year in years:
            data = trade_graphs[year]
            optimizer.zero_grad()
            embeddings = model(data)
            src, dst = data.edge_index
            pred_weights = (embeddings[src] * embeddings[dst]).sum(dim=1)
            true_weights = data.edge_attr
            loss = F.mse_loss(pred_weights, true_weights)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch}, Loss: {total_loss / len(years)}")
    return model


def visualize_embeddings(model, trade_graphs, years, device, labels, selected_indices):
    embeddings_list = []
    year_labels = []

    for year in years:
        data = trade_graphs[year].to(device)
        with torch.no_grad():
            embeddings = model(data).cpu().numpy()

        embeddings = embeddings[selected_indices]
        embeddings_list.append(embeddings)
        year_labels.extend([year] * embeddings.shape[0])

    X = np.vstack(embeddings_list)
    years_array = np.array(year_labels)

    # Apply t-SNE
    tsne = TSNE(n_components=2, perplexity=3, random_state=42)
    tsne_results = tsne.fit_transform(X)

    # Plot
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=years_array, cmap='jet', alpha=0.6)

    # year_markers = np.array(years_array)
    for year in np.unique(years_array):
        year_indices = np.where(years_array == year)[0]
        year_points = tsne_results[year_indices]

        for i in range(len(year_points)):
            for j in range(i + 1, len(year_points)):
                plt.plot([year_points[i, 0], year_points[j, 0]],
                         [year_points[i, 1], year_points[j, 1]],
                         color='gray', alpha=0.3, linewidth=0.5)

    # Annotate points with labels
    for i, label in enumerate(labels*len(years)):
        plt.text(tsne_results[i, 0], tsne_results[i, 1], label, fontsize=12, ha='right', va='bottom')

    plt.colorbar(scatter, label='country_sector')
    plt.title("t-SNE Visualization of Node Embeddings (1995-2019)")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.show()