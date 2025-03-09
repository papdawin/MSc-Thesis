from .multilayer import get_louvain_by_year_multilayer
from .singelayer import get_louvain_by_year, get_corep_by_year, get_graph_by_year
from plotly.subplots import make_subplots
from ..localmemory_operations import load_csv
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.cluster.hierarchy import linkage, dendrogram
import numpy as np

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
    g = sns.clustermap(table, cmap='coolwarm', linewidths=0.5, linecolor='gray',method='ward',
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


import networkx as nx
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
import pickle


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
    years = list(range(1995, 2021))
    trade_graphs = {}
    for year in years:
        trade_graphs[year] = get_graph_by_year(year)

    features_per_year = {year: extract_features(graph) for year, graph in trade_graphs.items()}

    # with open("stuff.pkl", "wb") as f:
    #     pickle.dump(features_per_year, f)
    # with open("stuff.pkl", "rb") as f:
    #     features_per_year = pickle.load(f)

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

    plt.figure(figsize=(10, 5))
    plt.plot(years, error_values, marker="o", label="Reconstruction Error")
    plt.xlabel("Year")
    plt.ylabel("Reconstruction Error")
    plt.title("Autoencoder-Based Shock Detection")
    plt.legend()
    plt.show()