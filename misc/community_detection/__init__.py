from .multilayer import get_louvain_by_year_multilayer
from .singelayer import get_louvain_by_year, get_corep_by_year
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