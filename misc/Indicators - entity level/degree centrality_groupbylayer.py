import pandas as pd
import networkx as nx
import plotly.express as px


dataf = pd.read_csv("../../Datasets/OECD_Original/ICIO2023_2020.csv", index_col=0)

num_of_entries = len(dataf.index) - 2
# num_of_entries = 200
dataf = dataf.iloc[:num_of_entries-1, :num_of_entries]

G = nx.from_pandas_adjacency(dataf)

nodes_grouped_by_country = {}
nodes_grouped_by_sector = {}
for (p, d) in G.nodes.data():
    # print(p, d)
    country, sector = p.split('_', 1)
    nodes_grouped_by_country.setdefault(country, []).append(p)
    nodes_grouped_by_sector.setdefault(sector, []).append(p)

print(nodes_grouped_by_country)

for country in nodes_grouped_by_country.keys():
    nodes = sorted(nodes_grouped_by_country[country])
    for node in nodes[1:]:
        G = nx.contracted_nodes(G, nodes[0], node, self_loops=False)
    G = nx.relabel_nodes(G, {nodes[0]: country})
    # G.add_node(country)
    # for country2 in nodes_grouped_by_country[country]:
    #     nx.contracted_nodes(G, country2, country, self_loops=False)
# merge_nodes()

print(G.nodes)
print(G.get_edge_data('ARG','AUS'))
# print(dataf)


DC = nx.degree_centrality(G)
# top_n_DC = dict(sorted(DC.items(), key=lambda x: x[1], reverse=True)[:100])

labels=[]
parents=[]
values=[]
for el in DC:
    labels.append(el[4:])
    parents.append(el[:3])
    values.append(DC[el])


df = pd.DataFrame({"parents": parents,
                   "labels": labels,
                   "dcvalues": values})

# print(df.sort_values(by=['dcvalues']))

# print(df['dcvalues'].where(df['parents']=='BRN').mean())
# print(df['dcvalues'].where(df['parents']=='SEN').mean())
# print(df['dcvalues'].where(df['parents']=='LAO').mean())
# print(df['dcvalues'].where(df['parents']=='EST').mean())

fig = px.scatter(df, y="parents", x="dcvalues", color="labels", title="Fokszám központiság 1995-ben",
                 labels={
                     "parents": "Országok",
                     "dcvalues": "Fokszám központiság",
                     "labels": "Szektorok"
                 })
fig.update_traces(marker_size=10)
fig.show()

