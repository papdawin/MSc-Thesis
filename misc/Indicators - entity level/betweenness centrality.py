import pandas as pd
import networkx as nx
import plotly.express as px

dataf = pd.read_csv("../../Datasets/OECD_Original/ICIO2023_2020.csv", index_col=0)

num_of_entries = len(dataf.index) - 2
dataf = dataf.iloc[:num_of_entries-1, :num_of_entries]

G = nx.from_pandas_adjacency(dataf)

BC = nx.betweenness_centrality(G)
# top_n_DC = dict(sorted(DC.items(), key=lambda x: x[1], reverse=True)[:100])

labels=[]
parents=[]
values=[]
for el in BC:
    labels.append(el[4:])
    parents.append(el[:3])
    values.append(BC[el])


df = pd.DataFrame({"parents": parents,
                   "labels": labels,
                   "bcvalues": values})

print(df.sort_values(by=['bcvalues']))

# print(df['dcvalues'].where(df['parents']=='BRN').mean())
# print(df['dcvalues'].where(df['parents']=='SEN').mean())
# print(df['dcvalues'].where(df['parents']=='LAO').mean())
# print(df['dcvalues'].where(df['parents']=='EST').mean())

# fig = px.scatter(df, y="parents", x="dcvalues", color="labels", title="Fokszám központiság 1995-ben",
#                  labels={
#                      "parents": "Országok",
#                      "dcvalues": "Fokszám központiság",
#                      "labels": "Szektorok"
#                  })
# fig.update_traces(marker_size=10)
# fig.show()

