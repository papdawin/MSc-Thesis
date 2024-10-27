import pandas as pd
import networkx as nx
import plotly.express as px

dataf = pd.read_csv("../../Datasets/OECD_Original/ICIO2023_2020.csv", index_col=0)

num_of_entries = len(dataf.index) - 2
dataf = dataf.iloc[:num_of_entries-1, :num_of_entries]

G = nx.from_pandas_adjacency(dataf)

pageranks = nx.pagerank(G)
top_n_pagerank = dict(sorted(pageranks.items(), key=lambda x: x[1], reverse=True)[:100])

labels=[]
parents=[]
values=[]
for el in top_n_pagerank:
    labels.append(el[4:])
    parents.append(el[:3])
    values.append(top_n_pagerank[el] * 1000)

df = pd.DataFrame({"parents": parents,
                   "labels": labels,
                   "values": values})

print(df.head(10))

fig = px.sunburst(df, path=['parents', 'labels'], values='values')
fig.show()
