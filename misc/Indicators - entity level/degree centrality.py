import pandas as pd
import networkx as nx
import plotly.express as px
import plotly.graph_objects as go

def fnc(year):
    dataf = pd.read_csv(f"../../Datasets/OECD_Original/ICIO2023_{year}.csv", index_col=0)

    num_of_entries = len(dataf.index) - 2
    dataf = dataf.iloc[:num_of_entries-1, :num_of_entries]

    G = nx.from_pandas_adjacency(dataf)

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

    df = df[['parents', 'dcvalues']].groupby('parents').mean()
    # df = df.sort_values(by=['dcvalues'])
    df = df.reset_index()
    return df

df_fin = fnc(1995)
df_fin['year'] = 1995

for i in range(2):
    df = fnc(i+1995)
    df['year'] = i+1995
    df_fin = pd.concat([df_fin, df])

# df1 = fnc(1995)
# df1['year'] = 1995
# df2 = fnc(2020)
# df2['year'] = 2020
# df = pd.concat([df1, df2])
# df = df.sort_values(by=['dcvalues'])
# fig = px.line(df, x="parents", y="dcvalues", color='year')
# fig = go.Figure()
# fig.add_trace(go.Scatter(x=df['parents'], y=df['dcvalues'], color=df['year'],
#                     mode='markers', name='markers'))
#
# fig.show()

fig = px.scatter(df_fin, y="dcvalues", x="parents", color="year", title="Fokszám központiság alakulása 1995-től 2020-ig (ország szerint csoportosítva)",
                 labels={
                     "parents": "Országok",
                     "dcvalues": "Fokszám központiság",
                     "labels": "Évek"
                 })
fig.update_traces(marker_size=10)
fig.show()

