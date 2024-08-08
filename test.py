import numpy as np
import pandas as pd
import networkx as nx
import plotly.graph_objects as go
from matplotlib import pyplot as plt
from networkx.drawing.nx_agraph import graphviz_layout

from layered_network_graph import LayeredNetworkGraph

dataf = pd.read_csv("Datasets/OECD_Original/ICIO2023_2020.csv", index_col=0)
# data2 = pd.read_csv("Datasets/OECD_Original/ICIO2023_1996.csv", index_col=0)

# only use the relevant data (REDUCED DATASET)
num_of_entries = len(dataf.index) - 2
# dataf = dataf.iloc[:num_of_entries-1, :num_of_entries]
# data2 = data2.iloc[:num_of_entries-1, :num_of_entries]
dataf = dataf.iloc[:1000, :1000]
# data2 = data2.iloc[:100, :100]


G = nx.from_pandas_adjacency(dataf)


# G2 = nx.from_pandas_adjacency(data2)
# nx.draw_networkx(G)


edge_to_rem = [(a, b) for a, b, attrs in G.edges(data=True) if attrs["weight"] <= 5000]
G.remove_edges_from(edge_to_rem)

G.remove_edges_from(list(nx.selfloop_edges(G)))
G.remove_nodes_from(list(nx.isolates(G)))

# pos = nx.spring_layout(G) # Define the layout for node positioning
weights = nx.get_edge_attributes(G,'weight').values()

a = []

for weight in weights:
    a.append(weight/10000)
# print(type(weights))
# print(weights)

# amin, amax = min(weights), max(weights)
# nw = [0*len(weights)+10]
# for i, val in enumerate(weights):
#     nw[i] = (val-amin) / (amax-amin)
UG = G.to_undirected()
for c in nx.connected_components(UG):
    subgraph = G.subgraph(c)
    if subgraph.number_of_nodes() > 2:
        # nx.draw(subgraph,
        #         pos=nx.spring_layout(subgraph),
        #         with_labels=True,
        #         # width=a,
        #         node_size=300,
        #         node_color='orange',
        #         font_size=10,
        #         font_color='blue')
        # plt.show()

        # print(len(nx.connected_components(UG)))
        # print(nx.connected_components(UG))

        pos = nx.spring_layout(subgraph)
        node_x = [x for x, y in pos.values()]
        node_y = [-y for x, y in pos.values()]
        edge_x = [x for n0, n1 in subgraph.edges for x in (pos[n0][0], pos[n1][0], None)]
        edge_y = [y for n0, n1 in subgraph.edges for y in (-pos[n0][1], -pos[n1][1], None)]

        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=0.5, color='#888'),
            hoverinfo='none',
            mode='lines')

        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers',
            hoverinfo='text',
            marker=dict(
                showscale=True,
                colorscale='YlGnBu',
                reversescale=True,
                color=[],
                size=10,
                colorbar=dict(
                    thickness=15,
                    title='Node Connections',
                    xanchor='left',
                    titleside='right'
                ),
                line_width=2))

        node_adjacencies = []
        for adjacencies in subgraph.adjacency():
            node_adjacencies.append(len(adjacencies[1]))

        node_text = []

        for idx, node in enumerate(subgraph.nodes(data=True)):
            node_text.append(node[0])
            # print(node[0])

        node_trace.marker.color = node_adjacencies
        node_trace.text = node_text

        fig = go.Figure(data=[edge_trace, node_trace],
                        layout=go.Layout(
                            showlegend=False,
                            hovermode='closest',
                            margin=dict(b=20,l=5,r=5,t=40),
                            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                        )
        fig.show()














# print(G.number_of_edges(), G.number_of_nodes()) # E: 3982729   N: 3465

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# LayeredNetworkGraph([G], ax=ax, layout=nx.fruchterman_reingold_layout)
# ax.set_axis_off()
# plt.show()
