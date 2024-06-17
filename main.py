import numpy as np
import pandas as pd
import networkx as nx
import plotly.graph_objects as go
from matplotlib import pyplot as plt

country_codes = pd.read_csv("./Datasets/BACI_HS92_V202401/country_codes_V202401.csv")
product_codes = pd.read_csv("./Datasets/BACI_HS92_V202401/product_codes_HS92_V202401.csv")
data_2022 = pd.read_csv("./Datasets/BACI_HS92_V202401/BACI_HS92_Y2022_V202401.csv")

print(data_2022.head())
print(data_2022.info())

G = nx.from_pandas_edgelist(data_2022, source='i', target='j', edge_attr=['k', 'v', 'q'], create_using=nx.DiGraph())

# country_codes.set_index("country_code", drop=True, inplace=True)
country_codes.drop(["country_name","country_iso2"], axis="columns", inplace=True)

country_dict = dict(country_codes.values)

G = nx.relabel_nodes(G, country_dict)

pos = nx.spring_layout(G) # Define the layout for node positioning
# nx.draw(G, pos, with_labels=True, node_size=300, node_color='skyblue', font_size=10, font_color='black')
# plt.show()

pos = nx.spring_layout(G)
node_x = [x for x, y in pos.values()]
node_y = [-y for x, y in pos.values()]

edge_x = [x for n0, n1 in G.edges for x in (pos[n0][0], pos[n1][0], None)]
edge_y = [y for n0, n1 in G.edges for y in (-pos[n0][1], -pos[n1][1], None)]

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
        # colorscale options
        #'Greys' | 'YlGnBu' | 'Greens' | 'YlOrRd' | 'Bluered' | 'RdBu' |
        #'Reds' | 'Blues' | 'Picnic' | 'Rainbow' | 'Portland' | 'Jet' |
        #'Hot' | 'Blackbody' | 'Earth' | 'Electric' | 'Viridis' |
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
node_text = []
for node, adjacencies in enumerate(G.adjacency()):
    node_adjacencies.append(len(adjacencies[1]))
    node_text.append('# of connections: '+str(len(adjacencies[1])))

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