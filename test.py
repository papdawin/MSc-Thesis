import networkx as nx
import plotly.graph_objects as go
import numpy as np

# Create a multi-layer graph with NetworkX
G = nx.Graph()
layers = 3  # number of layers
nodes_per_layer = 5
for layer in range(layers):
    for i in range(nodes_per_layer):
        G.add_node((layer, i), layer=layer)
    if layer > 0:
        for i in range(nodes_per_layer):
            G.add_edge((layer - 1, i), (layer, i))

# Generate node positions in 3D space
pos = {n: (n[1] * np.cos(layer * np.pi / 3),
           n[1] * np.sin(layer * np.pi / 3),
           n[0]) for n, layer in nx.get_node_attributes(G, 'layer').items()}

# Plot with Plotly
edge_x, edge_y, edge_z = [], [], []
for edge in G.edges():
    x0, y0, z0 = pos[edge[0]]
    x1, y1, z1 = pos[edge[1]]
    edge_x += [x0, x1, None]
    edge_y += [y0, y1, None]
    edge_z += [z0, z1, None]

edge_trace = go.Scatter3d(x=edge_x, y=edge_y, z=edge_z, mode='lines')
node_x, node_y, node_z = zip(*pos.values())
node_trace = go.Scatter3d(x=node_x, y=node_y, z=node_z, mode='markers')

fig = go.Figure(data=[edge_trace, node_trace])
fig.show()
