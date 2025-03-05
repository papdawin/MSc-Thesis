import networkx as nx
import pymnet
from misc.centrality_measures.multilayer import remove_non_weighed_edges

def visualize_dataframe(df, sector_map, country_map):
    dataframes = [sub_df for _, sub_df in df.groupby('Exporter_country')]
    graphs = []
    for sub_df in dataframes:
        sub_df = sub_df.rename(columns={'Value (million USD)': 'weight'})
        graph = nx.from_pandas_edgelist(sub_df, source='Exporter_country', target='Importer_country',
                                        edge_attr='weight', create_using=nx.DiGraph())
        graph = remove_non_weighed_edges(graph)
        graphs.append(graph.to_undirected())

    net_social = pymnet.MultiplexNetwork(couplings="categorical", fullyInterconnected=False)

    for entry in dataframes:
        for idx, row in entry.iterrows():
            if not row['Exporter_country'] == row['Importer_country']:
                net_social[(country_map[int(row['Exporter_country'])], country_map[int(row['Importer_country'])],
                            sector_map[int(row['Exporter_sector'])])] = row['Value (million USD)']

    fig_social = pymnet.draw(net_social, layergap=1, figsize=(10, 10), defaultNodeSize=2, layout="circular",
                             layerPadding=0.2, defaultLayerLabelLoc=(0.9, 0.9))
    fig_social.show()
    # pymnet.webplot(net_social, outputfile="asd.html")