import pandas as pd
import networkx as nx
import plotly.express as px
import matplotlib.pyplot as plt

from misc import *
from misc.centrality_measures.multilayer import get_network_country_yearly

uri = "neo4j://localhost"
username = "neo4j"
password = "Admin123"

# # Save graph to db, and load it back
# G = load_database_to_graph("./Datasets/OECD_Original/ICIO2023_1995.csv")
# driver = create_db_connection(uri, username, password)
# save_graph_to_neo4j(G, driver, year="1995", tag="original")
# G_loaded = load_graph_from_neo4j(driver)


# introductory_graph()
# degree_centrality()
# betweenness_centrality()
# harmonic_centrality()
# page_rank()
# edge_betweenness_centrality() # not implemented
density()
# original_df = load_data_to_dataframe("./Datasets/OECD_Original/ICIO2023_2020.csv")
# original_df2 = load_data_to_dataframe("./Datasets/OECD_Original/ICIO2023_2020.csv")
# calculate_singe_layer_indicators(original_df)





# driver.close()
