import pandas as pd
import networkx as nx
import plotly.express as px
import matplotlib.pyplot as plt

from misc import *

uri = "neo4j://localhost"
username = "neo4j"
password = "Admin123"

# # Save graph to db, and load it back
# G = load_database_to_graph("./Datasets/OECD_Original/ICIO2023_1995.csv")
# driver = create_db_connection(uri, username, password)
# save_graph_to_neo4j(G, driver, year="1995", tag="original")
# G_loaded = load_graph_from_neo4j(driver)


df = load_transformed_data_to_dataframe("./Datasets/OECD_Transformed/OECD_1995.csv")
# calculate_multilayer_network_indicators(df, 'sector')
calculate_multilayer_network_indicators(df, 'country')
# df = load_data_to_dataframe("./Datasets/OECD_Original/ICIO2023_1995.csv")
# calculate_singe_layer_indicators(df)








# driver.close()
