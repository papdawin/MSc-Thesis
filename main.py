import pandas as pd
import networkx as nx
import plotly.express as px

from misc import *

dataf = pd.read_csv("./Datasets/OECD_Original/ICIO2023_2020.csv", index_col=0)

uri = "neo4j://localhost:7687"
username = "neo4j"
password = "Admin123"

G = load_database_to_graph("./Datasets/OECD_Original/ICIO2023_1995.csv")

driver = create_db_connection(uri, username, password)
print("saving..")
save_graph_to_neo4j(G, driver, year="1995", tag="original")
print("saved")
G_loaded = load_graph_from_neo4j(driver)

print(G_loaded.nodes(data=True))
print(G_loaded.edges(data=True))

driver.close()
