from misc import *


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
# edge_betweenness_centrality()
# density()
# louvain()
# core_periphery()
# community_change_detection()
# shock_analysis()
# autoencoder_shock_detection()
block_change_prediction()



# driver.close()
