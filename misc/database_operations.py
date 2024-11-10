from neo4j import GraphDatabase
import networkx as nx


def create_db_connection(uri, username, password):
    driver = GraphDatabase.driver(uri, auth=(username, password))
    return driver


def save_graph_to_neo4j(G, driver, year, tag):
    with driver.session() as session:
        for node_id, node_data in G.nodes(data=True):
            # Update node_data with year and tag properties
            node_data["year"] = year
            node_data["tag"] = tag

            session.run(
                "CREATE (n:Node {id: $id, year: $year, tag: $tag})",
                id=node_id,
                year=node_data["year"],
                tag=node_data["tag"],)

        for edge in G.edges(data=True):
            source, target, edge_data = edge
            session.run(
                "MATCH (a:Node {id: $source}), (b:Node {id: $target}) CREATE (a)-[:RELATIONSHIP {weight: $weight}]->(b)",
                source=source,
                target=target,
                weight=edge_data["weight"])


def load_graph_from_neo4j(driver):
    with driver.session() as session:
        result = session.run("MATCH (n) RETURN n")
        nodes = [record["n"] for record in result]

        G_loaded = nx.Graph()
        G_loaded.add_nodes_from([(node["id"], node["properties"]) for node in nodes])

        result = session.run("MATCH (a)-[r]->(b) RETURN a, r, b")
        edges = [(record["a"]["id"], record["b"]["id"], record["r"]["properties"]) for record in result]
        G_loaded.add_edges_from(edges)

    return G_loaded
