from .database_operations import *
from .localmemory_operations import *
from .centrality_measures import *
from .community_detection import *


def degree_centrality():
    first_graph()
    second_graph()


def betweenness_centrality():
    third_graph()


def harmonic_centrality():
    fourth_graph()


def page_rank():
    fifth_graph()


def edge_betweenness_centrality():
    sixth_graph()


def density():
    seventh_graph()
    eighth_graph(intralayer=True)
    ninth_graph(intralayer=True)
    eighth_graph(intralayer=False)
    ninth_graph(intralayer=False)


def homophily():
    tenth_graph()


def louvain():
    eleventh_graph()
    twelfth_graph()


def core_periphery():
    thirteenth_graph()


def community_change_detection():
    fourteenth_graph()


def shock_analysis():
    fifteenth_graph()


def autoencoder_shock_detection():
    sixteenth_graph()


def block_change_prediction():
    seventeenth_graph()