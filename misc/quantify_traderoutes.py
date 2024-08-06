import numpy as np
import pandas as pd
import networkx as nx
import plotly.graph_objects as go
from matplotlib import pyplot as plt

for i in range(1995, 2020):
    dataf = pd.read_csv(f"../Datasets/OECD_Transformed/OECD_{i}.csv", index_col=0)

    opt = dataf.groupby("Exporter_country")["Value (million USD)"].sum().to_frame().T
    # opt2 = opt.transpose()
    # print(opt)
    opt.to_csv("../Datasets/export_between_countries.csv", mode='a', index=False, header=False)