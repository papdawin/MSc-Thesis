## Time series analysis of the characteristics of multilayer commercial and collaboration networks, detection of anomalies and shocks
#### Data Science MSc Thesis
###### Veszprém, 2025.05.03

[![en](https://img.shields.io/badge/version-English-blue.svg)](https://github.com/papdawin/MSc-Thesis/blob/master/README.md)
[![hu](https://img.shields.io/badge/version-Hungarian-brown.svg)](https://github.com/papdawin/MSc-Thesis/blob/master/README.hu.md)

In my diploma thesis, I focused primarily on the analysis of global trade networks, with
particular attention to centrality measures, autoencoder models, graph neural networks
(GNN), embedding techniques, and with their help, the analysis of time series changes
and anomaly detection.
The basis of the analysis was the OECD Inter-Country Input-Output (ICIO) database,
which contains trade data for 76 countries between 1995 and 2020, divided into sectors
according to the ISIC Rev. 4 standard. I first converted these data from an adjacency
matrix to an edge list format in order to create aggregated and multilayer networks. I
created NetworkX graphs from the completed networks, and then converted them into
PyGData objects in the case of Graph-Neural solutions.
I examined several centrality measures on the created graphs, analyzed the results
obtained in this way, then visualized and documented the obtained results.
For the time series analysis of the data, I created an autoencoder-based model and an
embedding model. The autoencoder-based solution enabled anomaly detection, i.e. the
identification of unusual or outlier trade patterns in the network that may indicate
economic shocks.
Using the embedding-based solution, I created a time series vector representation, with
which I modeled the trade network transformations that occurred over the years. I then
implemented these approaches using the Graph Neural Network, using the GNN
embedding formed this way to learn the complex connection structure of graphs, which
were able to explore the nonlinear relationships between nodes and edges, thus further
improving the detection of anomalies and the understanding of the network dynamics in
the data set.
Overall, during the research, I analyzed centrality indicators, using embedding to detect
time series changes, and using an autoencoder to detect anomalies in the trade network. I
then developed a Graph Neural alternative to these, which I evaluated and compared with
the traditional method. The results obtained allowed for a deeper understanding of
network dynamics and a more accurate identification of economic anomalies.
