The files include all the code used to generate samples of random graphs for the purpose of studying their embedding.

- graphfunctions.py : Defining all the functions I use building and visualising graphs, calculating the 2nd and 3rd eigenvectors, and plotting their embedding.
- thresholdgraph.py : Simulating graphs by sampling vertices uniformly around the unit circle and connecting all vertices that are less than a given threshold distance apart from eachother.
- modifiedthreshold.py : Simulating graphs by sampling vertices non-uniformly around the unit circle and connecting them with probability that depends on their distance on the circle.
- removingedges.py : Simulating graphs as seen in the paper by Richter and Rocha. We create a model (perfectly circulant) graph and remove edges independently with constant probability p.
- wsmodel.py : Simulating graphs produced by the networkx implementation of the Watts-Strogatz model.
