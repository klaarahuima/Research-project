The files include all the code used to generate samples of random graphs for the purpose of studying their embedding.

- graphfunctions.py : Defining all the functions I use building and visualising graphs, calculating the 2nd and 3rd eigenvectors, and plotting their embedding, calculatin inversions
      degree distributions, etc.
- simulategraphs.py: MAIN FILE. Use this to simulate graphs. You can define what kind of graph you want to generate and choose which properties to display (embedding, errors, eigenvectors, normal visualisation
        of graph, degree distributionm etc.)
- circulants.py : Defining functions used to generate ideal circulant graphs (exactly twice as many vertices in 1st quadrant.
- nearestneighbours.py : Defining functions used to generat graphs where vertices are sampled randomly around the unit circle and connected to their n nearest neighbours.
- plottingsinusoidals.py : Used to plot the sinusoidal functions matching the 2nd and 3rd eigenvectors. Used to calculate their parameters.
- removingedges.py : Simulating graphs as seen in the paper by Richter and Rocha. We create a model (perfectly circulant) graph and remove edges independently with constant probability p.
