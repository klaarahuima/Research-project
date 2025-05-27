**Random Graph Research Project**, Spring 2025

The files include all the Python code used to generate samples of random graphs for the purpose of studying their embedding.

The files are split into 2 types: files you run and files I've defined functions in that I import into the files that I run.

Function defintions:
- graphfunctions.py : Defining all the functions I use building and visualising graphs, calculating the 2nd and 3rd eigenvectors, and plotting their embedding, calculating inversions
      degree distributions, etc.
- circulants.py : Defining functions used to generate ideal circulant graphs (exactly twice as many vertices in 1st quadrant) and split the eigenvectors based on the quadrant in which the vertex is (into 2 parts: Q1 and Q2-Q4).
- nearestneighbours.py : Defining functions used to generate graphs where vertices are sampled randomly around the unit circle and connected to their n nearest neighbours.
- removingedges.py : Simulating graphs as seen in the paper by Richter and Rocha. We create a model (perfectly circulant) graph and remove edges independently with constant probability p. I don't actually run this file, but use it to import the functions I defined in it.

Files I run:
- simulategraphs.py: MAIN FILE. Use this to simulate graphs. You can define what kind of graph you want to generate and choose which properties to display (embedding, errors, eigenvectors, normal visualisation
        of graph, degree distribution etc.)
- curvefitting.py: Used to plot an eigenvector and overlay it with an optimal sinusoidal fit to the data. (Calculates the sinusoidal fit from the vector data and visualises it.)
- plottingsinusoidals.py : Used to plot the sinusoidal functions matching the 2nd and 3rd eigenvectors. Used to calculate their parameters and visualise the relationship between the sinusoidal parameters of the eigenvectors and the order of the graph. Useful for both ideal and random graphs.
- simulatinginversions.py : Used to visualise the relationship between number of inversions of a random graph and its order.

The rest of the files were not used to study the results we deemed significant, but rather used at the beginning of the research project to study random behaviour of a larger variety of random graphs.

I ran this project on locally on my PC with PyCharm software. I created a folder with all the following files. If all the files are imported properly, running **simulategraphs.py** will do all the work for generating graphs for you. If you want to study the sinusoidal patterns, you will have to run **plottingsinusoidals.py**.

Another alternative is to run the project on Jupyter notebooks. This is especially useful as this can be done on the Dal clusters. In that case, you import all the files as they are, and create
an empty notebook in which you can then generate graphs like I have done in simulategraphs.py. Do not forget to import all the packages I did at the top of simulategraphs.py. I had some problems with including some packages like networkx in the Jupyter notebook environment on the Dal clusters. I hope you don't have the same issues.

I tried to comment the files thoroughly. Do email me if something is very unclear. AI is also probably quite successful in trying to explain what I have tried to do in my code, as well as helping you modify it to your needs. Never delete code though, you might regret it (I know I did).

I don't know how long my Dal email is up and running. You can always reach me at klaara.huima@gmail.com. 

Good luck!

Last updated: 27.5.2025

