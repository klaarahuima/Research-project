import networkx as nx

def create_cycle_graph(n):
    """
    Create a cycle graph with n vertices.
    """
    return nx.cycle_graph(n)

def duplicate_vertices(G, vertices_to_duplicate):
    """
    Duplicate the specified vertices in the graph.

    Parameters:
    - G: networkx graph
    - vertices_to_duplicate: list of vertices to duplicate

    Returns:
    - Updated graph with duplicated vertices
    """
    new_graph = G.copy()
    for v in vertices_to_duplicate:
        if v not in new_graph:
            print(f"Vertex {v} not found")
            continue

        # Generate a new unique name for the duplicated vertex
        new_v = f"{v}_copy"

        # Add the new vertex and connect it to the same neighbors
        new_graph.add_node(new_v)
        for neighbor in new_graph.neighbors(v):
            new_graph.add_edge(new_v, neighbor)

        new_graph.add_edge(v, new_v)

    return new_graph

def create_graph(order, duplication):
    cycle_graph = create_cycle_graph(order)

    # Specify which vertices to duplicate
    vertices_to_duplicate = [i for i in range(0, duplication)]

    # Duplicate the specified vertices
    updated_graph = duplicate_vertices(cycle_graph, vertices_to_duplicate)
    return updated_graph