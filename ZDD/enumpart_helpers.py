import networkx as nx
import matplotlib.pyplot as plt
from tqdm import tqdm

"""
Small helpers
"""

def tup(l):
    return tuple(sorted(l))

def tupUnion(s, t):
    return tup(set(s).union(set(t)))

def vertices_from_edges(edges):
    vertices = []
    for e in edges:
        vertices.append(e[0])
        vertices.append(e[1])
    return set(vertices)

"""
Frontier functions
"""

def compute_frontier(edges, i):
    """
    Inputs:
        - set of edges of the base graph
        - index of edge being processed
    Output:
        - the current frontier
    """
    m = len(edges)
    if i == 0 or i == m:
        return set()
    else:
        processed_vertices = vertices_from_edges(edges[:i])
        unprocessed_vertices = vertices_from_edges(edges[i:])
        return processed_vertices.intersection(unprocessed_vertices)

def compute_all_frontiers(edges):
    """
    Input:
        - iterable of edges in base graph
    Output:
        - list of all m+1 frontiers
    """
    frontiers = []
    for i in range(len(edges)+1):
        frontiers.append(compute_frontier(edges, i))
    return frontiers

def count_paths(ZDD, node_labels, edge_labels, pos):
    """
    Probably a very slow way of counting paths!
    """
    p = {0: 0, 1:1} # initialize paths at 0, 1 terminal nodes

    for n in tqdm(reversed(list(ZDD.nodes))):
        if type(n) is int:
            continue
        count = 0
        for edge in edge_labels.keys(): # this is a waste!
            if edge[0] == n:
                count += p[edge[1]]
        p[n] = count
    for n in p.keys():
        if type(n) is not int:
            n.paths = p[n]
    return max(p.values())

"""
ZDD functions
"""

def initialize_ZDD(Nodes):
    ZDD = nx.Graph()
    node_labels, edge_labels, pos = {}, {}, {}
    first_node = Nodes[0][0]
    ZDD.add_node(first_node)
    node_labels[first_node] = "e_1"
    pos[first_node] = (8,10)
    return ZDD, node_labels, edge_labels, pos

def add_terminal_nodes(ZDD, node_labels, pos):
    ZDD.add_nodes_from([0, 1])
    node_labels.update({node:label for (node, label) in zip(range(2), ["0", "1"])})
    pos.update({node:coord for (node, coord) in zip(range(2), [(7,0), (9,0)])})
#     return ZDD, node_labels, pos

def add_edge(n, n_prime, x, ZDD, edge_labels):
    ZDD.add_edge(n, n_prime)
    edge_labels[(n, n_prime)] = f"{x}"
    return

def add_node(n, n_prime, x, i, ZDD, node_labels, edge_labels, pos):
    width = 1/(i+1)
    xPos = width if x else -width
    ZDD.add_node(n_prime)
    node_labels[n_prime] = f"e_{i+2}"
    pos[n_prime] = (pos[n][0] + xPos, 9-i)
    add_edge(n, n_prime, x, ZDD, edge_labels)
    return

def draw_ZDD(ZDD, node_labels, edge_labels, pos):
    plt.figure(figsize=(10,10))
    clear_output(wait=True)
    chosen_edges = []
    unchosen_edges = []
    edge_types_styles = ["dashed", "solid"]
    for edge in ZDD.edges:
        try:
            chosen = edge_labels[edge] == "1"
        except:
            chosen = edge_labels[edge[1], edge[0]] == "1"
        if chosen:
            chosen_edges.append(edge)
        else:
            unchosen_edges.append(edge)

    nx.draw(ZDD,
            pos=pos,
            labels=node_labels,
            edgelist=[],
            node_size=500,
            node_color="silver")
    if 0 in node_labels.keys():
         nx.draw(ZDD,
            pos=pos,
            labels=node_labels,
            edgelist=[],
            nodelist=[0,1],
            node_size=1000,
            node_color="red",
            node_shape="s")
    for i, edgelist in enumerate([unchosen_edges, chosen_edges]):
        nx.draw_networkx_edges(ZDD,
                               pos=pos,
                               edgelist=edgelist,
                               style=edge_types_styles[i],
                               width=2)
    plt.show()
    time.sleep(0)
    return

"""
Grid and Edge Ordering Functions
"""

def se_diag(x,y,x_max,y_max):
    edge_list = []
    while y > 0 and x < x_max:
        edge_list = edge_list + [((x,y),(x,y-1)),((x,y-1),(x+1,y-1))]
        x,y = x + 1, y - 1
    if y > 0:
        edge_list = edge_list + [((x,y),(x,y-1))]
    return edge_list

def nw_diag(x,y,x_max,y_max):
    edge_list = []
    while x > 0 and y < y_max:
        edge_list = edge_list + [((x,y),(x-1,y)),((x-1,y),(x-1,y+1))]
        x,y = x - 1, y + 1
    if x > 0:
        edge_list = edge_list + [((x,y),(x-1,y))]
    return edge_list

def optimal_grid_edge_order(G):
    #assumes G is grid with (x,y) nodes
    x_max = max([v[0] for v in G.nodes()])
    y_max = max([v[1] for v in G.nodes()])
    edge_list = []
    if x_max >= y_max:
        for y in range(y_max):
            edge_list = edge_list + se_diag(0,y+1,x_max,y_max)
        for x in range(x_max):
            edge_list = edge_list + [((x,y_max),(x+1,y_max))] + se_diag(x+1,y_max,x_max,y_max)
    else:
        for x in range(x_max):
            edge_list = edge_list + nw_diag(x+1,0,x_max,y_max)
        for y in range(y_max):
            edge_list = edge_list + [((x_max,y),(x_max,y+1))] + nw_diag(x_max,y+1,x_max,y_max)
    assert(len(edge_list)==len(G.edges()))
    for e in edge_list:
        assert(e in G.edges())
    return edge_list

def optimal_queen_grid_edge_order(G):
    #assumes G is grid with (x,y) nodes
    x_max = max([v[0] for v in G.nodes()])
    y_max = max([v[1] for v in G.nodes()])
    edge_list = []
    if x_max >= y_max:
        for x in range(x_max):
            for y in range(y_max):
                if x == 0:
                    edge_list.append(((x,y),(x,y+1)))
                edge_list.append(((x,y),(x+1,y)))
                edge_list.append(((x+1,y),(x,y+1)))
                edge_list.append(((x,y),(x+1,y+1)))
                edge_list.append(((x+1,y),(x+1,y+1)))
            edge_list.append(((x,y_max),(x+1,y_max)))
    else:
        for y in range(y_max):
            for x in range(x_max):
                if y == 0:
                    edge_list.append(((x,y),(x+1,y)))
                edge_list.append(((x,y),(x,y+1)))
                edge_list.append(((x+1,y),(x,y+1)))
                edge_list.append(((x,y),(x+1,y+1)))
                edge_list.append(((x,y+1),(x+1,y+1)))
            edge_list.append(((x_max,y),(x_max,y+1)))
    assert(len(set(edge_list))==len(G.edges()))
    for e in edge_list:
        assert(e in G.edges())
    return edge_list

def queen_grid_graph(dim):
    G = nx.grid_graph(dim=dim)
    G.add_edges_from([
        ((x, y), (x+1, y+1))
        for x in range(dim[1]-1)
        for y in range(dim[0]-1)
    ] + [
        ((x+1, y), (x, y+1))
        for x in range(dim[1]-1)
        for y in range(dim[0]-1)
    ])
    return G
