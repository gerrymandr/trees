from tqdm import tqdm
import networkx as nx
import copy
from enumpart_classes import *
from enumpart_helpers import *

def makeNewNode(edges, K, n, i, x, frontiers):
    v, w = n.edge
    m = len(edges)
    n_prime = copy.deepcopy(n)

    # construct the frontiers
    prevFrontier, currFrontier = frontiers[i], frontiers[i+1]

    # if either of the vertices are entering the frontier, add them to comp
    for u in (v, w):
        if u not in prevFrontier:
            n_prime.comp.add(tup([u]))
    for vertex_set in n_prime.comp:
        if v in vertex_set:
            C_v = copy.copy(vertex_set) # not sure if copy is needed
        if w in vertex_set:
            C_w = copy.copy(vertex_set)

    # if we are adding this edge, the components should now be connected
    if x == 1:
        n_prime.comp.discard(C_v) # remove x if it exists
        n_prime.comp.discard(C_w)
        n_prime.comp.add(tupUnion(C_v, C_w))
        # if they are a forbidden pair, return 0
        if C_v != C_w and tup((C_v, C_w)) in n_prime.fps:
            return 0
        # otherwise, replace all instances of C_v/C_w with their union in the fps
        else:
            fps_list = list(n_prime.fps)
            for elem in fps_list:
                new_elem = set(elem)
                if C_v in new_elem or C_w in new_elem:
                    if C_v in new_elem:
                        new_elem.remove(C_v)
                        new_elem.add(tupUnion(C_v, C_w))
                    if C_w in elem:
                        new_elem.remove(C_w)
                        new_elem.add(tupUnion(C_v, C_w))
                    new_elem = tup(new_elem)
                    n_prime.fps.remove(elem)
                    n_prime.fps.add(new_elem)

    else:
        # if we aren't adding this edge and the components are connected,
        # this violates induced subgraph, so return 0
        if C_v == C_w:
            return 0
        # if they aren't connected, add their connected components to the fps
        else:
            n_prime.fps.add(tup((C_v, C_w)))

    # if either of the vertices are leaving the frontier...
    for u in (v, w):
        if u not in currFrontier:
            # if u is an isolated component, increment our cc by 1
            if tup([u]) in n_prime.comp:
                n_prime.cc += 1
                # remove it from the range of the comp or fps's views...
                n_prime.comp.remove(tup([u]))
                # if we have too many cc's, return 0
                if n_prime.cc > max(K):
                    return 0

            # if u is a part of a connected component, take it out
            comp_list = list(n_prime.comp)
            for elem in comp_list:
                if u in elem:
                    new_elem = set(elem)
                    new_elem.remove(u)
                    new_elem = tup(new_elem)
                    n_prime.comp.remove(elem)
                    n_prime.comp.add(new_elem)

            # if {{u}, X} is in n_prime.fps for any X in n_prime.comp, take it out
            fps_list = list(n_prime.fps)
            for elem in fps_list:
                if tup([u]) in elem:
                    n_prime.fps.remove(elem)
                # same with u if it is a part of a vertex set in fps
                for vertex_set in elem:
                    if len(vertex_set) > 1 and u in vertex_set:
                        new_elem = set(elem)
                        for vertex_set in elem:
                            if len(vertex_set) > 1 and u in vertex_set:
                                new_vertex_set = set(vertex_set)
                                new_vertex_set.remove(u)
                                new_elem.remove(vertex_set)
                                new_elem.add(tup(new_vertex_set))
                                new_elem = tup(new_elem)
                        n_prime.fps.remove(elem)
                        n_prime.fps.add(new_elem)

    # if we are on the last edge, check to see if we have partitioned
    # the graph into the right number of cells
    if i == (m-1):
        if n_prime.cc in K:
            return 1
        else:
            return 0

    return n_prime

def constructZDD(G, K, edges, draw=False):
    m = len(edges)
    Nodes = [copy.deepcopy([]) for _ in range(m)]
    Nodes[0].append(Node(edges[0]))
    ZDD, node_labels, edge_labels, pos = initialize_ZDD(Nodes)
    frontiers = compute_all_frontiers(edges)
    for i in tqdm(range(m)):
        for n in Nodes[i]:
            for x in [0, 1]:
                n_prime = makeNewNode(edges, [K], n, i, x, frontiers)

                if n_prime == 0 or n_prime == 1:
                    if 0 not in ZDD.nodes:
                        add_terminal_nodes(ZDD, node_labels, pos)
                    add_edge(n, n_prime, x, ZDD, edge_labels)
                else:
                    n_prime = Node(edges[i+1], n_prime.cc, n_prime.comp, n_prime.fps)
                    foundCopy = False
                    for n_pp in Nodes[i+1]:
                        if n_prime.cc == n_pp.cc and n_prime.comp == n_pp.comp and n_prime.fps == n_pp.fps:
                            foundCopy = True
                            ncopy = n_pp
                    if foundCopy:
                        add_edge(n, ncopy, x, ZDD, edge_labels)
                    else:
                        Nodes[i+1].append(n_prime)
                        add_node(n, n_prime, x, i, ZDD, node_labels, edge_labels, pos)
                if draw:
                    draw_ZDD(ZDD, node_labels, edge_labels, pos)
    return (ZDD, node_labels, edge_labels, pos)
