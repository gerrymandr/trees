function prepare_zdd(dims,d,make_grid::Function,make_edges::Function)
    n = dims[1]
    g = make_grid([n,n])
    g_edges = make_edges(g, n, n)
    g_edges = convert_lightgraphs_edges_to_node_edges(g_edges)
    return construct_zdd(g, n, d, g_edges)
end
