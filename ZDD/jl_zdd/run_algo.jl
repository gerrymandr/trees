using Pkg; Pkg.activate("zdd", shared=true)
using LightGraphs
using GraphPlot
using StatProfilerHTML
using Profile

include("weightless_zdd.jl")
# include("zdd.jl")

function run_algo(dim, k, d)
    g = grid([dim, dim])
    g_edges = optimal_grid_edge_order_diags(g, dim, dim)
    g_edges = convert_lightgraphs_edges_to_node_edges(g_edges)
    zdd = construct_zdd(g, k, d, g_edges)
end

run_algo(3, 3, 0) # test

Profile.clear_malloc_data()

# @profilehtml run_algo(7, 7, 0)
run_algo(7, 7, 0)
