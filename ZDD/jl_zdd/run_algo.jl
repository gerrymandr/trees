using Pkg; Pkg.activate("zdd", shared=true)
using LightGraphs
using GraphPlot
# using ProfileView
using StatProfilerHTML
using Profile

# include("weightless_zdd.jl")
include("zdd.jl")

function run_algo(dim, k, d)
# to compile it the first time round
    # println("Making a small ZDD for compiling purposes")
    g = grid([dim, dim])
    g_edges = optimal_grid_edge_order(g, dim, dim)
    g_edges = convert_lightgraphs_edges_to_node_edges(g_edges)
    zdd = construct_zdd(g, k, d, g_edges)
end

# @profview zdd = construct_zdd(g, k)

# Profile.clear()

# println("Actually making the ZDD we want")
# dim = 6
# const g = grid([6, 6])
# const g_edges = optimal_grid_edge_order(g, 6, 6)
# # zdd = construct_zdd(g, k)
# # @profilehtml zdd = construct_zdd(g, 12, g_edges)
# zdd = construct_zdd(g, 12, g_edges)

run_algo(3, 3, 0) # test

Profile.clear_malloc_data()

run_algo(6, 6, 0)
