using LightGraphs
using StatProfilerHTML

x = parse(Int, ARGS[1])
y = parse(Int, ARGS[2])
k = parse(Int, ARGS[3])
d = parse(Int, ARGS[4])
contiguity = ARGS[5]
weighted = ARGS[6] == "weighted"

if weighted
  include("jl_zdd/weighted.jl")
else
  include("jl_zdd/weightless.jl")
end

zdd = construct_zdd(grid([2,2]), 2, 0, convert_lightgraphs_edges_to_node_edges(optimal_grid_edge_order_diags(grid([2,2]), 2, 2)))

println("Calculating all possible districtings of a $(x)x$y grid into $k districts of size $(Int((x*x)/k))+/-$(d), with $contiguity contiguity, weighted is $weighted.")
if contiguity == "rook"
	g = grid([x,y])
	g_edges = optimal_grid_edge_order_diags(g, x, y)
	results = " "
elseif contiguity == "queen"
	g = queen_grid([x,y])
	g_edges = optimal_queen_grid_edge_order(g, x, y)
	results = ""
else
	println("ERROR: contiguity must be 'rook' or 'queen'.")
end

g_edges = convert_lightgraphs_edges_to_node_edges(g_edges)

ret = @timed zdd = construct_zdd(g, k, d, g_edges); nothing
bytes = Base.summarysize(zdd)

spaces = convert(Int,25 - (trunc(log(count_paths(zdd)) / log(10)) + 1))
spacer = " "^spaces
if !weighted
  d = 99
end
results *= "$contiguity $(x)x$y grid -> $k districts (size $(Int((x*x)/k))+/-$(d)): $(count_paths(zdd))$(spacer)(Julia enumpart, took $(ret[2]), $(bytes) bytes)\n"
open("outputs/julia_enumpart.txt", "a") do file
	write(file, results)
end
	
