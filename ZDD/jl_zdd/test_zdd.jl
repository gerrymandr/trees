println("This script will test the enumpart algorithm stored in 'zdd.jl'")
println("*")
using LightGraphs
println("*")
include("zdd.jl")
println("*")

### Compiling `construct_zdd()` ###
grid_edges = convert_lightgraphs_edges_to_node_edges(optimal_grid_edge_order_diags(grid([2,2]), 2, 2))
construct_zdd(grid([2,2]), 2, 0, grid_edges); nothing
###

solutions = Dict(
  # (contiguity, n, k, d) => partitions,
    ("rook", 2, 2, 0) => 2,
    ("rook", 2, 2, 3) => 6,
    ("rook", 3, 3, 0) => 10,
    ("rook", 3, 3, 3) => 226,
    ("rook", 4, 4, 0) => 117,
    ("rook", 4, 4, 3) => 29069,
    ("rook", 5, 5, 0) => 4006,
    ("rook", 5, 5, 3) => 6194822,
    ("rook", 6, 6, 0) => 451206,
    ("rook", 6, 6, 3) => 4564190094,
    ("queen", 2, 2, 0) => 3,
    ("queen", 2, 2, 3) => 7,
    ("queen", 3, 3, 0) => 36,
    ("queen", 3, 3, 3) => 746,
    ("queen", 4, 4, 0) => 2620,
    ("queen", 4, 4, 3) => 710622,
    ("queen", 5, 5, 0) => 1397790,
    ("queen", 5, 5, 3) => 2467619460,
)

contiguities = ["rook", "queen"]
ns = [2,3,4,5]
ds = [0, 3]
global tests, correct = 0, 0
for contiguity ∈ contiguities
    for n ∈ ns
        for d ∈ ds
            # don't check queen 5 5 3 for time purposes
            if contiguity == "queen" && n == 5 && d == 3 continue end

            global tests += 1
            if contiguity == "rook"
                g = grid([n,n])
                g_edges = optimal_grid_edge_order_diags(g, n, n)
            else
                g = queen_grid([n,n])
                g_edges = optimal_queen_grid_edge_order(g, n, n)
            end
            g_edges = convert_lightgraphs_edges_to_node_edges(g_edges)

            if contiguity == "rook" print(" ") end # for lining up test results

            print("$contiguity $(n)x$(n) grid -> $n districts (d = $d): ")
            ret = @timed zdd = construct_zdd(g, n, d, g_edges); nothing
            time = ret[2]
            sols = count_paths(zdd)
            if sols == solutions[(contiguity, n, n, d)]
                print("*** CORRECT :) ***\n")
                println(" - partitions: $(sols)")
                println(" - time: $(time) secs.")
                global correct += 1
            else
                print("*** INCORRECT :( ***\n")
                println(" - Calculated partitions: $(sols)")
                println(" -     Actual partitions: $(solutions[(contiguity, n, k, d)])")
                println(" - time: $(time) secs.")
            end
        end
    end
end

println("*\n*\n*")
println("***** TESTS PASSED $correct/$tests *****")
