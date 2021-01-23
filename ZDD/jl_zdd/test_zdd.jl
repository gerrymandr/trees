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
println("***** SOLUTION TESTS PASSED $correct/$tests *****")


###
# Node-merging tests

"""
All of these were built on Jan 22, 2021 using
* the optimal_grid_edge_order_diags() function
* k = m where m x m is the size of the grid and k denotes the number of partitions.
* d = 0 i.e the only exactly population balanced components are allowed.

There is no guarantee that these numbers are correct, and even less
of a guarantee that these are the most optimal numbers. Indeed, if we succeed in
detecting dead nodes early on then these numbers will definitely change.
All of this is to say -- if these tests fail then there is a real chance that
the tests are faulty in themselves, especially if the zdd that is failing tests
is smaller than these zdds.
"""

node_per_level_2x2 = Dict{Float64,Int64}(
    4.0 => 2,
    2.0 => 2,
    3.0 => 2,
    5.0 => 2,
    1.0 => 1
    )

node_per_level_3x3 = Dict{Float64, Int64}(
    2.0  => 2,
    11.0 => 5,
    7.0  => 9,
    9.0  => 12,
    10.0 => 13,
    8.0  => 9,
    6.0  => 8,
    4.0  => 5,
    3.0  => 3,
    5.0  => 6,
    13.0 => 2,
    12.0 => 4,
    1.0  => 1
    )

node_per_level_4x4 = Dict{Float64,Int64}(
  18.0 => 149,
  2.0  => 2,
  16.0 => 116,
  11.0 => 52,
  21.0 => 40,
  7.0  => 13,
  9.0  => 32,
  25.0 => 2,
  10.0 => 43,
  19.0 => 57,
  17.0 => 129,
  8.0  => 24,
  22.0 => 40,
  6.0  => 11,
  24.0 => 9,
  4.0  => 6,
  3.0  => 3,
  5.0  => 9,
  20.0 => 51,
  23.0 => 9,
  13.0 => 76,
  14.0 => 82,
  15.0 => 101,
  12.0 => 66,
  1.0  => 1
  )

global merge_tests, merge_correct = 0, 0
function test_node_merging(g::SimpleGraph, dims, merge_solutions)
    global merge_tests += 1
    n = dims[1]
    d = 0


    println("Node-merging check $(n)x$(n) grid -> $n districts (d = 0): ")

    g_edges = optimal_grid_edge_order_diags(g, dims[1], dims[2])
    g_edges = convert_lightgraphs_edges_to_node_edges(g_edges)
    zdd = construct_zdd(g, n, d, g_edges, viz=true)

    _, loc_ys = node_locations(zdd, g_edges) # from visualization.jl

    u = unique(loc_ys)
    level_counts = Dict([(i,count(x -> x == i, loc_ys)) for i in u])

    if level_counts == merge_solutions[n-1]
        println("*** CORRECT :) ***\n")
        global merge_correct += 1
    else
        println("*** INCORRECT :( ***\n")
        println("Expected: ", merge_solutions[n-1])
        println()
        println("Got: ", level_counts)
        println()
    end
end

merge_solutions = [node_per_level_2x2, node_per_level_3x3, node_per_level_4x4]

for i in 2:4
    dims = [i, i]
    g = grid(dims)
    test_node_merging(g, dims, merge_solutions)
end

println("*\n*\n*")
println("***** NODE MERGING TESTS PASSED $merge_correct/$merge_tests *****")
