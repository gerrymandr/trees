module weighted

using Test
include("../weighted.jl")

solutions_rook = Dict(
  # ([n,k], d) => partitions,
    ([2,2], 0) => 2,
    ([2,2], 3) => 6,
    ([3,3], 0) => 10,
    ([3,3], 3) => 226,
    ([4,4], 0) => 117,
    ([4,4], 3) => 29069,
    ([5,5], 0) => 4006,
    ([5,5], 3) => 6194822,
    #([6,6], 0) => 451206,
    #([6,6], 3) => 4564190094,
)

solutions_queen = Dict(
  # ([n,k], d) => partitions,
    ([2,2], 0) => 3,
    ([2,2], 3) => 7,
    ([3,3], 0) => 36,
    ([3,3], 3) => 746,
    ([4,4], 0) => 2620,
    ([4,4], 3) => 710622,
    ([5,5], 0) => 1397790,
    #([5,5], 3) => 2467619460
)


function test_paths(solutions::Dict{Tuple{Array{Int64,1},Int64},Int64},
                    make_grid::Function,
                    make_edges::Function)
    for (case, truth) ∈ solutions
        n = case[1][1]
        d = case[2]

        g = make_grid([n,n])
        g_edges = make_edges(g, n, n)
        g_edges = convert_lightgraphs_edges_to_node_edges(g_edges)
        zdd = construct_zdd(g, n, d, g_edges)
        paths = count_paths(zdd)
        @test paths == truth
    end
end

@testset "Weighted Correctness Tests" begin

    @testset "Rook Contiguity" begin
        test_paths(solutions_rook, grid, optimal_grid_edge_order_diags)
    end

    @testset "Queen Contiguity" begin
        test_paths(solutions_queen, queen_grid, optimal_queen_grid_edge_order)
    end
end

end









