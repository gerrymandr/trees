module correctness_weighted

using Test
include("../weighted.jl")
include("./utils.jl")

const solutions_rook = Dict(
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

const solutions_queen = Dict(
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
        n, k = case[1][1], case[1][2]
        d = case[2]

        zdd = prepare_zdd([n,k], d, make_grid, make_edges)
        calculated_paths = count_paths(zdd)

        @test calculated_paths == truth
    end
end

@testset "Weighted" begin

    @testset "Rook Contiguity" begin
        test_paths(solutions_rook, grid, optimal_grid_edge_order_diags)
    end

    @testset "Queen Contiguity" begin
        test_paths(solutions_queen, queen_grid, optimal_queen_grid_edge_order)
    end
end
end









