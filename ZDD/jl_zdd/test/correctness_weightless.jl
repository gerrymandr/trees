module correctness_weightless

using Test
include("../weightless.jl")
include("./utils.jl")

const solutions_rook = Dict(
  # ([n,k], d) => partitions,
    ([2,2], 0) => 6,
    ([2,2], 3) => 6,
    ([3,3], 0) => 258,
    ([4,4], 0) => 62741,
    ([5,5], 0) => 72137699,
    ([5,5], 3) => 72137699,
    ([6,6], 0) => 356612826084
)

const solutions_queen = Dict(
  # ([n,k], d) => partitions,
    ([2,2], 0) => 7,
    ([2,2], 3) => 7,
    ([3,3], 0) => 782,
    ([4,4], 0) => 1130612,
    ([5,5], 0) => 19258645522
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



@testset "Weightless" begin

    @testset "Rook Contiguity" begin
        test_paths(solutions_rook, grid, optimal_grid_edge_order_diags)
    end

    @testset "Queen Contiguity" begin
        test_paths(solutions_queen, queen_grid, optimal_queen_grid_edge_order)
    end
end
end







