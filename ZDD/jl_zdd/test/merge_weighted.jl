module merge_weighted

using Test
include("../weighted.jl")
include("./utils.jl")

const solutions_rook = Dict(
    ([2,2],0) => 9,
    ([3,3],0) => 79,
    ([4,4],0) => 1120,
    ([5,5],0) => 21494,
)

const solutions_queen = Dict(
    ([2,2],0) => 18,
    ([3,3],0) => 336,
    ([4,4],0) => 13883,
    ([5,5],0) => 1067538,
)

function test_merge(solutions::Dict{Tuple{Array{Int64,1},Int64},Int64},
                    make_grid::Function,
                    make_edges::Function)
    for (case, truth) ∈ solutions
        n, k = case[1][1], case[1][2]
        d = case[2]

        zdd = prepare_zdd([n,k], d, make_grid, make_edges)
        calculated_size = num_nodes(zdd)

        @test calculated_size  == truth
    end
end

@testset "Weighted" begin

    @testset "Rook Contiguity" begin
        test_merge(solutions_rook, grid, optimal_grid_edge_order_diags)
    end

    @testset "Queen Contiguity" begin
        test_merge(solutions_queen, queen_grid, optimal_queen_grid_edge_order)
    end
end

end
