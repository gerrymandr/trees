using Test

const testdir = dirname(@__FILE__)

include("./utils.jl")

const tests = Dict(
    "Correctness" => ["correctness_weighted", "correctness_weightless"],
    "Merge" => ["merge_weighted", "merge_weightless"]
)


@testset "ZDD Julia Tests" begin
    for (category,files) in tests
        @testset "$category" begin
            for f in files
                tp = joinpath(testdir, "$(f).jl")
                include(tp)
            end
        end
    end
end