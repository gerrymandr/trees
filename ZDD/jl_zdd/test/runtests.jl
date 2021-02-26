

using Test

const testdir = dirname(@__FILE__)


tests = [
    "correctness_weighted",
    "correctness_weightless",
    "merge"
]


@testset "ZDD Julia" begin
    for t in tests
        tp = joinpath(testdir, "$(t).jl")
        include(tp)
    end
end