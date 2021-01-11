# grid.jl

function queen_grid(dims)
    g = grid(dims)
    r = dims[1]
    c = dims[2]
    for n ∈ 1:r*c
        if n == 1
            add_edge!(g, n, c+1)
        elseif n == r
            add_edge!(g, n, n+r-1)
        elseif n == r*c
            add_edge!(g, n, n-r-1)
        elseif n == r*(c-1)+1
            add_edge!(g, n, n-r+1)
        elseif n ∈ 1:r
            add_edge!(g, n, n+r-1)
            add_edge!(g, n, n+r+1)
        elseif (n-1) % r == 0
            add_edge!(g, n, n-r+1)
            add_edge!(g, n, n+r+1)
        elseif n % r == 0
            add_edge!(g, n, n-r-1)
            add_edge!(g, n, n+r-1)
        elseif n ∈ c*r:-1:(r-1)c+1
            add_edge!(g, n, n-r+1)
            add_edge!(g, n, n-r-1)
        else
            add_edge!(g, n, n+r-1)
            add_edge!(g, n, n+r+1)
            add_edge!(g, n, n-r-1)
            add_edge!(g, n, n-r+1)
        end
    end
    return g
end
