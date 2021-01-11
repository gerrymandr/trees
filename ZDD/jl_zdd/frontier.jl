# frontier.jl

function compute_frontier(edges, i::Int)
    """
    Inputs:
        - set of edges of the base graph
        - index of edge being processed
    Output:
        - the current frontier
    """
    m = length(edges)
    if i == 1 || i == m+1
        return Set{Int}()
    else
        processed_nodes = nodes_from_edges(edges[1:i-1])
        unprocessed_nodes = nodes_from_edges(edges[i:m])
        return intersect(Set(processed_nodes), Set(unprocessed_nodes))
    end
end

function compute_all_frontiers(g::SimpleGraph, g_edges)::Vector{Set{UInt8}}
    frontiers = Vector{Set{UInt8}}([Set{UInt8}() for a in 1:ne(g)+1])
    for i ∈ 1:ne(g)+1
        frontiers[i] = compute_frontier(g_edges, i)
    end
    return frontiers
end


function nodes_from_edges(edges)::Set{UInt8}
    """
    """
    nodes = Set{UInt8}()
    for e in edges
        push!(nodes, e.edge₁)
        push!(nodes, e.edge₂)
    end
    return nodes
end

function frontier_distribution(g::SimpleGraph, g_edges::Array{LightGraphs.SimpleGraphs.SimpleEdge{Int}, 1})
    frontiers = compute_all_frontiers(g, g_edges)
    for i ∈ 0:maximum(length(frontier) for frontier in frontiers)
        c = count(length(frontier) == i for frontier in frontiers)
    end
    return
end

function frontier_distribution(frontiers::Array{Set{Int},1})
    for i ∈ 0:maximum(length(frontier) for frontier in frontiers)
        c = count(length(frontier) == i for frontier in frontiers)
    end
    return
end
