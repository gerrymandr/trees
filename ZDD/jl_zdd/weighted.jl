using DataStructures
using LightGraphs
using Statistics
import Base: Dict
using JSON

include("weighted_node.jl")
include("node_auxillaries.jl")
include("grid.jl")
include("frontier.jl")
include("edge_ordering.jl")
include("zdd.jl")
include("count_enumerate.jl")
include("visualization.jl")


function make_new_node(g::SimpleGraph,
                       g_edges::Array{NodeEdge,1},
                       k::Int,
                       n::Node,
                       i::Int,
                       x::Int8,
                       d::Int,
                       frontiers::Array{Set{UInt8}, 1},
                       lower_bound::Int32,
                       upper_bound::Int32,
                       zero_terminal::Node,
                       one_terminal::Node,
                       fp_container::Vector{ForbiddenPair},
                       rm_container::Vector{ForbiddenPair},
                       lower_vs::Vector{UInt8},
                       recycler::Stack{Node}
                       )
    """
    """
    u = g_edges[i].edge₁
    v = g_edges[i].edge₂

    n′ = custom_deepcopy(n, recycler, x)

    prev_frontier, curr_frontier = frontiers[i], frontiers[i+1]

    Cᵤ, Cᵥ = components(u, v, n′)

    if x == 1
        connect_components!(n′, Cᵤ, Cᵥ)
        @inbounds if n′.comp_weights[max(Cᵤ, Cᵥ)] > upper_bound # --> 0 if new connected component is too big
            push!(recycler, n′)
            return zero_terminal
        end
        if Cᵤ != Cᵥ && ForbiddenPair(min(Cᵤ, Cᵥ), max(Cᵤ, Cᵥ)) in n′.fps
            push!(recycler, n′)
            return zero_terminal
        else
            replace_components_with_union!(n′, Cᵤ, Cᵥ, fp_container, rm_container)
        end
    else
        if Cᵤ == Cᵥ
            push!(recycler, n′)
            return zero_terminal
        else
            push!(n′.fps, ForbiddenPair(min(Cᵤ, Cᵥ), max(Cᵤ, Cᵥ)))
        end
    end

    for a in prev_frontier
        if a ∉ curr_frontier
            @inbounds a_comp = n′.comp_assign[a]
            comp_assign = @view n.comp_assign[n′.first_idx:a_comp] # indices > a_comp wouldn't be labeled as a_comp, so we can ignore them

            if count(x -> x == a_comp, comp_assign) == 1
                @inbounds if n′.comp_weights[a_comp] < lower_bound
                    push!(recycler, n′)
                    return zero_terminal
                end
                @inbounds n′.comp_weights[a_comp] = 0
                n′.cc += 1
                if n′.cc > k
                    push!(recycler, n′)
                    return zero_terminal
                end
            end
            remove_vertex_from_node!(n′, a, fp_container, rm_container, lower_vs)
        end
    end

    if i == length(g_edges)
        if n′.cc == k
            push!(recycler, n′)
            return one_terminal
        else
            push!(recycler, n′)
            return zero_terminal
        end
    end

    return n′
end

function connect_components!(n::Node, Cᵤ::UInt8, Cᵥ::UInt8)
    """
    If the two components are different, remove the smaller component from n.comp,
    and update n.comp_assign.
    Adjust n.comp_weights to remove the smaller-indexed component and add its weight into the larger one.
    """
    assignment = max(Cᵤ, Cᵥ)
    to_change = min(Cᵤ, Cᵥ)
    if Cᵤ != Cᵥ
        map!(val -> val == to_change ? assignment : val, n.comp_assign, n.comp_assign)
        @inbounds n.comp_weights[assignment] += n.comp_weights[to_change]
        @inbounds n.comp_weights[to_change] = 0
    end
end


function adjust_node!(node::Node,
                      vertex_comp::UInt8,
                      fp_container::Vector{ForbiddenPair},
                      rm_container::Vector{ForbiddenPair},
                      lower_vs::Vector{UInt8})
    """
    """
    # there is atleast one lower vertex number that has the higher comp
    # number and needs to be adjusted
    lower_vertices!(vertex_comp, node.comp_assign, lower_vs) #findall(x->x==vertex_comp, node.comp_assign)
    new_max = maximum(lower_vs)

    # change comp.assign
    for v in lower_vs
        node.comp_assign[v] = new_max
    end

    # change comp_weights
    if new_max != vertex_comp
        node.comp_weights[new_max] = node.comp_weights[vertex_comp]
        node.comp_weights[vertex_comp] = 0
    end

    # change ForbiddenPair
    for fp in node.fps
        if vertex_comp == fp.comp₁
            other = fp.comp₂
            push!(rm_container, fp)
            push!(fp_container, ForbiddenPair(min(new_max, other), max(new_max, other)))
        elseif vertex_comp == fp.comp₂
            other = fp.comp₁
            push!(rm_container, fp)
            push!(fp_container, ForbiddenPair(min(new_max, other), max(new_max, other)))
        end
    end
    filter!(x -> x ∉ rm_container, node.fps)
    append!(node.fps, fp_container)
    empty!(fp_container)
    empty!(rm_container)
end
