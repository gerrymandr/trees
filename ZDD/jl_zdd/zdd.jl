using DataStructures
using LightGraphs
using Statistics

include("node.jl")
include("grid.jl")
include("frontier.jl")
include("edge_ordering.jl")

mutable struct ZDD{G<:SimpleDiGraph, S<:SimpleGraph, N<:Node}
    # graph stuff
    graph::G
    nodes::Dict{UInt64, Int64}
    nodes_complete::Dict{N, Int64} # used only when viz = True
    edges::Dict{Tuple{N, N}, Int8}      # used only when viz = True
    edge_multiplicity::Set{Tuple{N, N}}
    base_graph::S
    root::N
    viz::Bool
end

function ZDD(g::SimpleGraph, root::Node; viz::Bool=false)::ZDD
    graph = SimpleDiGraph(3) # 2 for terminal nodes and 1 for the root
    nodes = Dict{UInt64, Int64}()
    nodes[hash(Node(0))] = 1
    nodes[hash(Node(1))] = 2
    nodes[hash(root)] = 3

    nodes_complete = Dict{Node, Int64}()
    nodes_complete[Node(0)] = 1
    nodes_complete[Node(1)] = 2
    nodes_complete[root] = 3

    edges = Dict{Tuple{Node,Node}, Int8}()
    edge_multiplicity = Set{Tuple{Node,Node}}()
    base_graph = g
    return ZDD(graph, nodes, nodes_complete, edges, edge_multiplicity, base_graph, root, viz)
end

# these need to be included only after the ZDD struct is defined
include("count_enumerate.jl")
include("visualization.jl")

function add_zdd_edge!(zdd::ZDD,
                       node₁::Node,
                       node₂::Node,
                       node₁_idx::Int64,
                       x::Int8)
    """ zdd_edge is represented as (Node, Node)
    """
    if zdd.viz
        if (node₁, node₂) in keys(zdd.edges)
            push!(zdd.edge_multiplicity, (node₁, node₂))
        else
            zdd.edges[(node₁, node₂)] = x
        end
    end

    node₂_idx = zdd.nodes[hash(node₂)]

    # add to simple graph
    add_edge!(zdd.graph, node₁_idx, node₂_idx)

    nothing
end

function num_edges(zdd::ZDD)
    length(zdd.edges) + length(zdd.edge_multiplicity)
end

function construct_zdd(g::SimpleGraph,
                       k::Int64,
                       d::Int64,
                       g_edges::Array{NodeEdge,1};
                       viz::Bool=false)::ZDD
    root = Node(g_edges[1], g)

    lower_bound = Int8(nv(g)/k - d) # TODO: extend to non-nice ratios
    upper_bound = Int8(nv(g)/k + d)

    zdd = ZDD(g, root, viz=viz)
    N = Vector{Set{Node}}([Set{Node}([]) for a in 1:ne(g)+1])
    N[1] = Set([root])
    frontiers = compute_all_frontiers(g, g_edges)
    xs = Vector{Int8}([0, 1])
    zero_terminal = Node(0)
    one_terminal = Node(1)
    fp_container = Vector{ForbiddenPair}([]) # reusable container

    for i = 1:ne(g)
        for n in N[i]
            n_idx = zdd.nodes[hash(n)]
            for x in xs
                n′ = make_new_node(g, g_edges, k, n, i, x, d, frontiers,
                                   lower_bound, upper_bound,
                                   zero_terminal, one_terminal,
                                   fp_container)

                if !(n′.label == NodeEdge(0, 0) || n′.label == NodeEdge(1, 1)) # if not a Terminal Node
                    n′.label = g_edges[i+1] # update the label of n′

                    if n′ ∉ N[i+1]
                        push!(N[i+1], n′)
                        add_zdd_node_and_edge!(zdd, n′, n, n_idx, x)
                        continue
                    end
                end
                add_zdd_edge!(zdd, n, n′, n_idx, x)
            end
        end
        # sizes = [Base.summarysize(n) for n in N[i]]
        # println("Level: ", i)
        # println("Minimum: ", minimum(sizes))
        # println("Mean: ", mean(sizes))
        # println()
        N[i] = Set{Node}([]) # release memory
    end

    return zdd
end



function make_new_node(g::SimpleGraph,
                       g_edges::Array{NodeEdge,1},
                       k::Int,
                       n::Node,
                       i::Int,
                       x::Int8,
                       d::Int,
                       frontiers::Array{Set{UInt8}, 1},
                       lower_bound::Int8,
                       upper_bound::Int8,
                       zero_terminal::Node,
                       one_terminal::Node,
                       fp_container::Vector{ForbiddenPair})
    """
    """
    u = g_edges[i].edge₁
    v = g_edges[i].edge₂

    n′ = custom_deepcopy(n)
    # n′ = deepcopy(n)
    prev_frontier, curr_frontier = frontiers[i], frontiers[i+1]

    add_vertex_as_component!(n′, u, prev_frontier)
    add_vertex_as_component!(n′, v, prev_frontier)

    Cᵤ, Cᵥ = components(u, v, n′)

    if x == 1
        connect_components!(n′, Cᵤ, Cᵥ)
        if n′.comp_weights[max(Cᵤ, Cᵥ)] > upper_bound # --> 0 if new connected component is too big
            return zero_terminal
        end
        if Cᵤ != Cᵥ && ForbiddenPair(min(Cᵤ, Cᵥ), max(Cᵤ, Cᵥ)) in n′.fps
            return zero_terminal
        else
            replace_components_with_union!(n′, Cᵤ, Cᵥ, fp_container)
        end
    else
        if Cᵤ == Cᵥ
            return zero_terminal
        else
            push!(n′.fps, ForbiddenPair(min(Cᵤ, Cᵥ), max(Cᵤ, Cᵥ)))
        end
    end

    for a in prev_frontier
        if a ∉ curr_frontier
            a_comp = n′.comp_assign[a]
            if a_comp in n′.comp && length(filter(x -> x == a_comp, n′.comp_assign)) == 1
                if n′.comp_weights[a_comp] < lower_bound
                    return zero_terminal
                end
                n′.comp_weights[a_comp] = 0
                n′.cc += 1
                if n′.cc > k
                    return zero_terminal
                end
            end
            remove_vertex_from_node_fps!(n′, a, fp_container)
        end
    end

    if i == length(g_edges)
        if n′.cc == k
            return one_terminal
        else
            return zero_terminal
        end
    end
    return n′
end


function add_vertex_as_component!(n′::Node, vertex::UInt8, prev_frontier::Set{UInt8})
    """ Add `u` or `v` or both to n`.comp if they are not in
        `prev_frontier`
    """
    if vertex ∉ prev_frontier
        push!(n′.comp, vertex)
        sort!(n′.comp) # needed for Node equality to increase Node merges
    end
    nothing
end

function replace_components_with_union!(node::Node, Cᵤ::UInt8, Cᵥ::UInt8, fp_container::Vector{ForbiddenPair})
    """
    update fps to replace the smaller component with the larger component
    (TODO: rename)
    """
    assignment = max(Cᵤ, Cᵥ)
    to_change = min(Cᵤ, Cᵥ)

    for fp in node.fps
        if to_change == fp.comp₁
            other = fp.comp₂
            delete!(node.fps, fp)
            push!(fp_container, ForbiddenPair(min(assignment, other), max(assignment, other)))
        elseif to_change == fp.comp₂
            other = fp.comp₁
            delete!(node.fps, fp)
            push!(fp_container, ForbiddenPair(min(assignment, other), max(assignment, other)))
        end
    end
    for fp in fp_container
        push!(node.fps, fp)
    end
    empty!(fp_container)
end


function connect_components!(n::Node, Cᵤ::UInt8, Cᵥ::UInt8)
    """
    If the two components are different, remove the smaller component from n.comp, and update n.comp_assign. Adjust n.comp_weights to remove the smaller-indexed component and add its weight into the larger one.
    """
    assignment = max(Cᵤ, Cᵥ)
    to_change = min(Cᵤ, Cᵥ)
    if Cᵤ != Cᵥ
        map!(val -> val == to_change ? assignment : val, n.comp_assign, n.comp_assign)
        filter!(x -> x != to_change, n.comp)
        n.comp_weights[assignment] += n.comp_weights[to_change]
        n.comp_weights[to_change] = 0
    end
end

function components(u::UInt8, v::UInt8, node::Node)::Tuple{UInt8, UInt8}
    """ Returns Cᵤ and Cᵥ which are the sets in `components` that contain
        vertices `u` and `v` respectively.
    """
    return node.comp_assign[u], node.comp_assign[v]
end

function remove_vertex_from_node_fps!(node::Node, vertex::UInt8, fp_container::Vector{ForbiddenPair})
    """
    """
    vertex_comp = node.comp_assign[vertex]

    for fp in node.fps
        if (vertex_comp == fp.comp₁ || vertex_comp == fp.comp₂) && length(filter(x -> x == vertex_comp, node.comp_assign)) == 1
            delete!(node.fps, fp)
            filter!(x -> x != vertex_comp, node.comp)
        end
    end

    node.comp_assign[vertex] = 0
    adjust_node!(node, vertex_comp, fp_container)
end

function adjust_node!(node::Node, vertex_comp::UInt8, fp_container::Vector{ForbiddenPair})
    """
    """
    if vertex_comp in node.comp_assign
        # there is atleast one lower vertex number that has the higher comp
        # number and needs to be adjusted
        lower_vertices = findall(x->x==vertex_comp, node.comp_assign)
        new_max = maximum(lower_vertices)

        # change comp.assign
        for v in lower_vertices
            node.comp_assign[v] = new_max
        end

        # change comp
        filter!(x -> x != vertex_comp, node.comp)
        push!(node.comp, new_max)
        sort!(node.comp) # needed for Node equality to increase Node merges

        if new_max != vertex_comp
            node.comp_weights[new_max] = node.comp_weights[vertex_comp]
            node.comp_weights[vertex_comp] = 0
        end

        # change ForbiddenPair
        for fp in node.fps
            if vertex_comp == fp.comp₁
                other = fp.comp₂
                delete!(node.fps, fp)
                push!(fp_container, ForbiddenPair(min(new_max, other), max(new_max, other)))
            elseif vertex_comp == fp.comp₂
                other = fp.comp₁
                delete!(node.fps, fp)
                push!(fp_container, ForbiddenPair(min(new_max, other), max(new_max, other)))
            end
        end
        for fp in fp_container
            push!(node.fps, fp)
        end
    end
    empty!(fp_container)
end

function add_zdd_node_and_edge!(zdd::ZDD, n′::Node, n::Node, n_idx::Int64, x::Int8)
    """
    """
    add_vertex!(zdd.graph)
    n′_idx = nv(zdd.graph)
    zdd.nodes[hash(n′)] = n′_idx

    if zdd.viz
        zdd.nodes_complete[n′] = n′_idx

        if (n, n′) in keys(zdd.edges)
            push!(zdd.edge_multiplicity, (n, n′))
        else
            zdd.edges[(n, n′)] = x
        end
    end

    # add to simple graph
    add_edge!(zdd.graph, n_idx, n′_idx)
end
