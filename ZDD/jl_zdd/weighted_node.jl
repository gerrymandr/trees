# node.jl
include("forbidden_pair.jl")

mutable struct Node
    label::NodeEdge
    comp_weights::Vector{UInt32}
    cc::UInt8                   # can hold only 256 possible values
    fps::Vector{ForbiddenPair}
    comp_assign::Vector{UInt8}  # only 256 possible values
    deadend::Bool
    first_idx::UInt8
    hash::UInt64
    paths::Int

    # allow for incomplete initialization
    function Node()::Node
        new()
    end

    function Node(i::Int)::Node # for Terminal Nodes
        node = new(NodeEdge(i, i), Vector{UInt32}([1]), 0, Vector{ForbiddenPair}(), Vector{UInt8}([1]), true, UInt8(1), 0, 0)
        node.hash = hash(node)
        return node
    end

    function Node(root_edge::NodeEdge, base_graph::SimpleGraph, weights::Vector{UInt32})::Node
        comp_assign = Vector{UInt8}([i for i in 1:nv(base_graph)])
        comp_weights = weights # initialize each vertex's population based on user input
        node = new(root_edge, comp_weights, 0, Vector{ForbiddenPair}(), comp_assign, true, UInt8(1), 0, 1)
        node.hash = hash(node)
        return node
    end

    function Node(label::NodeEdge, comp_weights::Vector{UInt32},
                  cc::UInt8, fps::Vector{ForbiddenPair}, comp_assign::Vector{UInt8},
                  deadend::Bool, first_idx::UInt8, paths::Int)::Node
        return new(label, comp_weights, cc, fps, comp_assign, deadend, first_idx, paths)
    end
end

function custom_deepcopy(n::Node, recycler::Stack{Node}, x::Int8)::Node
    if x == 1
        return n
    end
    if isempty(recycler)
        comp_weights = Vector{UInt32}(undef, length(n.comp_weights))
        comp_assign = zeros(UInt8, length(n.comp_assign))
        fps = Vector{ForbiddenPair}(undef, length(n.fps))

        copy_to_vec_from_idx!(n.comp_weights, comp_weights, n.first_idx)
        copy_to_vec_from_idx!(n.comp_assign, comp_assign, n.first_idx)
        copy_to_vec!(n.fps, fps)

        return Node(n.label, comp_weights, n.cc, fps, comp_assign, true, n.first_idx, n.paths)
    else
        n′ = pop!(recycler)

        # empty the fields
        resize!(n′.fps, length(n.fps))

        # fill
        copy_to_vec!(n.fps, n′.fps)
        copy_to_vec_from_idx!(n.comp_weights, n′.comp_weights, n.first_idx)
        copy_to_vec_from_idx!(n.comp_assign, n′.comp_assign, n.first_idx)

        n′.cc = n.cc
        n′.first_idx = n.first_idx
        n′.deadend = n.deadend
        n′.label = n.label
        n′.paths = n.paths

        return n′
    end
end

function Base.hash(n::Node)
    """ 
    """
    comp_weights = @view n.comp_weights[n.first_idx:end]
    comp_assign = @view n.comp_assign[n.first_idx:end]

    hash(n.label, hash(n.cc, hash(comp_weights, hash(comp_assign, hash(n.fps)))))
end

function node_summary(node::Node)
    println("Label: ", readable(node.label))
    println("cc: ", readable(node.cc))
    println("fps: ", readable(node.fps))
    println("comp_assign: ", readable(node.comp_assign))
    println("comp_weights: ", readable(node.comp_weights))
    println()
end
