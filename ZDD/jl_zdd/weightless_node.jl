
include("forbidden_pair.jl")

# The 1st node is always the 0-terminal, and the 2nd node is always the 1 terminal. Adding the first node to the ZDD
# means the ZDD will have 3 nodes, the node + the two terminal nodes
mutable struct Node
    label::NodeEdge
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
        node = new(NodeEdge(i, i), 0, Vector{ForbiddenPair}(), Vector{UInt8}([]), true, UInt8(1), 0, 0)
        node.hash = hash(node)
        return node
    end

    function Node(root_edge::NodeEdge, base_graph::SimpleGraph, weights::Vector{UInt32})::Node
        """ the `weights` argument is unused, and is only here to support compatibility with
            weighted zdds.
        """
        comp_assign = Vector{UInt8}([i for i in 1:nv(base_graph)])
        node = new(root_edge, 0, Vector{ForbiddenPair}(), comp_assign, true, UInt8(1), 0, 1)
        node.hash = hash(node)
        return node
    end

    function Node(label::NodeEdge,
                  cc::UInt8, fps::Vector{ForbiddenPair}, comp_assign::Vector{UInt8},
                  deadend::Bool, first_idx::UInt8, paths::Int)::Node
        return new(label, cc, fps, comp_assign, deadend, first_idx, 0, paths)
    end
end

function custom_deepcopy(n::Node, recycler::Stack{Node}, x::Int8)::Node
    if x == 1
        return n
    end
    if isempty(recycler)
        comp_assign = zeros(UInt8, length(n.comp_assign))
        fps = Vector{ForbiddenPair}(undef, length(n.fps))

        copy_to_vec_from_idx!(n.comp_assign, comp_assign, n.first_idx)
        copy_to_vec!(n.fps, fps)

        return Node(n.label, n.cc, fps, comp_assign, true, n.first_idx, n.paths)
    else
        n′ = pop!(recycler)

        # empty the fields
        resize!(n′.fps, length(n.fps))

        # fill
        copy_to_vec!(n.fps, n′.fps)
        copy_to_vec_from_idx!(n.comp_assign, n′.comp_assign, n.first_idx)

        n′.cc = n.cc
        n′.first_idx = n.first_idx
        n′.deadend = n.deadend
        n′.label = n.label
        n′.paths = n.paths

        return n′
    end
end

function Base.hash(n::Node, h::UInt)
    """ Reference: https://stackoverflow.com/questions/3404715/c-sharp-hashcode-for-array-of-ints
    """
    comp_assign = @view n.comp_assign[n.first_idx:end]

    ca = hash_arr(comp_assign)
    fps = hash_arr(n.fps)

    total = 17 * ca + 47 * fps + 71 * n.cc

    hash(n.label, hash(total))
end

function node_summary(node::Node)
    println("Label: ", readable(node.label))
    println("cc: ", readable(node.cc))
    println("comp: ", readable(node.comp))
    println("fps: ", readable(node.fps))
    println("comp_assign: ", readable(node.comp_assign))
    println()
end
