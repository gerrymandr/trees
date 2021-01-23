# node.jl
import Base: isequal, ==, isless

struct NodeEdge
    edge₁::UInt8
    edge₂::UInt8
end

function convert_lightgraphs_edges_to_node_edges(g_edges::Array{LightGraphs.SimpleGraphs.SimpleEdge{Int64},1})::Array{NodeEdge, 1}
    node_edges = Array{NodeEdge, 1}([])
    for edge in g_edges
        push!(node_edges, NodeEdge(edge.src, edge.dst))
    end
    return node_edges
end

######################## ForbiddenPair #########################
# comp₁ < comp₂ by requirement
struct ForbiddenPair
    comp₁::UInt8
    comp₂::UInt8
end

function Base.:(==)(fp_1::ForbiddenPair, fp_2::ForbiddenPair)
    (fp_1.comp₁ == fp_2.comp₁) && (fp_1.comp₂ == fp_2.comp₂)
end

function Base.isless(fp_1::ForbiddenPair, fp_2::ForbiddenPair)
    fp_1.comp₁ < fp_2.comp₁
end

Base.hash(fp::ForbiddenPair, h::UInt) = hash(fp.comp₁, hash(fp.comp₂, hash(:ForbiddenPair, h)))

#######################   Node   ##################################

# The 1st node is always the 0-terminal, and the 2nd node is always the 1 terminal. Adding the first node to the ZDD
# means the ZDD will have 3 nodes, the node + the two terminal nodes
mutable struct Node
    label::NodeEdge
    comp::Vector{UInt8}       # can hold 256 possible values
    comp_weights::Vector{UInt8} # the max population of a component can only be 256
    cc::UInt8                   # can hold only 256 possible values
    fps::Vector{ForbiddenPair}
    comp_assign::Vector{UInt8}  # only 256 possible values
    deadend::Bool

    # allow for incomplete initialization
    function Node()::Node
        new()
    end

    function Node(i::Int)::Node # for Terminal Nodes
        return new(NodeEdge(i, i), Vector{UInt8}(), Vector{UInt8}(), 0, Vector{ForbiddenPair}(), Vector{UInt8}([]), true)
    end

    function Node(root_edge::NodeEdge, base_graph::SimpleGraph)::Node
        comp_assign = Vector{UInt8}([i for i in 1:nv(base_graph)])
        comp_weights = Vector{UInt8}([1 for i in 1:nv(base_graph)]) # initialize each vertex's population to be 1.
        return new(root_edge, Vector{UInt8}(), comp_weights, 0, Vector{ForbiddenPair}(), comp_assign, true)
    end

    function Node(label::NodeEdge, comp::Vector{UInt8}, comp_weights::Vector{UInt8},
                  cc::UInt8, fps::Vector{ForbiddenPair}, comp_assign::Vector{UInt8}, deadend::Bool)::Node
        return new(label, comp, comp_weights, cc, fps, comp_assign, deadend)
    end
end

function copy_to_vec!(vec₁::Vector{T}, vec₂::Vector{T}) where T
    """ Copy items from vec₁ into vec₂.
        It is assumed that length(vec₂) >= length(vec₁)
    """
    for (i, item) in enumerate(vec₁)
        @inbounds vec₂[i] = item
    end
end

function copy_to_set!(set₁::Set{T}, set₂::Set{T}) where T
    for item in set₁
        push!(set₂, item)
    end
end

function custom_deepcopy(n::Node)::Node
    comp = Vector{UInt8}(undef, length(n.comp))
    comp_weights = Vector{UInt8}(undef, length(n.comp_weights))
    comp_assign = Vector{UInt8}(undef, length(n.comp_assign))
    fps = Vector{ForbiddenPair}(undef, length(n.fps))
    # fps = Set{ForbiddenPair}()

    copy_to_vec!(n.comp, comp)
    copy_to_vec!(n.comp_weights, comp_weights)
    copy_to_vec!(n.comp_assign, comp_assign)
    copy_to_vec!(n.fps, fps)

    return Node(n.label, comp, comp_weights, n.cc, fps, comp_assign, true)
end

function Base.:(==)(node₁::Node, node₂::Node)
    node₁.cc == node₂.cc &&
    node₁.comp_weights == node₂.comp_weights &&
    node₁.label == node₂.label &&
    node₁.comp == node₂.comp &&
    node₁.fps == node₂.fps &&
    node₁.comp_assign == node₂.comp_assign
end

Base.hash(n::Node, h::UInt) = hash(n.label, hash(n.comp, hash(n.cc, hash(n.fps, hash(n.comp_weights, hash(n.comp_assign, hash(:Node, h)))))))

function node_summary(node::Node)
    println("Label: ", readable(node.label))
    println("cc: ", readable(node.cc))
    println("comp: ", readable(node.comp))
    println("fps: ", readable(node.fps))
    println("comp_assign: ", readable(node.comp_assign))
    println("comp_weights: ", readable(node.comp_weights))
    println()
end

function readable(edge::NodeEdge)::String
    "NodeEdge(" * string(Int64(edge.edge₁)) * " -> " * string(Int64(edge.edge₂)) * ")"
end

function readable(comp::Array{UInt8, 1})::Array{Int64, 1}
    Array{Int, 1}([Int64(x) for x in comp])
end

function readable(cc::UInt8)::Int64
    Int64(cc)
end

function readable(fp::ForbiddenPair)::String
    "ForbiddenPair(" * string(Int64(fp.comp₁)) * " -> " * string(Int64(fp.comp₂)) * ")"
end

function readable(fp_set::Set{ForbiddenPair})::Set{String}
    readable_set = Set{String}([])
    for fp in fp_set
        push!(readable_set, readable(fp))
    end
    readable_set
end

function readable(comp_weights::Dict{Int8, Int8})::Dict{Int, Int}
    readable_dict = Dict{Int, Int}()
    for (k, v) in comp_weights
        readable_dict[Int(k)] = v
    end
    readable_dict
end
