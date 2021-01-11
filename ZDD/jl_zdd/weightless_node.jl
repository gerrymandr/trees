# node.jl
import Base: isequal, ==

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

Base.hash(fp::ForbiddenPair, h::UInt) = hash(fp.comp₁, hash(fp.comp₂, hash(:ForbiddenPair, h)))

#######################   Node   ##################################

# The 1st node is always the 0-terminal, and the 2nd node is always the 1 terminal. Adding the first node to the ZDD
# means the ZDD will have 3 nodes, the node + the two terminal nodes
mutable struct Node
    label::NodeEdge
    comp::Array{UInt8, 1} # can hold 256 possible values
    cc::UInt8 # can hold only 256 possible values
    fps::Set{ForbiddenPair}
    comp_assign::Vector{UInt8} # only 256 possible values
end

function Node(i::Int)::Node # for Terminal Nodes
    return Node(NodeEdge(i, i), Array{UInt8, 1}(), 0, Set{ForbiddenPair}(), Vector{UInt8}([]))
end

function Node(root_edge::NodeEdge, base_graph::SimpleGraph)::Node
    comp_assign = Vector{UInt8}([i for i in 1:nv(base_graph)])
    return Node(root_edge, Array{UInt8, 1}(), 0, Set{ForbiddenPair}(), comp_assign)
end

function Base.:(==)(node₁::Node, node₂::Node)
    node₁.cc == node₂.cc &&
    node₁.label == node₂.label &&
    node₁.comp == node₂.comp &&
    node₁.fps == node₂.fps &&
    node₁.comp_assign == node₂.comp_assign
end

Base.hash(n::Node, h::UInt) = hash(n.label, hash(n.comp, hash(n.cc, hash(n.fps, hash(n.comp_assign, hash(:Node, h))))))

function node_summary(node::Node)
    println("Label: ", node.label)
    println("cc: ",node.cc)
    println("comp: ",node.comp)
    println("fps: ",node.fps)
    println("comp_assign: ", node.comp_assign)
    println()
end
