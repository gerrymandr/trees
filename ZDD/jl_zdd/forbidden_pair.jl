
import Base: isequal, ==, isless, Dict

###################### NodeEdge ##############################
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

function convert_node_edges_to_lightgraphs_edges(node_edges::Vector{NodeEdge})::Array{LightGraphs.SimpleGraphs.SimpleEdge{Int64},1}
    lg_edges = Array{LightGraphs.SimpleGraphs.SimpleEdge{Int64},1}([])
    for node_edge in node_edges
        push!(lg_edges, LightGraphs.SimpleGraphs.SimpleEdge{Int64}(node_edge.edge₁, node_edge.edge₂))
    end
    return lg_edges
end

######################## ForbiddenPair #########################

# comp₁ < comp₂ by requirement
struct ForbiddenPair
    comp₁::UInt8
    comp₂::UInt8
end

==(p::ForbiddenPair, q::ForbiddenPair) = (p.comp₁==q.comp₁) & (p.comp₂==q.comp₂)

isequal(p::ForbiddenPair, q::ForbiddenPair) = isequal(p.comp₁,q.comp₁) & isequal(p.comp₂,q.comp₂)
isless(p::ForbiddenPair, q::ForbiddenPair) = ifelse(!isequal(p.comp₁,q.comp₁), isless(p.comp₁,q.comp₁), isless(p.comp₂,q.comp₂))

Base.hash(fp::ForbiddenPair, h::UInt) = hash(fp.comp₁, hash(fp.comp₂, h))
