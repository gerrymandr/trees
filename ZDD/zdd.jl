using DataStructures
using LightGraphs
import Base: isequal, ==

struct ForbiddenPair
    comp₁::Int
    comp₂::Int
end

# The 1st node is always the 0-terminal, and the 2nd node is always the 1 terminal. Adding the first node to the ZDD
# means the ZDD will have 3 nodes, the node + the two terminal nodes
abstract type NodeZDD end

mutable struct Node<:NodeZDD
    label::AbstractEdge
    comp::Set{Int}
    cc::Int
    fps::Set{Set{Int}}
    comp_assign::Array{Int}
end

struct TerminalNode<:NodeZDD
    label::Int
end

function Node(root_edge::AbstractEdge, base_graph::SimpleGraph)
    comp_assign = [i for i in 1:nv(base_graph)]
    return Node(root_edge, Set{Int}(), 0, Set{Set{Int}}(), comp_assign)
end

mutable struct ZDD
    # graph stuff
    graph::SimpleDiGraph
    nodes::Dict{N, Int64} where N <: NodeZDD
    edges::Dict{Tuple{N, N}, Int64} where N <: NodeZDD
    edge_multiplicity::Set{Tuple{N, N}} where N <: NodeZDD
    base_graph::SimpleGraph
    root::Node
end

function get_node_from_seq(zdd::ZDD, seq::Array{Int})
    curr_node = zdd.root
    g_edges = collect(edges(g))

    for (i, x) in enumerate(seq)
        if i == 1
            for node in keys(zdd.nodes)
                if (zdd.root, node) in keys(zdd.edges) && node.label == g_edges[i+1] && zdd.edges[zdd.root, node] == x
                    curr_node = node
                    break
                end
            end
        else
            for node in keys(zdd.nodes)
                if (node, curr_node) in keys(zdd.edges) && node.label == g_edges[i+1] && zdd.edges[node, curr_node] == x
                    curr_node = node
                    break
                elseif (curr_node, node) in keys(zdd.edges) && node.label == g_edges[i+1] && zdd.edges[curr_node, node] == x
                    curr_node = node
                    break
                end
            end
        end
    end
    return curr_node
end

function ZDD(g::SimpleGraph, root::Node)
    graph = SimpleDiGraph(3) # 2 for terminal nodes and 1 for the root
    nodes = Dict{NodeZDD, Int64}()
    nodes[TerminalNode(0)] = 1
    nodes[TerminalNode(1)] = 2
    nodes[root] = 3
    edges = Dict{Tuple{NodeZDD,NodeZDD},Int64}()
    edge_multiplicity = Set{Tuple{NodeZDD,NodeZDD}}()
    base_graph = g
    return ZDD(graph, nodes, edges, edge_multiplicity, base_graph, root)
end

function Base.:(==)(node₁::Node, node₂::Node)
    node₁.cc == node₂.cc &&
    node₁.label == node₂.label &&
    # min(node₁.label.src, node₁.label.dst) == min(node₂.label.src, node₂.label.dst) &&
    # max(node₁.label.src, node₁.label.dst) == max(node₂.label.src, node₂.label.dst) &&
    # node₁.label == node₂.label
    issetequal(node₁.comp, node₂.comp) &&
    issetequal(deepcopy(node₁.fps), deepcopy(node₂.fps))
    # node₁.fps == node₂.fps
end

# function fps_equality(fps₁, fps₂)
#     counter = 0
#     for fp in fps₁
#         if fp in fps₂
#             counter += 1
#         else
#             return false
#         end
#     end
#     return counter == length(fps₂)
# end

function add_zdd_edge!(zdd::ZDD, zdd_edge::Tuple{NodeZDD, NodeZDD}, x::Int)
    """ zdd_edge is represented as (Node, Node)
    """
    @assert x in [0, 1]
    node₁, node₂ = zdd_edge

    if (node₁, node₂) in keys(zdd.edges)
        push!(zdd.edge_multiplicity, (node₁, node₂))
    else
        zdd.edges[(node₁, node₂)] = x
    end

    # get node indexes
    node₁_idx = zdd.nodes[node₁]
    node₂_idx = zdd.nodes[node₂]

    # add to simple graph
    add_edge!(zdd.graph, (node₁_idx, node₂_idx))
end

function num_edges(zdd::ZDD)
    length(zdd.edges) + length(zdd.edge_multiplicity)
end

function construct_zdd(g::SimpleGraph, k::Int)
    # select root
    g_edges = collect(edges(g))
    root = Node(g_edges[1], g)

    zdd = ZDD(g, root)
    N = [Set{NodeZDD}() for a in 1:ne(g)+1]
    N[1] = Set([root])
    g_edges = collect(edges(g))

    for i = 1:ne(g)
        for n in N[i]
            for x in [0, 1]
                n′ = make_new_node(g_edges, k, n, i, x)

                if !(n′ isa TerminalNode)
                    n′.label = g_edges[i+1] # update the label of n′
                    found_copy = false

                    for n′′ in N[i+1]
                        # println("Check")
                        # node_summary(n′)
                        # node_summary(n′′)
                        # println(n′′ == n′)
                        # println(n′′.cc == n′.cc)
                        # println(n′′.label == n′.label)
                        # println(issetequal(n′′.comp, n′.comp))
                        # println(issetequal(n′′.fps, n′.fps))
                        if n′′ == n′
                            n′ = n′′
                            found_copy = true
                            break
                        end
                    end
                    if !found_copy
                        push!(N[i+1], n′)
                    end
                end

                add_zdd_node!(zdd, n′)
                add_zdd_edge!(zdd, (n, n′), x)
            end
        end
    end
    return zdd
end

function node_summary(node::Node)
    println("Label: ", node.label)
    println("cc: ",node.cc)
    println("comp: ",node.comp)
    println("fps: ",node.fps)
    println("comp_assign: ", node.comp_assign)
    println()
end

function make_new_node(g_edges, k::Int, n::NodeZDD, i::Int, x::Int)
    """
    """
    u = g_edges[i].src
    v = g_edges[i].dst

    n′ = deepcopy(n)
    prev_frontier, curr_frontier = compute_frontiers(g_edges, i)

    add_vertex_as_component!(n′, u, v, prev_frontier)
    Cᵤ, Cᵥ = components(u, v, n′)

    if x == 1
        connect_components!(n′, Cᵤ, Cᵥ)

        if Cᵤ != Cᵥ && Set([Cᵤ, Cᵥ]) in n′.fps
            return TerminalNode(0)
        else
            replace_components_with_union!(n′, Cᵤ, Cᵥ)
        end
    else
        if Cᵤ == Cᵥ
            return TerminalNode(0)
        else
            push!(n′.fps, Set([Cᵤ, Cᵥ]))
        end
    end

    for a in [u, v]
        if a ∉ curr_frontier
            a_comp = n′.comp_assign[a]

            if a_comp in n′.comp && length(filter(x -> x == a_comp, n′.comp_assign)) == 1
                n′.cc += 1
                if n′.cc > k
                    return TerminalNode(0)
                end
            end
            # remove_vertex_from_node_component!(n′, a)
            remove_vertex_from_node_fps!(n′, a)
        end
    end

    if i == length(g_edges)
        if n′.cc == k
            return TerminalNode(1)
        else
            return TerminalNode(0)
        end
    end

    return n′
end

function add_vertex_as_component!(n′::Node, u::Int, v::Int, prev_frontier::Set{Int})
    """ Add `u` or `v` or both to n`.comp if they are not in
        `prev_frontier`
    """
    for vertex in [u, v]
        if vertex ∉ prev_frontier
            push!(n′.comp, vertex)
        end
    end
end

function replace_components_with_union!(node::Node, Cᵤ::Int, Cᵥ::Int)
    """
    """
    assignment = maximum([Cᵤ, Cᵥ])
    to_change = minimum([Cᵤ, Cᵥ])
    for fp in node.fps
        if to_change in fp
            pop!(fp, to_change)
            push!(fp, assignment)
        end
    end
end

function connect_components!(n::Node, Cᵤ::Int, Cᵥ::Int)
    """
    """
    assignment = maximum([Cᵤ, Cᵥ])
    to_change = minimum([Cᵤ, Cᵥ])
    if Cᵤ != Cᵥ
        n.comp_assign = map(val -> val == to_change ? assignment : val, n.comp_assign) # TODO: replace with inplace
        delete!(n.comp, to_change)
    end
end

function components(u::Int, v::Int, node::Node)::Tuple{Int, Int}
    """ Returns Cᵤ and Cᵥ which are the sets in `components` that contain
        vertices `u` and `v` respectively.
    """
    return node.comp_assign[u], node.comp_assign[v]
end

function compute_frontiers(edges, i::Int)
    """
    """
    m = length(edges)
    if i == 1
        prev_frontier = Set{Int}()
    else
        processed_nodes = nodes_from_edges(edges[1:i-1])
        unprocessed_nodes = nodes_from_edges(edges[i:m])
        prev_frontier = intersect(Set(processed_nodes), Set(unprocessed_nodes))
    end

    if i == m
        curr_frontier = Set{Int}()
    else
        processed_nodes = nodes_from_edges(edges[1:i])
        unprocessed_nodes = nodes_from_edges(edges[i+1:m])
        curr_frontier = intersect(Set(processed_nodes), Set(unprocessed_nodes))
    end

    return prev_frontier, curr_frontier
end

function nodes_from_edges(edges)::Set{Int}
    """
    """
    nodes = Set{Int}()
    for e in edges
        push!(nodes, e.src)
        push!(nodes, e.dst)
    end
    return nodes
end

function isequal(edge₁::AbstractEdge, edge₂::AbstractEdge)::Bool
    """ TODO: change this to edge_1 and edge_2
    """
    min(edge₁.src, edge₁.dst) == min(edge₂.src, edge₂.dst) &&
    max(edge₁.src, edge₁.dst) == max(edge₂.src, edge₂.dst)
end

function remove_vertex_from_node_component!(node::Node, vertex::Int)
    """ Removes all occurences of `vertex` from `node`.comp
    """
    node.comp_assign[vertex] = 0
end

function remove_vertex_from_node_fps!(node::Node, vertex::Int)
    """
    """
    vertex_comp = node.comp_assign[vertex]

    for fp in node.fps
        if vertex_comp in fp && length(filter(x -> x == vertex_comp, node.comp_assign)) == 1
            delete!(node.fps, fp)
            delete!(node.comp, vertex_comp)
        end
    end

    node.comp_assign[vertex] = 0
end

function add_zdd_node!(zdd::ZDD, node::N) where N <: NodeZDD
    """
    """
    if node ∉ keys(zdd.nodes)
        # println("adding vertex")
        # node_summary(node)
        add_vertex!(zdd.graph)
        zdd.nodes[node] = nv(zdd.graph)
    end
end

"""
Visualization
"""

function draw(zdd::ZDD, g)
    """
    """
    e_colors = edge_colors(zdd)
    node_labels = label_nodes(zdd)
    loc_xs, loc_ys = node_locations(zdd)

    node_colors = fill("lightseagreen", nv(zdd.graph))
    node_colors[[1, 2]] = ["orange", "orange"]

    gplot(zdd.graph, loc_xs, loc_ys,
          nodefillc=node_colors,
          nodelabel=node_labels,
          edgestrokec=e_colors)
end

function edge_colors(zdd)
    edge_colors = []
    for edge in edges(zdd.graph)
        node₁ = get_corresponding_node(zdd, edge.src)
        node₂ = get_corresponding_node(zdd, edge.dst)
        try
            push!(edge_colors, zdd.edges[(node₁, node₂)])
        catch e
            push!(edge_colors, zdd.edges[(node₂, node₁)])
        end
    end

    colors = []
    for e in edge_colors
        if e == 0
            push!(colors, "red")
        else
            push!(colors, "green")
        end
    end
    return colors
end

function node_locations(zdd)
    """
    """
    label_occs = label_occurences(zdd)

    tree_depth = length(keys(label_occs)) - 1 # - 1 because the two terminals will be on the same depth
    tree_width = maximum(values(label_occs))
    loc_xs = fill(-1., nv(zdd.graph))
    loc_xs[[1, 2]] = [tree_width/3 , tree_width*2/3]
    loc_ys = fill(-1., nv(zdd.graph))
    loc_ys[[1, 2]] = [tree_depth, tree_depth]

    labels_seen = Dict{AbstractEdge, Int}()
    add_locations(zdd::ZDD, zdd.root, loc_xs, loc_ys, tree_width, label_occs, labels_seen)

    loc_xs = Float64.(loc_xs)
    loc_ys = Float64.(loc_ys)

    return loc_xs, loc_ys
end

function node_by_idx(zdd::ZDD, idx)
    for (node, val) in zdd.nodes
        if val == idx
            return node
        end
    end
end

function add_locations(zdd::ZDD, node, loc_xs, loc_ys, tree_width, label_occs, labels_seen)
    """
    """
    if node isa TerminalNode
        return
    end

    if loc_xs[zdd.nodes[node]] == -1
        if node.label in keys(labels_seen)
            labels_seen[node.label] += 1
        else
            labels_seen[node.label] = 1
        end

        loc_xs[zdd.nodes[node]] = labels_seen[node.label] * (tree_width / (label_occs[node.label] + 1))
        loc_ys[zdd.nodes[node]] = findfirst(y -> y == node.label, collect(edges(zdd.base_graph)))
    end

    ns = outneighbors(zdd.graph, zdd.nodes[node])
    neighbors = []
    for n in ns
        push!(neighbors, node_by_idx(zdd, n))
    end

    if length(neighbors) == 1
        add_locations(zdd, neighbors[1], loc_xs, loc_ys, tree_width, label_occs, labels_seen)
        return

    elseif length(neighbors) == 2
        order_1 = (node, neighbors[1])
        order_2 = (node, neighbors[2])

        try
            zdd.edges[order_1]
        catch e
            order_1 = (neighbors[1], node)
        end

        try
            zdd.edges[(node, neighbors[2])]
        catch e
            order_1 = (neighbors[2], node)
        end

        if zdd.edges[order_1] == 0 && zdd.edges[order_2] == 1
            neighbors = [neighbors[1], neighbors[2]]
        else
            neighbors = [neighbors[2], neighbors[1]]
        end

        for neighbor in neighbors
            add_locations(zdd, neighbor, loc_xs, loc_ys, tree_width, label_occs, labels_seen)
        end
    else
        println("THIS SHOULD NOT HAPPEN")
    end
end

function label_occurences(zdd::ZDD)
    """
    """
    occs = Dict{Union{AbstractEdge, Int}, Int}()
    for node in keys(zdd.nodes)
        if node.label in keys(occs)
            occs[node.label] += 1
        else
            occs[node.label] = 1
        end
    end
    return occs
end

function get_corresponding_node(zdd, node_id)
    for (k , v) in zdd.nodes
        if v == node_id
            return k
        end
    end
end

function label_nodes(zdd::ZDD)
    """
    """
    node_labels = Array{Union{AbstractEdge, Int}, 1}(undef, nv(zdd.graph))
    for node in keys(zdd.nodes)
        node_labels[zdd.nodes[node]] = node.label
    end
    node_labels
end
