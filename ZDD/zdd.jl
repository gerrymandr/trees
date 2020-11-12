using DataStructures
using LightGraphs
import Base: isequal

# The 1st node is always the 0-terminal, and the 2nd node is always the 1 terminal. Adding the first node to the ZDD
# means the ZDD will have 3 nodes, the node + the two terminal nodes
abstract type NodeZDD end

mutable struct Node<:NodeZDD
    label::AbstractEdge
    comp::Set{Set{Int}}
    cc::Int
    fps::Set{Set{Set{Int}}}
end

struct TerminalNode<:NodeZDD
    label::Int
end

function Node(root_edge::AbstractEdge)
    return Node(root_edge, Set{Set{Int}}(), 0, Set{Set{Set{Int}}}())
end

mutable struct ZDD
    # graph stuff
    graph::SimpleGraph
    nodes::Dict{N, Int64} where N <: NodeZDD
    edges::Dict{Tuple{N, N}, Int64} where N <: NodeZDD
    base_graph::SimpleGraph
end

function ZDD(g::SimpleGraph, root::Node)
    graph = SimpleGraph(3) # 2 for terminal nodes and 1 for the root
    nodes = Dict{NodeZDD, Int64}()
    nodes[TerminalNode(0)] = 1
    nodes[TerminalNode(1)] = 2
    nodes[root] = 3
    edges = Dict{Tuple{NodeZDD,NodeZDD},Int64}()
    base_graph = g
    return ZDD(graph, nodes, edges, base_graph)
end

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
    for node in keys(zdd.nodes)
        if node isa TerminalNode
            continue
        end
        if node.label in keys(labels_seen)
            labels_seen[node.label] += 1
        else
            labels_seen[node.label] = 1
        end
        loc_xs[zdd.nodes[node]] = labels_seen[node.label] * (tree_width / (label_occs[node.label] + 1))
        loc_ys[zdd.nodes[node]] = findfirst(y -> y == node.label, collect(edges(g)))
    end

    loc_xs = Float64.(loc_xs)
    loc_ys = Float64.(loc_ys)

    return loc_xs, loc_ys
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

function add_zdd_edge!(zdd::ZDD, zdd_edge::Tuple{NodeZDD, NodeZDD}, x::Int)
    """ zdd_edge is represented as (Node, Node)
    """
    @assert x in [0, 1]
    node₁, node₂ = zdd_edge
    zdd.edges[(node₁, node₂)] = x

    # get node indexes
    node₁_idx = zdd.nodes[node₁]
    node₂_idx = zdd.nodes[node₂]

    # add to simple graph
    add_edge!(zdd.graph, (node₁_idx, node₂_idx))
end

function construct_zdd(g::SimpleGraph, k::Int)
    # select root
    g_edges = collect(edges(g))
    root = Node(g_edges[1], Set(), 0, Set())

    zdd = ZDD(g, root)
    N = [Set{NodeZDD}() for a in 1:ne(g)+1]
    N[1] = Set([root])
    g_edges = collect(edges(g))

    for i = 1:ne(g)
        # println("i ", i)
        for n in N[i]
            # println("N[i] ", N[i])
            # println("n ", n)
            for x in [0, 1]
                # println("x ", x)
                n′ = make_new_node(g_edges, [k], n, i, x)
                # println("n′ : ", n′)

                if !(n′ isa TerminalNode)
                    # println("Not Terminal")
                    # update the label of n′
                    n′.label = g_edges[i+1]
                    found_copy = false

                    for n′′ in N[i+1]
                        if isequal(n′′, n′)
                            # n′ = n′′
                            found_copy = true
                            # print("copy!")
                            # add_zdd_node!(zdd, n′′)
                            add_zdd_edge!(zdd, (n, n′′), x)
                            break
                        end
                    end
                    if !found_copy
                        # println("Pushing!")
                        # println("N[i+1] before: ", N[i+1])
                        push!(N[i+1], n′)
                        add_zdd_node!(zdd, n′)
                        add_zdd_edge!(zdd, (n, n′), x)
                        # println("N[i+1] after: ", N[i+1])
                    end

                else
                    add_zdd_node!(zdd, n′)
                    add_zdd_edge!(zdd, (n, n′), x)
                end

                # println("Adding ZDD Node!")
                # println("ZDD before: ", zdd)
                # add_zdd_node!(zdd, n′)
                # println("ZDD after: ", zdd)

                # println("Adding ZDD edge")
                # println("ZDD before: ", zdd)
                # add_zdd_edge!(zdd, (n, n′), x)
                # println("ZDD after: ", zdd)
                # println()
            end
        end
    end
    return zdd
end

function add_vertex_to_component!(n′::Node, u::Int, v::Int, prev_frontier::Set{Int})
    """ Add `u` or `v` or both to n`.comp if they are not in
        `prev_frontier`
    """
    for vertex in [u, v]
        if vertex ∉ prev_frontier
            push!(n′.comp, Set([vertex]))
        end
    end
end

function make_new_node(g_edges, K, n, i, x)
    """
    """
    u = g_edges[i].src
    v = g_edges[i].dst

    n′ = deepcopy(n)

    prev_frontier, curr_frontier = compute_frontiers(g_edges, i)

    add_vertex_to_component!(n′, u, v, prev_frontier)
    Cᵤ, Cᵥ = components(u, v, n′.comp)

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
            if Set([a]) in n′.comp
                n′.cc += 1
                if n′.cc > maximum(K)
                    return TerminalNode(0)
                end
            end

            remove_vertex_from_node_component!(n′, a)
            remove_vertex_from_node_fps!(n′, a)
        end
    end

    if i == length(g_edges)
        if n′.cc in K
            return TerminalNode(1)
        else
            return TerminalNode(0)
        end
    end

    return n′
end

function replace_components_with_union!(node::Node, Cᵤ::Set{Int}, Cᵥ::Set{Int})
    """
    """
    for fp in node.fps
        for comp in fp
            if comp == Cᵤ || comp == Cᵥ
                pop!(fp, comp)
                push!(fp, Set(union(Cᵤ, Cᵥ)))
            end
        end
    end
end

function connect_components!(n::Node, Cᵤ::Set{Int}, Cᵥ::Set{Int})
    """
    """
    if Cᵤ in n.comp
        pop!(n.comp, Cᵤ)
    end
    if Cᵥ in n.comp
        pop!(n.comp, Cᵥ)
    end
    n.comp = union(n.comp, Set([union(Cᵤ, Cᵥ)])) # we connect the components
end

function components(u::Int, v::Int, components::Set{Set{Int}})::Tuple{Set{Int}, Set{Int}}
    """ Returns Cᵤ and Cᵥ which are the sets in `components` that contain
        vertices `u` and `v` respectively.
    """
    Cᵤ = Set{Int}([])
    Cᵥ = Set{Int}([])
    for s in components
        if u in s
            Cᵤ = s
        end
        if v in s
            Cᵥ = s
        end
    end
    return Cᵤ, Cᵥ
end

function compute_frontiers(edges, i)
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

function isequal(node₁::Node, node₂::Node)::Bool
    """
    """
    # min(node₁.label.src, node₁.label.dst) == min(node₂.label.src, node₂.label.dst) &&
    # max(node₁.label.src, node₁.label.dst) == max(node₂.label.src, node₂.label.dst) &&
    issetequal(node₁.comp, node₂.comp) &&
    node₁.cc == node₂.cc &&
    issetequal(node₁.fps, node₂.fps)
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
    delete!(node.comp, Set([vertex]))
    for comp in node.comp
        delete!(comp, vertex)
    end
end

function remove_vertex_from_node_fps!(node::Node, vertex::Int)
    """
    """
    for comp in node.comp
        delete!(node.fps, Set([Set([vertex]), comp]))
    end
    for fp in node.fps
        for comp in fp
            delete!(comp, vertex)
        end
    end
end

function add_zdd_node!(zdd::ZDD, node::N) where N <: NodeZDD
    """
    """
    if node ∉ keys(zdd.nodes)
        add_vertex!(zdd.graph)
        zdd.nodes[node] = nv(zdd.graph)
    end
end
