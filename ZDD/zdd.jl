using DataStructures
using LightGraphs
import Base: isequal, ==

# comp₁ < comp₂ by requirement
mutable struct ForbiddenPair
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
    fps::Set{ForbiddenPair}
    comp_assign::Array{Int}
end

struct TerminalNode<:NodeZDD
    label::Int
end

function Node(root_edge::AbstractEdge, base_graph::SimpleGraph)
    comp_assign = [i for i in 1:nv(base_graph)]
    return Node(root_edge, Set{Int}(), 0, Set{ForbiddenPair}(), comp_assign)
end

mutable struct ZDD
    # graph stuff
    graph::SimpleDiGraph
    nodes::Dict{N, Int64} where N <: NodeZDD
    edges::Dict{Tuple{N, N}, Int64} where N <: NodeZDD
    edge_multiplicity::Set{Tuple{N, N}} where N <: NodeZDD
    base_graph::SimpleGraph
    root::Node
    paths::Dict{N, Int64} where N <: NodeZDD
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
    paths = Dict{NodeZDD, Int64}()
    paths[TerminalNode(0)] = 0
    paths[TerminalNode(1)] = 1
    paths[root] = -1
    return ZDD(graph, nodes, edges, edge_multiplicity, base_graph, root, paths)
end

function Base.:(==)(node₁::Node, node₂::Node)
    node₁.cc == node₂.cc &&
    node₁.label == node₂.label &&
    node₁.comp == node₂.comp &&
    # node₁.fps == node₂.fps
    fps_equality(node₁.fps, node₂.fps)
end

function Base.:(==)(node₁::TerminalNode, node₂::Node)
    false
end

function Base.:(==)(node₁::Node, node₂::TerminalNode)
    false
end

function Base.:(==)(fp_1::ForbiddenPair, fp_2::ForbiddenPair)
    (fp_1.comp₁ == fp_2.comp₁) && (fp_1.comp₂ == fp_2.comp₂)
end

Base.hash(fp::ForbiddenPair, h::UInt) = hash(fp.comp₁, hash(fp.comp₂, hash(:ForbiddenPair, h)))
Base.hash(n::Node, h::UInt) = hash(n.label, hash(n.comp, hash(n.cc, hash(n.fps, hash(:Node, h)))))
Base.hash(n::TerminalNode, h::UInt) = hash(n.label, hash(:TerminalNode, h))

function fps_equality(fps₁, fps₂)
    if length(fps₁) != length(fps₂)
        return false
    end

    counter = 0
    for fpᵢ in fps₁
        found = false
        # if fpᵢ ∉ fps₂
        #     return false
        #     # counter += 1
        # # else
        #     # return false
        # end
        for fpⱼ in fps₂
            if fpᵢ == fpⱼ
                counter += 1
                found = true
                break
            end
        end
        if !found
            return false
        end
    end

    # return counter == length(fps₂)
    true
end


function add_zdd_edge!(zdd::ZDD, zdd_edge::Tuple{NodeZDD, NodeZDD}, x::Int)
    """ zdd_edge is represented as (Node, Node)
    """
    node₁, node₂ = zdd_edge
    # tup = (node₁, node₂)

    if zdd_edge in keys(zdd.edges)
        push!(zdd.edge_multiplicity, zdd_edge)
    else
        zdd.edges[zdd_edge] = x
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

function construct_zdd(g::SimpleGraph, k::Int, g_edges::Array{LightGraphs.SimpleGraphs.SimpleEdge{Int64},1})
    # optimal_ordering ? g_edges = optimal_grid_edge_order(g, dims[1],dims[2]) : g_edges = collect(edges(g))
    frontier_distribution(g, g_edges)
    # select root
    root = Node(g_edges[1], g)

    zdd = ZDD(g, root)
    N = [Set{NodeZDD}() for a in 1:ne(g)+1]
    N[1] = Set([root])
    frontiers = compute_all_frontiers(g, g_edges)

    for i = 1:ne(g)
        for n in N[i]
            for x in [0, 1]
                n′ = make_new_node(g_edges, k, n, i, x, frontiers)

                if !(n′ isa TerminalNode)
                    n′.label = g_edges[i+1] # update the label of n′
                    found_copy = false

                    for n′′ in N[i+1]
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
    calculate_paths!(zdd)
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

function make_new_node(g_edges, k::Int, n::NodeZDD, i::Int, x::Int, frontiers::Array{Set{Int}, 1})
    """
    """
    u = g_edges[i].src
    v = g_edges[i].dst

    n′ = deepcopy(n)
    prev_frontier, curr_frontier = frontiers[i], frontiers[i+1]

    add_vertex_as_component!(n′, u, v, prev_frontier)
    Cᵤ, Cᵥ = components(u, v, n′)

    if x == 1
        connect_components!(n′, Cᵤ, Cᵥ)

        if Cᵤ != Cᵥ && ForbiddenPair(min(Cᵤ, Cᵥ), max(Cᵤ, Cᵥ)) in n′.fps
            return TerminalNode(0)
        else
            replace_components_with_union!(n′, Cᵤ, Cᵥ)
        end
    else
        if Cᵤ == Cᵥ
            return TerminalNode(0)
        else
            push!(n′.fps, ForbiddenPair(min(Cᵤ, Cᵥ), max(Cᵤ, Cᵥ)))
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
        if to_change == fp.comp₁
            fp.comp₁ = assignment
        elseif to_change == fp.comp₂
            fp.comp₂ = assignment
        end
        if fp.comp₁ > fp.comp₂
            flip!(fp)
        end
    end
end

function flip!(fp::ForbiddenPair)
    holder = fp.comp₁
    fp.comp₁ = fp.comp₂
    fp.comp₂ = holder
end

function connect_components!(n::Node, Cᵤ::Int, Cᵥ::Int)
    """
    """
    assignment = maximum([Cᵤ, Cᵥ])
    to_change = minimum([Cᵤ, Cᵥ])
    if Cᵤ != Cᵥ
        map!(val -> val == to_change ? assignment : val, n.comp_assign, n.comp_assign)
        delete!(n.comp, to_change)
    end
end

function components(u::Int, v::Int, node::Node)::Tuple{Int, Int}
    """ Returns Cᵤ and Cᵥ which are the sets in `components` that contain
        vertices `u` and `v` respectively.
    """
    return node.comp_assign[u], node.comp_assign[v]
end

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

function compute_all_frontiers(g::SimpleGraph, g_edges)
    frontiers = [Set{Int}() for a in 1:ne(g)+1]
    for i ∈ 1:ne(g)+1
        frontiers[i] = compute_frontier(g_edges, i)
    end
    return frontiers
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
        if (vertex_comp == fp.comp₁ || vertex_comp == fp.comp₂) && length(filter(x -> x == vertex_comp, node.comp_assign)) == 1
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
        add_vertex!(zdd.graph)
        zdd.nodes[node] = nv(zdd.graph)
        zdd.paths[node] = -1
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
        if haskey(zdd.edges, (node₁, node₂))
            push!(edge_colors, zdd.edges[(node₁, node₂)])
        elseif haskey(zdd.edges, (node₂, node₁))
            push!(edge_colors, zdd.edges[(node₂, node₁)])
        # else
        #     println("whoop")
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

function queen_grid(dims)
    g = grid(dims)
    r = dims[1]
    c = dims[2]
    for n ∈ 1:r*c
        if n == 1
            add_edge!(g, n, c+1)
        elseif n == r
            add_edge!(g, n, n+r-1)
        elseif n == r*c
            add_edge!(g, n, n-r-1)
        elseif n == r*(c-1)+1
            add_edge!(g, n, n-r+1)
        elseif n ∈ 1:c
            add_edge!(g, n, n+r-1)
            add_edge!(g, n, n+r+1)
        elseif (n-1) % r == 0
            add_edge!(g, n, n-r+1)
            add_edge!(g, n, n+r+1)
        elseif n % r == 0
            add_edge!(g, n, n-r-1)
            add_edge!(g, n, n+r-1)
        elseif n ∈ c*r:-1:(r-1)c+1
            add_edge!(g, n, n-r+1)
            add_edge!(g, n, n-r-1)
        else
            add_edge!(g, n, n+r-1)
            add_edge!(g, n, n+r+1)
            add_edge!(g, n, n-r-1)
            add_edge!(g, n, n-r+1)
        end
    end
    return g
end

"""
Functions on ZDDs
"""
function calculate_paths!(zdd::ZDD)
    """
    """
    sorted_nodes = sort(collect(zdd.nodes), rev=true, by=pair->pair[2])
    node_at = Dict(int => node for (node, int) ∈ zdd.nodes)
    for (n, i) ∈ sorted_nodes
        if n isa TerminalNode
            continue
        end
        child_nodes = [node_at[j] for j in zdd.graph.fadjlist[i]]
        paths = 0
        for child_node ∈ child_nodes
            zdd.paths[child_node] != -1 ? paths += zdd.paths[child_node] : error("We somehow ordered the nodes wrong...")
        end
    zdd.paths[n] = paths
    end
    return nothing
end

"""
Edge Ordering
"""

function se_diag(x,y,x_max,y_max)
    edge_list = Tuple{Tuple{Int, Int}, Tuple{Int, Int}}[]
    while(y > 0 && x < x_max)
        push!(edge_list, ((x,y),(x,y-1)))
        push!(edge_list, ((x,y-1),(x+1,y-1)))
        x += 1
        y -= 1
    end
    if y > 0
        push!(edge_list, ((x,y),(x,y-1)))
    end
    return edge_list
end

function nw_diag(x,y,x_max,y_max)
    edge_list = Tuple{Tuple{Int, Int}, Tuple{Int, Int}}[]
    while(x > 0 && y < y_max)
        push!(edge_list, ((x,y),(x-1,y)))
        push!(edge_list, ((x-1,y),(x-1,y+1)))
        x -= 1
        y += 1
    end
    if x > 0
        push!(edge_list, ((x,y),(x-1,y)))
    end
    return edge_list
end

function optimal_grid_edge_order(g::SimpleGraph, r::Int, c::Int)
    x_max = c-1
    y_max = r-1
    python_edge_list = Tuple{Tuple{Int, Int}, Tuple{Int,Int}}[]
    if x_max >= y_max # think about if this changes python -> julia
        for y ∈ 0:y_max-1
            python_edge_list = vcat(python_edge_list, se_diag(0,y+1,x_max,y_max))
        end
        for x ∈ 0:x_max-1
            push!(python_edge_list, ((x,y_max),(x+1,y_max)))
            python_edge_list = vcat(python_edge_list, se_diag(x+1,y_max,x_max,y_max))
        end
    else
        for x ∈ 0:x_max-1
            python_edge_list = vcat(python_edge_list, nw_diag(x+1,0,x_max,y_max))
        end
        for y ∈ 0:y_max-1
            push!(python_edge_list, ((x_max,y),(x_max,y+1)))
            python_edge_list = vcat(python_edge_list, nw_diag(x_max,y+1,x_max,y_max))
        end
    end
    julia_edge_list = [convert_python_edges_to_julia(edge, r) for edge in python_edge_list]
    @assert length(julia_edge_list) == length(edges(g))
    for e in julia_edge_list
        @assert e in edges(g)
    end
    return julia_edge_list
end

function optimal_queen_grid_edge_order(g::SimpleGraph, r::Int, c::Int)
    x_max = c-1
    y_max = r-1
    python_edge_list = Tuple{Tuple{Int, Int}, Tuple{Int,Int}}[]
    if x_max >= y_max
        for x ∈ 0:x_max-1
            for y ∈ 0:y_max-1
                if x == 0
                    push!(python_edge_list, ((x,y),(x,y+1)))
                end
                push!(python_edge_list, ((x,y),(x+1,y)))
                push!(python_edge_list, ((x+1,y),(x,y+1)))
                push!(python_edge_list, ((x,y),(x+1,y+1)))
                push!(python_edge_list, ((x+1,y),(x+1,y+1)))
            end
            push!(python_edge_list, ((x,y_max),(x+1,y_max)))
        end
    else
        for y ∈ 0:y_max-1
            for x ∈ 0:x_max-1
                if y == 0
                    push!(python_edge_list, ((x,y),(x+1,y)))
                end
                push!(python_edge_list, ((x,y),(x,y+1)))
                push!(python_edge_list, ((x+1,y),(x,y+1)))
                push!(python_edge_list, ((x,y),(x+1,y+1)))
                push!(python_edge_list, ((x,y+1),(x+1,y+1)))
            end
            push!(python_edge_list, ((x_max,y),(x_max,y+1)))
        end
    end
    julia_edge_list = [convert_python_edges_to_julia(edge, r) for edge in python_edge_list]
    @assert length(julia_edge_list) == length(edges(g))
    for e in julia_edge_list
        @assert e in edges(g)
    end
    return julia_edge_list
end

function convert_python_vertices_to_julia(t::Tuple{Int,Int}, c::Int)
        x, y = t[1], t[2]
        return c*x+y+1
end

function convert_python_edges_to_julia(e::Tuple{Tuple{Int, Int}, Tuple{Int,Int}}, c::Int)
        vtx_1 = convert_python_vertices_to_julia(e[1], c)
        vtx_2 = convert_python_vertices_to_julia(e[2], c)
        return LightGraphs.SimpleGraphs.SimpleEdge(minmax(vtx_1, vtx_2))
end

function frontier_distribution(g::SimpleGraph, g_edges::Array{LightGraphs.SimpleGraphs.SimpleEdge{Int}, 1})
    frontiers = compute_all_frontiers(g, g_edges)
    for i ∈ 0:maximum(length(frontier) for frontier in frontiers)
        c = count(length(frontier) == i for frontier in frontiers)
        println("$c frontiers of length $i")
    end
    return
end

function frontier_distribution(frontiers::Array{Set{Int},1})
    for i ∈ 0:maximum(length(frontier) for frontier in frontiers)
        c = count(length(frontier) == i for frontier in frontiers)
        println("$c frontiers of length $i")
    end
    return
end
