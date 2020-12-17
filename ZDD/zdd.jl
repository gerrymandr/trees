using DataStructures
using LightGraphs
import Base: isequal, ==

# comp₁ < comp₂ by requirement
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
    comp_weights::Dict{Int, Int}
    cc::Int
    fps::Set{ForbiddenPair}
    comp_assign::Array{Int}
end

struct TerminalNode<:NodeZDD
    label::Int
end

function Node(root_edge::AbstractEdge, base_graph::SimpleGraph)
    comp_assign = [i for i in 1:nv(base_graph)]
    comp_weights = Dict((vtx, 1) for vtx ∈ 1:nv(base_graph))
    return Node(root_edge, Set{Int}(), comp_weights, 0, Set{ForbiddenPair}(), comp_assign)
end

mutable struct ZDD
    # graph stuff
    graph::SimpleDiGraph
    nodes::Dict{N, Int64} where N <: NodeZDD
    edges::Dict{Tuple{N, N}, Int64} where N <: NodeZDD
    edge_multiplicity::Set{Tuple{N, N}} where N <: NodeZDD
    base_graph::SimpleGraph
    root::Node
    paths_to_terminal::Dict{N, Int64} where N <: NodeZDD
    paths_to_root::Dict{N, Int} where N <: NodeZDD
    paths::Dict{N, Int} where N <: NodeZDD
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
    paths_to_terminal = Dict{NodeZDD, Int64}()
    paths_to_terminal[TerminalNode(0)] = 0
    paths_to_terminal[TerminalNode(1)] = 1
    paths_to_terminal[root] = -1
    paths_to_root = Dict{NodeZDD, Int64}()
    paths_to_root[TerminalNode(0)] = -1
    paths_to_root[TerminalNode(1)] = -1
    paths_to_root[root] = 1
    paths = Dict{NodeZDD, Int64}()
    paths[TerminalNode(0)] = -1
    paths[TerminalNode(1)] = -1
    paths[root] = 1
    return ZDD(graph, nodes, edges, edge_multiplicity, base_graph, root, paths_to_terminal, paths_to_root, paths)
end

function Base.:(==)(node₁::Node, node₂::Node)
    node₁.cc == node₂.cc &&
    node₁.comp_weights == node₂.comp_weights &&
    node₁.label == node₂.label &&
    node₁.comp == node₂.comp &&
    node₁.fps == node₂.fps &&
    node₁.comp_assign == node₂.comp_assign
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

function add_zdd_edge!(zdd::ZDD, zdd_edge::Tuple{NodeZDD, NodeZDD}, x::Int)
    """ zdd_edge is represented as (Node, Node)
    """
    node₁, node₂ = zdd_edge

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

function construct_zdd(g::SimpleGraph, k::Int, d::Int, g_edges::Array{LightGraphs.SimpleGraphs.SimpleEdge{Int64},1})
    # optimal_ordering ? g_edges = optimal_grid_edge_order(g, dims[1],dims[2]) : g_edges = collect(edges(g))
    frontier_distribution(g, g_edges)
    # select root
    root = Node(g_edges[1], g)

    zdd = ZDD(g, root)
    N = [Set{NodeZDD}([]) for a in 1:ne(g)+1]
    N[1] = Set([root])
    frontiers = compute_all_frontiers(g, g_edges)

    for i = 1:ne(g)
        for n in N[i]
            for x in [0, 1]
                n′ = make_new_node(g, g_edges, k, n, i, x, d, frontiers)
                # node_summary(n′)
                if !(n′ isa TerminalNode)
                    n′.label = g_edges[i+1] # update the label of n′

                    if n′ ∉ N[i+1]
                        push!(N[i+1], n′)
                        add_zdd_node!(zdd, n′)
                    end
                end
                add_zdd_edge!(zdd, (n, n′), x)
            end
        end
    end
    calculate_paths_to_terminal!(zdd)
    calculate_paths_to_root!(zdd)
    calculate_enumeration_paths!(zdd)
    return zdd
end

function node_summary(node::Node)
    println("Label: ", node.label)
    println("cc: ",node.cc)
    println("comp: ",node.comp)
    println("fps: ",node.fps)
    println("comp_assign: ", node.comp_assign)
    println("comp_weights: ", node.comp_weights)
    println()
end

function node_summary(node::TerminalNode)
    println("Label: ", node.label)
    println()
end

function make_new_node(g::SimpleGraph, g_edges, k::Int, n::NodeZDD, i::Int, x::Int, d::Int, frontiers::Array{Set{Int}, 1})
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
        # println("at step $i, curr frontier is $curr_frontier")
        if a ∉ curr_frontier
            a_comp = n′.comp_assign[a]

            if a_comp in n′.comp && length(filter(x -> x == a_comp, n′.comp_assign)) == 1
                # println("Now in cc incrementing land")
                # node_summary(n′)
                lower_bound = Int(nv(g)/k - d) # TODO: extend to non-nice ratios
                upper_bound = Int(nv(g)/k + d)
                # println(lower_bound, upper_bound)
                if n′.comp_weights[a_comp] ∉ lower_bound:upper_bound
                    # println("Determined a bad connected component containing $a with weight $(n′.comp_weights[a_comp]):")
                    # node_summary(n′)
                    return TerminalNode(0)
                end
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
    update fps to replace the smaller component with the larger component
    (TODO: rename)
    """
    assignment = max(Cᵤ, Cᵥ)
    to_change = min(Cᵤ, Cᵥ)
    changed = []
    for fp in node.fps
        if to_change == fp.comp₁
            other = fp.comp₂
            delete!(node.fps, fp)
            push!(changed, ForbiddenPair(min(assignment, other), max(assignment, other)))
        elseif to_change == fp.comp₂
            other = fp.comp₁
            delete!(node.fps, fp)
            push!(changed, ForbiddenPair(min(assignment, other), max(assignment, other)))
        end
    end
    for fp in changed
        push!(node.fps, fp)
    end
end


function connect_components!(n::Node, Cᵤ::Int, Cᵥ::Int)
    """
    If the two components are different, remove the smaller component from n.comp, and update n.comp_assign. Adjust n.comp_weights to remove the smaller-indexed component and add its weight into the larger one.
    """
    assignment = max(Cᵤ, Cᵥ)
    to_change = min(Cᵤ, Cᵥ)
    if Cᵤ != Cᵥ
        map!(val -> val == to_change ? assignment : val, n.comp_assign, n.comp_assign)
        delete!(n.comp, to_change)
        n.comp_weights[assignment] += n.comp_weights[to_change]
        n.comp_weights[to_change] = n.comp_weights[assignment]
        # delete!(n.comp_weights, to_change)
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
    adjust_node!(node, vertex_comp)
end

function adjust_node!(node::Node, vertex_comp::Int)
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
        delete!(node.comp, vertex_comp)
        push!(node.comp, new_max)

        # change ForbiddenPair
        changed = []
        for fp in node.fps
            if vertex_comp == fp.comp₁
                other = fp.comp₂
                delete!(node.fps, fp)
                push!(changed, ForbiddenPair(min(new_max, other), max(new_max, other)))
            elseif vertex_comp == fp.comp₂
                other = fp.comp₁
                delete!(node.fps, fp)
                push!(changed, ForbiddenPair(min(new_max, other), max(new_max, other)))
            end
        end
        for fp in changed
            push!(node.fps, fp)
        end
    end
end

function add_zdd_node!(zdd::ZDD, node::N) where N <: NodeZDD
    """
    """
    if !haskey(zdd.nodes, node)
        add_vertex!(zdd.graph)
        zdd.nodes[node] = nv(zdd.graph)
        zdd.paths[node] = -1
    end
end

"""
Visualization
"""

function draw(zdd::ZDD, g, g_edges)
    """
    """
    e_colors = edge_colors(zdd)
    node_labels = label_nodes(zdd)
    loc_xs, loc_ys = node_locations(zdd, g_edges)

    node_colors = fill("lightseagreen", nv(zdd.graph))
    node_colors[[1, 2]] = ["orange", "orange"]

    gplot(zdd.graph, loc_xs, loc_ys,
          nodefillc=node_colors,
          nodelabel=node_labels,
          edgestrokec=e_colors,
          nodelabelsize=0.8)
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

function node_locations(zdd, g_edges)
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
    add_locations(zdd::ZDD, zdd.root, loc_xs, loc_ys, tree_width, label_occs, labels_seen, g_edges)

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

function add_locations(zdd::ZDD, node, loc_xs, loc_ys, tree_width, label_occs, labels_seen, g_edges)
    """
    """
    if node isa TerminalNode
        return
    end

    # put in the location of the node
    if loc_xs[zdd.nodes[node]] == -1
        if node.label in keys(labels_seen)
            labels_seen[node.label] += 1
        else
            labels_seen[node.label] = 1
        end

        loc_xs[zdd.nodes[node]] = labels_seen[node.label] * (tree_width / (label_occs[node.label] + 1))
        loc_ys[zdd.nodes[node]] = findfirst(y -> y == node.label, g_edges) #collect(edges(zdd.base_graph)))
    end

    # recurse on the neighbors of the node
    neighbors = [node_by_idx(zdd, n) for n in outneighbors(zdd.graph, zdd.nodes[node])]

    if length(neighbors) == 1
        add_locations(zdd, neighbors[1], loc_xs, loc_ys, tree_width, label_occs, labels_seen, g_edges)

    elseif length(neighbors) == 2
        order_1 = (node, neighbors[1])
        order_2 = (node, neighbors[2])

        if zdd.edges[order_1] == 0 && zdd.edges[order_2] == 1
            neighbors = [neighbors[1], neighbors[2]]
        else
            neighbors = [neighbors[2], neighbors[1]]
        end

        for neighbor in neighbors
            add_locations(zdd, neighbor, loc_xs, loc_ys, tree_width, label_occs, labels_seen, g_edges)
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
        elseif n ∈ 1:r
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
function calculate_paths_to_terminal!(zdd::ZDD)
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
            zdd.paths_to_terminal[child_node] != -1 ? paths += zdd.paths_to_terminal[child_node] : error("We somehow ordered the nodes wrong...")
        end
        zdd.paths_to_terminal[n] = paths
    end
    return nothing
end

function calculate_paths_to_root!(zdd::ZDD)
    """
    """
    function calculate_paths!(n::NodeZDD, i::Int, zdd::ZDD)
        """
        TODO: Abstract this out so calculate_paths_to_terminal can use it
        """
        child_nodes = [node_at[j] for j in reversed_zdd.fadjlist[i]]
        paths = 0
        for child_node ∈ child_nodes
            zdd.paths_to_root[child_node] != -1 ? paths += zdd.paths_to_root[child_node] : error("We somehow ordered the nodes wrong...")
        end
        zdd.paths_to_root[n] = paths
        return nothing
    end

    sorted_nodes = sort(collect(zdd.nodes), by=pair->pair[2])
    node_at = Dict(int => node for (node, int) ∈ zdd.nodes)
    reversed_zdd = reverse(zdd.graph)
    for (n, i) ∈ sorted_nodes
        if i <= 3
            continue # do terminal nodes later
        end
        calculate_paths!(n, i, zdd)
    end
    for (n, i) ∈ sorted_nodes
        if i <= 2 # just terminal nodes
            calculate_paths!(n, i, zdd)
        end
    end
    return nothing
end

function calculate_enumeration_paths!(zdd::ZDD)
    for node ∈ zdd.nodes
        n = node[1]
        zdd.paths[n] = zdd.paths_to_root[n]*zdd.paths_to_terminal[n]
    end
    return nothing
end

function count_paths(zdd::ZDD)::Int
    """ Returns the number of paths to the 1 Node in `zdd`
    """
    terminal_level = Dict{Int, Int}()

    terminal_level[2] = 1 # the 1 terminal Node is always at node 2
    depth = 1             # depth at bottom of the ZDD tree is 1
    return count_paths(zdd, terminal_level, depth)
end

function count_paths(zdd::ZDD, prev_level::Dict{Int, Int}, curr_depth::Int)::Int
    """ Recursively finds the number of paths to a node.
        This function is called at each depth of the ZDD tree.
        Arguments:
            zdd        : ZDD object
            prev_level : Dict where keys are the nodes in the lower level, and
                         the values are the paths from that node to the 1 terminal node.
            curr_depth : depth traversed in the ZDD tree, from the bottom.
    """
    if curr_depth == ne(zdd.base_graph) + 1
        @assert length(prev_level) == 1 # we should be at the root so only 1
        @assert 3 in keys(prev_level)   # the root is always node 3
        return prev_level[3]
    end

    curr_level = Dict{Int, Int}()

    for node in keys(prev_level)
        for i in inneighbors(zdd.graph, node)
            if haskey(curr_level, i)
                curr_level[i] += prev_level[node]
            else
                curr_level[i] = prev_level[node]
            end
        end
    end

    return count_paths(zdd, curr_level, curr_depth+1)
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

"""
Enumerating Plans
"""
function choose_positive_child!(zdd::ZDD, node::NodeZDD, plan::Array{LightGraphs.SimpleGraphs.SimpleEdge{Int}, 1}, plans::Array{Array{LightGraphs.SimpleGraphs.SimpleEdge{Int64},N} where N,1})
    node_idx = zdd.nodes[node]
    node_at = Dict(int => node for (node, int) ∈ zdd.nodes)
    if node == zdd.root
        if zdd.paths[node] == 0
            return nothing
        end
        zdd.paths[node] -= 1
        zdd.paths_to_terminal[node] -= 1
    end
    if !(node isa TerminalNode)  # terminate when we reach the 1-terminal
        child_nodes = [(node_at[j], zdd.edges[(node, node_at[j])]) for j ∈ zdd.graph.fadjlist[node_idx]]
        sort!(child_nodes, by=pair->pair[2])
        for (child_node, i) ∈ child_nodes
            if zdd.paths[child_node] > 0 # choose a child that has some path to 1-terminal
                edge_type = zdd.edges[(node, child_node)]
#                 println("On node $(node.label), its $edge_type-child, $(child_node.label) has $(zdd.paths[child_node]) paths")
#                 if(zdd.paths_to_terminal[child_node] <= 0 && count(plan == partial_plan for partial_plan ∈ map(x->x[begin:length(plan)], plans)) > 0)
#                     println("We're aborting this particular path though...")
#                     continue
#                 end
                if !(child_node isa TerminalNode)
                    zdd.paths[child_node] -= 1
                    zdd.paths_to_terminal[child_node] -= 1
                    zdd.paths_to_root[child_node] -= 1
                end
                i == 1 && push!(plan, node.label) # retain this edge if 1-arc
                global next_node = child_node
                break # if we chose the 0-arc child, don't go on to the 1-arc child this time
            else
                continue # if the 0-arc child doesn't have any paths, try the 1-arc child
            end
        end
        return choose_positive_child!(zdd, next_node, plan, plans)
    else
        push!(plans, plan)
#         println("Found a plan:")
#         println(plan)
#         println()
        return nothing
    end
end

function enumerate_plans(zdd::ZDD)
    num_plans = zdd.paths[zdd.root]
    println("Enumerating all $num_plans plans")
    plans = Array{LightGraphs.SimpleGraphs.SimpleEdge{Int}}[]
    root = zdd.root

    while zdd.paths[root] > 0
        plan = LightGraphs.SimpleGraphs.SimpleEdge{Int}[]
        choose_positive_child!(zdd, root, plan, plans)
    end
    return plans
end
