# visualization.jl

"""
Visualization
"""

function draw(zdd::ZDD, g, g_edges)
    """
    """
    if zdd.viz == false
        println("Cannot draw ZDD, viz option is turned off.")
        return nothing
    end

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
    label_occs = label_occurences(zdd) #  this is a Dict{NodeEdge, Int}

    tree_depth = length(keys(label_occs)) - 1 # - 1 because the two terminals will be on the same depth
    tree_width = maximum(values(label_occs))
    loc_xs = fill(-1., nv(zdd.graph))
    loc_xs[[1, 2]] = [tree_width/3 , tree_width*2/3]
    loc_ys = fill(-1., nv(zdd.graph))
    loc_ys[[1, 2]] = [tree_depth, tree_depth]

    labels_seen = Dict{NodeEdge, Int}()
    add_locations(zdd::ZDD, zdd.root, loc_xs, loc_ys, tree_width, label_occs, labels_seen, g_edges)

    loc_xs = Float64.(loc_xs)
    loc_ys = Float64.(loc_ys)

    return loc_xs, loc_ys
end

function node_by_idx(zdd::ZDD, idx)
    for (node, val) in zdd.nodes_complete
        if val == idx
            return node
        end
    end
end

function add_locations(zdd::ZDD, node, loc_xs, loc_ys, tree_width, label_occs, labels_seen, g_edges)
    """
    """
    if node.label == NodeEdge(0,0) || node.label == NodeEdge(1,1)
        return
    end

    # put in the location of the node
    if loc_xs[zdd.nodes_complete[node]] == -1
        if node.label in keys(labels_seen) # i wont be able to extract a node here.
            labels_seen[node.label] += 1
        else
            labels_seen[node.label] = 1
        end

        loc_xs[zdd.nodes_complete[node]] = labels_seen[node.label] * (tree_width / (label_occs[node.label] + 1))
        loc_ys[zdd.nodes_complete[node]] = findfirst(y -> y == node.label, g_edges) #collect(edges(zdd.base_graph)))
    end

    # recurse on the neighbors of the node
    neighbors = [node_by_idx(zdd, n) for n in outneighbors(zdd.graph, zdd.nodes_complete[node])]

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

# function label_occurences(zdd::ZDD)
#     """
#     """
#     keys = Set(values(zdd.node_labels))
#     arr = values(zdd.node_labels)
#     occs = Dict{NodeEdge, Int}([])
#     for key in keys
#         occs[key] = count(x-> x == key, arr)
#     end
#
#     # insert root
#     occs[zdd.root.label] = 1
#
#     # insert terminal nodes
#     occs[NodeEdge(0,0)] = 1
#     occs[NodeEdge(1,1)] = 1
#
#     occs
# end

function label_occurences(zdd::ZDD)
    """
    """
    occs = Dict{NodeEdge, Int}()
    for node in keys(zdd.nodes_complete)
        if node.label in keys(occs)
            occs[node.label] += 1
        else
            occs[node.label] = 1
        end
    end
    return occs
end

function get_corresponding_node(zdd, node_id)
    for (k , v) in zdd.nodes_complete
        if v == node_id
            return k
        end
    end
end

function readable(node_labels::Array{NodeEdge, 1})
    map(x -> string(Int64(x.edge₁)) * " -> " * string(Int64(x.edge₂)), node_labels)
end

function label_nodes(zdd::ZDD)
    """
    """
    node_labels = Array{NodeEdge, 1}(undef, nv(zdd.graph))
    for node in keys(zdd.nodes_complete)
        node_labels[zdd.nodes_complete[node]] = node.label
    end
    readable(node_labels)
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
