# count and enumerate plans

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
    function calculate_paths!(n::Node, i::Int, zdd::ZDD)
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

function count_paths(zdd::ZDD, g_edges::Array{NodeEdge,1}, file::String)::Int
    """ Returns the number of paths to the 1 Node in `zdd`
    """
    terminal_level = Dict{Int, Int}()

    terminal_level[2] = 1 # the 1 terminal Node is always at node 2
    depth = 1             # depth at bottom of the ZDD tree is 1
    return count_paths(zdd, terminal_level, depth, g_edges, file)
end

function count_paths(zdd::ZDD, prev_level::Dict{Int, Int}, curr_depth::Int, g_edges::Array{NodeEdge,1}, file::String)::Int
    """ Recursively finds the number of paths to a node.
        This function is called at each depth of the ZDD tree.
        Arguments:
            zdd        : ZDD object
            prev_level : Dict where keys are the nodes in the lower level, and
                         the values are the paths from that node to the 1 terminal node.
            curr_depth : depth traversed in the ZDD tree, from the bottom.
    """
    node_at = Dict(int => node for (node, int) ∈ zdd.nodes_complete)
    if curr_depth == ne(zdd.base_graph) + 1
        @assert length(prev_level) == 1 # we should be at the root so only 1
        @assert 3 in keys(prev_level)   # the root is always node 3
        node_at[3].deadend = false # set the root node deadend column to false
        print_node_attributes(zdd, g_edges, file)
        return prev_level[3]
    end

    curr_level = Dict{Int, Int}()

    for node in keys(prev_level) # for child node with a path to terminal
        node_at[node].deadend = false
        for i in inneighbors(zdd.graph, node) # do for each parent
            if haskey(curr_level, i) # if the parent has already been treated, add the paths from this child
                curr_level[i] += prev_level[node]
            else # otherwise, start it of with these paths
                curr_level[i] = prev_level[node]
            end
        end
    end

    return count_paths(zdd, curr_level, curr_depth+1, g_edges, file)
end

function count_paths(zdd::ZDD)::UInt128
    """ Return the total number of plans that go to the 1 terminal
    """
    zdd.paths
end

# function count_paths(zdd::ZDD)::Int
#     """ Returns the number of paths to the 1 Node in `zdd`
#     """
#     terminal_level = Dict{Int, Int}()
#
#     terminal_level[2] = 1 # the 1 terminal Node is always at node 2
#     depth = 1             # depth at bottom of the ZDD tree is 1
#     return count_paths(zdd, terminal_level, depth)
# end

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

    for node in keys(prev_level) # for child node with a path to terminal

        for i in inneighbors(zdd.graph, node) # do for each parent
            if haskey(curr_level, i) # if the parent has already been treated, add the paths from this child
                curr_level[i] += prev_level[node]
            else # otherwise, start it of with these paths
                curr_level[i] = prev_level[node]
            end
        end
    end

    return count_paths(zdd, curr_level, curr_depth+1)
end


function print_node_attributes(node::Node, g_edges::Array{NodeEdge,1}, file::String)
    edge = readable(node.label)
    label = findfirst(x -> readable(x) == edge, g_edges)
    if isnothing(label)
        label = length(g_edges)
    else
        label -= 1
    end
    cc = readable(node.cc)
    fps = readable(node.fps)
    comp_assign = readable(node.comp_assign)
    comp_weights = readable(node.comp_weights)
    deadend = node.deadend
    open(file, "a") do io
        write(io, "$label\t$edge\t$cc\t$fps\t$comp_assign\t$comp_weights\t$deadend\n")
    end
    return nothing
end

function print_node_attributes(zdd::ZDD, g_edges::Array{NodeEdge,1}, file::String)
    nodes = keys(zdd.nodes_complete)
    open(file, "a") do io
        write(io, "level\tedge\tcc\tfps\tcomp_assign\tcomp_weights\tdeadend\n")
    end
    for node in nodes
        print_node_attributes(node, g_edges, file)
    end
    return nothing
end

function enumerate_paths(zdd::ZDD, g_edges) # think about typecasting the result
    terminal_level = Dict{Int, Array{Array{Int, 1},1}}()
    terminal_level[2] = Array{Array{Int,1},1}()
    push!(terminal_level[2], [])
    depth = 1
    return enumerate_paths(zdd, terminal_level, depth, g_edges)
end

function enumerate_paths(zdd::ZDD, prev_level::Dict{Int, Array{Array{Int, 1},1}}, curr_depth::Int, g_edges)
    if curr_depth == ne(zdd.base_graph) + 1
        @assert length(prev_level) == 1
        @assert 3 in keys(prev_level)
        @assert length(prev_level[3]) == count_paths(zdd) # not sure about this one
        result = Array{Array{LightGraphs.SimpleGraphs.SimpleEdge{Int},1},1}()
        for path in prev_level[3]
            edge_path = Array{LightGraphs.SimpleGraphs.SimpleEdge{Int},1}()
            for (i, elem) ∈ enumerate(path)
                if elem == 1
                    push!(edge_path, g_edges[i])
                end
            end
            push!(result, edge_path)
        end
        return result
    end

    curr_level = Dict{Int, Array{Array{Int, 1},1}}()
    node_at = Dict(int => node for (node, int) ∈ zdd.nodes)

    for child ∈ keys(prev_level)
        for parent ∈ inneighbors(zdd.graph, child)
            edge = zdd.edges[(node_at[parent], node_at[child])]
            for subpath ∈ prev_level[child]
                extended_subpath = deepcopy(subpath)
                prepend!(extended_subpath, edge)
                if haskey(curr_level, parent)
                    push!(curr_level[parent], extended_subpath)
                else
                    curr_level[parent] = [extended_subpath]
                end
            end
        end
    end
    return enumerate_paths(zdd, curr_level, curr_depth+1, g_edges)
end

"""
Enumerating Plans
"""
function choose_positive_child!(zdd::ZDD, node::Node, plan::Array{LightGraphs.SimpleGraphs.SimpleEdge{Int}, 1}, plans::Array{Array{LightGraphs.SimpleGraphs.SimpleEdge{Int64},N} where N,1})
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
