# count and enumerate plans

function count_paths(zdd::ZDD)::UInt128
    """ Return the total number of plans that go to the 1 terminal
    """
    zdd.paths
end

"""
Enumerating Plans
"""
function enumerate_plans(zdd::ZDD)
    """
    """
    # get plans
    plans = Vector{Vector{NodeEdge}}()
    plan = Vector{NodeEdge}()
    traverse_zdd(zdd, zdd.graph[3], plans, plan, 1)

    # turn plans into LightGraphs edges
    lg_plans = Vector{Vector{LightGraphs.SimpleGraphs.SimpleEdge{Int64}}}()
    for plan in plans
        lg_plan = convert_node_edges_to_lightgraphs_edges(plan)
        push!(lg_plans, lg_plan)
    end
    lg_plans
end

function traverse_zdd(zdd::ZDD, node, plans, plan, idx)
    """ Recursive function that traverses the ZDD by going to the zero edge of ``node"
        first and then to the one edge of ``node".

    ``plans" is the list of plans that end in the one terminal.
    ``plan" is the list of edges being build for the current plan.
    ``idx" is the index of the ``zdd.edge_ordering" we are at currently.
    """

    if node.zero == 1 # 0 terminal
        nothing
    elseif node.zero == 2
        push!(plans, deepcopy(plan)) # 1 terminal
    else
        traverse_zdd(zdd, zdd.graph[node.zero], plans, plan, idx+1)
    end

    push!(plan, zdd.edge_ordering[idx])

    if node.one == 1  # 0 terminal
        nothing
    elseif node.one == 2 # 1 terminal
        push!(plans, deepcopy(plan))
    else
        traverse_zdd(zdd, zdd.graph[node.one], plans, plan, idx+1)
    end

    pop!(plan)
end

""" Printing functions
"""
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
