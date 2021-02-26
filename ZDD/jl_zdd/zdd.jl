struct ZDD_Node
    zero::Int
    one::Int
end

mutable struct ZDD{N<:Node, S<:SimpleGraph}
    graph::Vector{ZDD_Node}
    nodes::Dict{UInt64, Int64}
    # nodes_complete::Dict{N, Int64}    # used only when viz = True
    base_graph::S
    root::N
    paths::Int
    viz::Bool
end

function ZDD(g::SimpleGraph, root::Node; viz::Bool=false)::ZDD
    graph = Vector{ZDD_Node}()

    zero_node_zdd = ZDD_Node(0, 0)
    one_node_zdd = ZDD_Node(1, 1)
    root_stub = ZDD_Node(7, 7) # these are "random" numbers

    push!(graph, zero_node_zdd)
    push!(graph, one_node_zdd)
    push!(graph, root_stub)

    zero_node_complete = Node(0)
    one_node_complete = Node(1)

    nodes = Dict{UInt64, Int64}()
    nodes[zero_node_complete.hash] = 1
    nodes[one_node_complete.hash] = 2
    nodes[root.hash] = 3

    base_graph = g
    paths = 0
    return ZDD(graph, nodes, base_graph, root, paths, viz)
end

function construct_zdd(g::SimpleGraph,
                       k::Int64,
                       d::Int64,
                       g_edges::Array{NodeEdge,1};
                       weights::Vector{Int64}=Vector{Int64}([1 for i in 1:nv(g)]),
                       viz::Bool=false,
                       save_fp::String="zdd_tree.txt")::ZDD
    # delete file if it already exists
    # rm(save_fp)

    weights = [convert(UInt32,i) for i in weights]
    root = Node(g_edges[1], g, weights)

    lower_bound = Int32(floor(sum(weights)/k - d))
    upper_bound = Int32(floor(sum(weights)/k + d))
    println("Accepting districts with population in [$lower_bound, $upper_bound]")

    zdd = ZDD(g, root, viz=viz)
    N = Vector{Set{Node}}([Set{Node}([]) for a in 1:ne(g)+1])
    N[1] = Set([root])
    frontiers = compute_all_frontiers(g, g_edges)
    xs = Vector{Int8}([0, 1])
    zero_terminal = Node(0)
    one_terminal = Node(1)
    fp_container = Vector{ForbiddenPair}([]) # reusable container
    rm_container = Vector{ForbiddenPair}([]) # reusable container
    reusable_set = Set{ForbiddenPair}([])
    recycler = Stack{Node}()
    lower_vs = Vector{UInt8}([])
    deleted_nodes = 0

    for i = 1:ne(g)
        for n in N[i]
            n_idx = zdd.nodes[n.hash]
            for x in xs
                n′ = make_new_node(g, g_edges, k, n, i, x, d, frontiers,
                                   lower_bound, upper_bound,
                                   zero_terminal, one_terminal,
                                   fp_container, rm_container, lower_vs, recycler)

                if n′ === one_terminal
                    zdd.paths += n.paths
                end

                if !(n′.label == NodeEdge(0, 0) || n′.label == NodeEdge(1, 1)) # if not a Terminal Node
                    n′.label = g_edges[i+1] # update the label of n′
                    reusable_unique!(n′.fps, reusable_set)
                    sort!(n′.fps, alg=QuickSort)
                    n′.hash = hash(n′)

                    if n′ ∈ N[i+1]
                        index = Base.ht_keyindex2!(N[i+1].dict, n′)
                        N[i+1].dict.keys[index].paths += n.paths
                    else
                        add_zdd_node_and_edge!(zdd, n′, n, n_idx, x, deleted_nodes)
                        push!(N[i+1], n′)
                        continue
                    end
                end
                add_zdd_edge!(zdd, n, n′, n_idx, x, deleted_nodes)
            end
        end
        deleted_nodes += length(N[i])
        # save_tree_so_far!(zdd, save_fp)
        erase_upper_levels!(zdd, N[i+1], zero_terminal, one_terminal, length(N[i])) # release memory
        N[i] = Set{Node}([])   # release memory
    end
    return zdd
end

function save_tree_so_far!(zdd::ZDD, save_fp::String)
    output_file = open(save_fp, "a")
    serialize(output_file, zdd.graph[3:length(zdd.graph)]) # ignore the first two because they are terminals
    close(output_file)
end

function erase_upper_levels!(zdd::ZDD, N::Set{Node}, zero_terminal::Node, one_terminal::Node, prev_layer_size::Int)
    """ N is the layer you want to keep
    """
    # collect hashes
    hashes = Set{UInt64}()
    for node in N
        push!(hashes, node.hash)
    end
    # dont delete the terminal hashes!
    push!(hashes, zero_terminal.hash)
    push!(hashes, one_terminal.hash)

    # delete everything not in hashes
    for node_hash in keys(zdd.nodes)
        if node_hash ∉ hashes
            pop!(zdd.nodes, node_hash)
        end
    end

    # delete previous layer
    for i in 1:prev_layer_size
        deleteat!(zdd.graph, 3) # don't delete terminal layers at pos 1 and 2
    end
end

function copy_to_vec!(vec::Vector{ForbiddenPair}, set::Set{ForbiddenPair})
    for item in set
        push!(vec, item)
    end
end

function reusable_unique!(vec::Vector{ForbiddenPair}, set::Set{ForbiddenPair})
    union!(set, vec)
    empty!(vec)
    copy_to_vec!(vec, set)
    empty!(set)
end

function replace_components_with_union!(
    node::Node,
    Cᵤ::UInt8,
    Cᵥ::UInt8,
    fp_container::Vector{ForbiddenPair},
    rm_container::Vector{ForbiddenPair}
    )
    """
    update fps to replace the smaller component with the larger component
    (TODO: rename)
    """
    assignment = max(Cᵤ, Cᵥ)
    to_change = min(Cᵤ, Cᵥ)

    for fp in node.fps
        if to_change == fp.comp₁
            other = fp.comp₂
            push!(rm_container, fp)
            push!(fp_container, ForbiddenPair(min(assignment, other), max(assignment, other)))
        elseif to_change == fp.comp₂
            other = fp.comp₁
            push!(rm_container, fp)
            push!(fp_container, ForbiddenPair(min(assignment, other), max(assignment, other)))
        end
    end
    filter!(x -> x ∉ rm_container, node.fps)
    append!(node.fps, fp_container)

    empty!(rm_container)
    empty!(fp_container)
end

function remove_vertex_from_node!(node::Node, vertex::UInt8, fp_container::Vector{ForbiddenPair},
                                  rm_container::Vector{ForbiddenPair}, lower_vs::Vector{UInt8})
    """
    """
    @inbounds vertex_comp = node.comp_assign[vertex]
    c = count(x -> x == vertex_comp, node.comp_assign)

    if c == 1
        @inbounds node.comp_assign[vertex] = 0
        for fp in node.fps
            if (vertex_comp == fp.comp₁ || vertex_comp == fp.comp₂)
                push!(fp_container, fp)
            end
        end

        filter!(x -> x ∉ fp_container, node.fps)
        empty!(fp_container)
    elseif c > 1
        @inbounds node.comp_assign[vertex] = 0
        adjust_node!(node, vertex_comp, fp_container, rm_container, lower_vs)
    end

    if vertex == node.first_idx
        node.first_idx += 1
    end
end

function components(u::UInt8, v::UInt8, node::Node)::Tuple{UInt8, UInt8}
    """ Returns Cᵤ and Cᵥ which are the sets in `components` that contain
        vertices `u` and `v` respectively.
    """
    return node.comp_assign[u], node.comp_assign[v]
end

function lower_vertices!(num::UInt8, arr::Vector{UInt8}, container::Vector{UInt8})
    """
    """
    empty!(container)
    for (i, x) in enumerate(arr)
        if x == num
            push!(container, i)
        end
    end
end

function add_zdd_node_and_edge!(zdd::ZDD, n′::Node, n::Node, n_idx::Int64, x::Int8, deleted_nodes::Int)
    """
    """
    new_node = ZDD_Node(0, 0)
    push!(zdd.graph, new_node)

    n′_idx = length(zdd.graph) + deleted_nodes
    zdd.nodes[n′.hash] = n′_idx

    n′.paths = n.paths
    curr_n_idx = n_idx - deleted_nodes

    # add to graph
    if x == 0
        zdd.graph[curr_n_idx] = ZDD_Node(n′_idx, zdd.graph[curr_n_idx].one)
    else
        zdd.graph[curr_n_idx] = ZDD_Node(zdd.graph[curr_n_idx].zero, n′_idx)
    end
end

function add_zdd_edge!(zdd::ZDD,
                       node₁::Node,
                       node₂::Node,
                       node₁_idx::Int64,
                       x::Int8,
                       deleted_nodes::Int)
    """ Add an edge from node₁ to node₂.
    """
    node₂_idx = zdd.nodes[node₂.hash]
    curr_node₁_idx = node₁_idx - deleted_nodes

    # add to graph
    if x == 0
        zdd.graph[curr_node₁_idx] = ZDD_Node(node₂_idx, zdd.graph[curr_node₁_idx].one)
    else
        zdd.graph[curr_node₁_idx] = ZDD_Node(zdd.graph[curr_node₁_idx].zero, node₂_idx)
    end
end
