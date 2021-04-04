include("zdd.jl")

function construct_half_zdd(g::SimpleGraph,
                            k::Int64, # can we change this to Int8? should we?
                            d::Int64,
                            g_edges::Array{NodeEdge,1},
                            weights::Vector{Int64}=Vector{Int64}([1 for i in 1:nv(g)]),
                            viz::Bool=false,
                            save_fp::String="zdd_tree.txt")::Set{Node}

    # delete file if it already exists
    if isfile(save_fp)
        rm(save_fp)
    end

    weights = Vector{UInt32}([convert(UInt32,i) for i in weights])
    root = Node(g_edges[1], g, weights)

    lower_bound = Int32(floor(sum(weights)/k - d))
    upper_bound = Int32(floor(sum(weights)/k + d))

    println("Constructing a half-ZDD...")

    zdd = ZDD(g, root, viz=viz)
    halfway = Int(ne(g)/2)
    N = Vector{Set{Node}}([Set{Node}([]) for a in 1:halfway+1])
    N[1] = Set([root]) # why not Set(root)
    frontiers = compute_all_frontiers(g, g_edges) # only need to do half...
    xs = Vector{Int8}([0,1])
    zero_terminal = Node(0)
    one_terminal = Node(1)
    fp_container = Vector{ForbiddenPair}([])
    rm_container = Vector{ForbiddenPair}([])
    reusable_set = Set{ForbiddenPair}([])
    recycler = Stack{Node}() # what is this?
    lower_vs = Vector{UInt8}([])

    for i = 1:halfway
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

                if !(n′.label == NodeEdge(0,0) || n′.label == NodeEdge(1,1)) # if not a Terminal Node
                    n′.label = g_edges[i+1] #  update the label of n′
                    reusable_unique!(n′.fps, reusable_set)
                    sort!(n′.fps, alg=QuickSort)
                    n′.hash = hash(n′)

                    if n′ in N[i+1]
                        index = Base.ht_keyindex2!(N[i+1].dict, n′)
                        N[i+1].dict.keys[index].paths += n.paths
                    else
                        add_zdd_node_and_edge!(zdd, n′, n, n_idx, x)
                        push!(N[i+1], n′)
                        continue
                    end
                end
                add_zdd_edge!(zdd, n, n′, n_idx, x) # the order of n and n′ are switched, but probably ok
            end
        end
        if i == halfway
            return N[i+1]
        end
        zdd.deleted_nodes += length(N[i])
        save_tree_so_far!(zdd, save_fp, length(N[i]))
        erase_upper_levels!(zdd, N[i+1], zero_terminal, one_terminal, length(N[i])) # release memory
        N[i] = Set{Node}([])   # release memory
        # println(i, ": ", Base.summarysize(zdd))
    end
    # return zdd
end

function frontier_sets(node, frontier)
    """
    Returns the sets of vertices in the frontier that are connected to each other.
    """
    sets = Set()
    seen_vertices = Set()
    frontier_list = reverse(collect(frontier)) # reverse probably unnecessary
    for v ∈ frontier_list
        if v ∈ seen_vertices
            continue
        end
        set = Set(findall(n -> n == v, node.comp_assign))
        if length(set) > 0
            push!(sets, set)
            for seen_vertex ∈ set
                push!(seen_vertices, seen_vertex)
            end
        end
    end
    return sets
end

function merge_nodes(fnode, bnode, frontier)
    """
    Use Union-Find type of thing to merge nodes....
    """
    local_fnode_comp_weights = deepcopy(fnode.comp_weights)
    
    ffrontier_sets = frontier_sets(fnode, frontier)
    bfrontier_sets = frontier_sets(bnode, frontier)
    
    labels = Dict()
    weights = Dict(v => -1 for v in frontier)
    for set ∈ ffrontier_sets
        for v ∈ set
            labels[v] = maximum(set)
        end
    end
    for bset ∈ bfrontier_sets
        # println("bset is: $bset")
        # println("Right now weights is: $weights")
        U = Set()
        w = 0
        # println("w is: $w")
        seen_fcomps = Set()
        for (i,v) ∈ enumerate(bset)
            vgroup = findall(x -> (labels[x] == labels[v]), collect(frontier))
            for g ∈ vgroup
                push!(U,collect(frontier)[g])
            end
            # println("inside enumerate bset, w is: $w")
            if !(labels[v] in seen_fcomps)
                w += local_fnode_comp_weights[labels[v]]
                push!(seen_fcomps, labels[v])
            end
            # println("still inside, w is: $w")
        end
        # println("after local_fnode etc. w is: $w")
        w_bset = maximum(bnode.comp_weights[v] for v ∈ bset) # if you just pick one, you might hit a weird 0
        w_intersection = length(bset) # TODO: generalize past unit weights
        # println("w_bset is: $w_bset\nw_intersection is: $w_intersection")
        w += (w_bset - w_intersection)
        U_frontier = intersect(U, frontier) # need this because `labels` only has frontier keys
        # println("U_frontier is $U_frontier")
        max_label = maximum(labels[u] for u ∈ U_frontier)
        for u ∈ U_frontier
            labels[u] = max_label
            weights[u] = w
            local_fnode_comp_weights[labels[u]] = w
        end
        # println("End of loop block weights is: $weights")
    end
        
    merged_fps = union(fnode.fps, bnode.fps)
    connected_components = Set(values(labels))
    return connected_components, labels, weights, merged_fps
end

function check_cc(fnode, bnode, connected_components, k)
    if fnode.cc + bnode.cc + length(connected_components) == k
        return true
    else
        return false
    end
end

function check_fps(fnode, bnode, connected_components, labels, merged_fps, frontier)
    for c ∈ connected_components
        idxs = findall(x -> labels[x] == c, collect(frontier))
        vtxs = [collect(frontier)[i] for i ∈ idxs]
        for v₁ ∈ vtxs
            for v₂ ∈ vtxs
                if v₁ != v₂
                    maybe_forbidden = ForbiddenPair(v₁, v₂)
                    if maybe_forbidden ∈ merged_fps
                        return false
                    end
                else
                    continue
                end
            end
        end
    end
    return true
end

function check_weights(fnode, bnode, weights, acceptable)
    # println(weights)
    for w ∈ values(weights)
        if w ∉ acceptable
            return false
        end
    end
    return true
end

function count_paths_from_halfway(fnodes, bnodes, middle_frontier, acceptable, k, verbose=false)
    flabels = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]
    num_paths = 0
    if verbose
        println("Comparing $(length(fnodes)) fnodes to $(length(bnodes)) bnodes...\n")
    end
    for (fi,fnode) ∈ enumerate(fnodes)
        for (bi,bnode) ∈ enumerate(bnodes)
            ccs, labels, weights, merged_fps = merge_nodes(fnode, bnode, middle_frontier)
            cc = check_cc(fnode, bnode, ccs, k) 
            fps = check_fps(fnode, bnode, ccs, labels, merged_fps, middle_frontier)
            ws = check_weights(fnode, bnode, weights, acceptable)
            if cc & fps & ws
                num_paths += fnode.paths * bnode.paths
                if verbose
                    println("($(flabels[fi]),$bi) contributes $(fnode.paths*bnode.paths) solns")
                end
            end
        end
    end
    return num_paths
end