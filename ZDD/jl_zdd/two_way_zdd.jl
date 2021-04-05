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
    Returns Set(C₁, C₂, ..., Cₙ)  where every v ∈ Cᵢ is in the same connected component,
    and Cᵢ ⊆ F where F is the set of all vertices in the frontier.
    """
    frontier_sets = Set()
    seen_vertices = Set()
    for v ∈ collect(frontier)
        if v ∈ seen_vertices
            continue
        end
        Cᵥ = Set(findall(n -> n == v, node.comp_assign))
        if length(Cᵥ) > 0
            push!(frontier_sets, Cᵥ)
            for seen_vertex ∈ Cᵥ
                push!(seen_vertices, seen_vertex)
            end
        end
    end
    return frontier_sets
end

function merge_nodes(fnode, bnode, frontier)
    """
    Merges two nodes and returns the following:
        - labels  :: Dict({v:l}) where v ∈ F and l ∈ F is v's connected component number
        - weights :: Dict({v:n}) where v ∈ F and n ∈ N is the weight of v's component
        - connected_components :: Set(C₁, C₂, ..., Cₙ) frontier sets of the merged node
        - merged_fps :: Array(ForbiddenPair) union of the FPS from each node
    
    Outline of algorithm:
        0. Make a copy of fnode.comp_weights since we'll be modifying it later
        1. Initialize `weights` dictionary to -1 for all v ∈ F
        2. Initialize `labels` dictionary from fnode's perspective
        3. For each connected component `bcomp` in bnode:
            # We want to figure out the merged component `M`, which is all the vertices in F
            # that are connected by the merge of bcomp and fnode. 
            a. Initialize `M` = Set() and `w` = 0 (the weight of all vertices in `M`)
            b. For each vertex v ∈ `bcomp`:
                i.  Find the set `fvcomp`: {x ∈ F | x touches v from fnode's perspective},
                    and add each x ∈ `fvcomp` to M
                ii. Once per fcomp, add the weight of fcomp to `w`
            c. The weight added to `w` by `bcomp` is the weight of bcomp minus the weight
               of the intersection of bcomp and any components it touches in fnode.
            d. Now that `M` and `w` are finalized, we store the information by:
                i.   Relabel `labels` such that every v ∈ M has is connected
                ii.  In `weights`, assign `w` to every v ∈ M 
                iii. Update our copy of fnode.comp_weights to `w` for future bcomps
        4. Make `merged_fps` — just the union of fnode.fps and bnode.fps
        5. Make `connected_components` — just the Set(values(labels))
    """
    local_fnode_comp_weights = deepcopy(fnode.comp_weights)
    
    ### Initialize labels, weights ###
    labels = Dict()
    weights = Dict(v => -1 for v ∈ frontier)
    for set ∈ frontier_sets(fnode, frontier)
        for v ∈ set
            labels[v] = maximum(set)
        end
    end

    for bcomp ∈ frontier_sets(bnode, frontier)
        M = Set()
        w = 0
        seen_fcomps = Set()
        for v ∈ bcomp
            fvcomp = findall(x -> (labels[x] == labels[v]), collect(frontier))
            for x ∈ fvcomp
                push!(M,collect(frontier)[x])
            end
            if !(labels[v] ∈ seen_fcomps)
                w += local_fnode_comp_weights[labels[v]]
                push!(seen_fcomps, labels[v])
            end
        end

        w_bcomp = maximum(bnode.comp_weights[v] for v ∈ bcomp) # if you just pick one, you might hit a weird 0
        w_intersection = length(bcomp) # TODO: generalize past unit weights
        w += (w_bcomp - w_intersection)

        max_label = maximum(labels[v] for v ∈ M)
        for v ∈ M
            labels[v] = max_label
            weights[v] = w
            local_fnode_comp_weights[labels[v]] = w
        end
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
    # flabels = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]
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
                    println("($fi,$bi) contributes $(fnode.paths*bnode.paths) solns")
                end
            end
        end
    end
    return num_paths
end