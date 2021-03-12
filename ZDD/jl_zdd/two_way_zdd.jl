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
                
function all_vertices_connected_to(v, node)
    representative_vertex = node.comp_assign[v]
    return Set(findall(==(representative_vertex), node.comp_assign))
end

function setup(fnode, bnode, frontier)
    fcomp, bcomp, intcomp = Dict(), Dict(), Dict() # add type annotations
    ffrontier, bfrontier, frontier_span = Dict(), Dict(), Dict() # add type annotations
    isolated_frontier_vtxs = Set()
    
    for v ∈ frontier
        fcomp[v] = all_vertices_connected_to(v, fnode)
        bcomp[v] = all_vertices_connected_to(v, bnode)
        intcomp[v] = intersect(fcomp[v],bcomp[v])
        
        ffrontier[v] = intersect(fcomp[v], frontier)
        bfrontier[v] = intersect(bcomp[v], frontier)
        frontier_span[v] = union(ffrontier[v], bfrontier[v])
        
        push!(isolated_frontier_vtxs, maximum(frontier_span[v]))
    end
    return fcomp, bcomp, intcomp, isolated_frontier_vtxs
end

function check_weights(fcomp, bcomp, intcomp, frontier, acceptable, w)
    for v ∈ frontier
        if !(w(fcomp[v]) + w(bcomp[v]) - w(intcomp[v]) ∈ acceptable)
            return false
        end
    end
    return true
end
        
function check_fps(fnode, bnode, fcomp, bcomp, intcomp, frontier)
    for v ∈ frontier
        for x ∈ fcomp[v]
            for y ∈ bcomp[v]
                if (x,y) ∈ union(fnode.fps, bnode.fps)
                    return false
                end
            end
        end
    end
    return true
end

function check_cc(fnode, bnode, k, isolated_frontier_vtxs)
    if fnode.cc + bnode.cc + length(isolated_frontier_vtxs) == k
        return true
    end
    return false
end

function is_compatible(fnode, bnode, frontier, acceptable, w, k, checking)
    fcomp, bcomp, intcomp, isolated_frontier_vtxs = setup(fnode, bnode, frontier)
    weights = check_weights(fcomp, bcomp, intcomp, frontier, acceptable, w)
    fps = check_fps(fnode, bnode, fcomp, bcomp, intcomp, frontier)
    cc = check_cc(fnode, bnode, k, isolated_frontier_vtxs)
    
    checklist = []
    if "weights" ∈ checking
        push!(checklist, weights)
    end
    if "fps" ∈ checking
        push!(checklist, fps)
    end
    if "cc" ∈ checking
        push!(checklist, cc)
    end
    if all(checklist)
        return true
    end
    return false
end

function w(v)
    return 1
end

function number_compatible(fnodes, bnodes, frontier, acceptable, w, k, checking)
    println("Total # of node pairs: $(length(fnodes)) * $(length(bnodes)) = $(length(fnodes) * length(bnodes))")
    num_partitions = 0
    for fnode ∈ fnodes
        for bnode ∈ bnodes
            if is_compatible(fnode, bnode, frontier, acceptable, w, k, checking)
                num_partitions += 1
            end
        end
    end
    return num_partitions
end