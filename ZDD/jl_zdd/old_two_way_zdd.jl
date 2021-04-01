include("zdd.jl")

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