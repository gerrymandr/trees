### This is where unused functions go to die.

function update_comp_assign!(node::Node, vertex_comp::UInt8)::UInt8
    """
    """
    indexes = Vector{UInt8}(undef, 0)
    for (i, item) in enumerate(node.comp_assign)
        if item == vertex_comp
            push!(indexes, i)
        end
    end
    new_max = maximum(indexes)

    for i in indexes
        node.comp_assign[i] = new_max
    end
    return new_max
end

function update_comp!(node::Node, vertex_comp::UInt8, new_max::UInt8)
    filter!(x -> x != vertex_comp, node.comp)
    push!(node.comp, new_max)
    sort!(node.comp) # needed for Node equality to increase Node merges
end

function update_comp_weights!(node::Node, vertex_comp::UInt8, new_max::UInt8)
    if new_max != vertex_comp
        node.comp_weights[new_max] = node.comp_weights[vertex_comp]
        node.comp_weights[vertex_comp] = 0
    end
end

function add_vertex_as_component!(n′::Node, vertex::UInt8, prev_frontier::Set{UInt8})
    """ Add `u` or `v` or both to n`.comp if they are not in
        `prev_frontier`
    """
    if vertex ∉ prev_frontier
        push!(n′.comp, vertex)
    end
end
