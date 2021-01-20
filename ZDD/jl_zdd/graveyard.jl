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

function adjust_node!(node::Node, vertex_comp::UInt8, fp_container::Vector{ForbiddenPair})
    """
    """
    if vertex_comp in node.comp_assign
        # there is atleast one lower vertex number that has the higher comp
        # number and needs to be adjusted
        new_max = update_comp_assign!(node, vertex_comp)
        update_comp!(node, vertex_comp, new_max)
        update_comp_weights!(node, vertex_comp, new_max)

        # change ForbiddenPair
        for fp in node.fps
            if vertex_comp == fp.comp₁
                other = fp.comp₂
                delete!(node.fps, fp)
                push!(fp_container, ForbiddenPair(min(new_max, other), max(new_max, other)))
            elseif vertex_comp == fp.comp₂
                other = fp.comp₁
                delete!(node.fps, fp)
                push!(fp_container, ForbiddenPair(min(new_max, other), max(new_max, other)))
            end
        end
        for fp in fp_container
            push!(node.fps, fp)
        end
    end
    empty!(fp_container)
end
