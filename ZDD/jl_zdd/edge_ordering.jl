# edge_ordering.jl

"""
Edge Ordering
"""

function se_diag(x,y,x_max,y_max)
    edge_list = Tuple{Tuple{Int, Int}, Tuple{Int, Int}}[]
    while(y > 0 && x < x_max)
        push!(edge_list, ((x,y),(x,y-1)))
        push!(edge_list, ((x,y-1),(x+1,y-1)))
        x += 1
        y -= 1
    end
    if y > 0
        push!(edge_list, ((x,y),(x,y-1)))
    end
    return edge_list
end

function nw_diag(x,y,x_max,y_max)
    edge_list = Tuple{Tuple{Int, Int}, Tuple{Int, Int}}[]
    while(x > 0 && y < y_max)
        push!(edge_list, ((x,y),(x-1,y)))
        push!(edge_list, ((x-1,y),(x-1,y+1)))
        x -= 1
        y += 1
    end
    if x > 0
        push!(edge_list, ((x,y),(x-1,y)))
    end
    return edge_list
end

function optimal_grid_edge_order_diags(g::SimpleGraph, r::Int, c::Int)
    x_max = c-1
    y_max = r-1
    python_edge_list = Tuple{Tuple{Int, Int}, Tuple{Int,Int}}[]
    if x_max >= y_max # think about if this changes python -> julia
        for y ∈ 0:y_max-1
            python_edge_list = vcat(python_edge_list, se_diag(0,y+1,x_max,y_max))
        end
        for x ∈ 0:x_max-1
            push!(python_edge_list, ((x,y_max),(x+1,y_max)))
            python_edge_list = vcat(python_edge_list, se_diag(x+1,y_max,x_max,y_max))
        end
    else
        for x ∈ 0:x_max-1
            python_edge_list = vcat(python_edge_list, nw_diag(x+1,0,x_max,y_max))
        end
        for y ∈ 0:y_max-1
            push!(python_edge_list, ((x_max,y),(x_max,y+1)))
            python_edge_list = vcat(python_edge_list, nw_diag(x_max,y+1,x_max,y_max))
        end
    end
    julia_edge_list = [convert_python_edges_to_julia(edge, r) for edge in python_edge_list]
    @assert length(julia_edge_list) == length(edges(g))
    for e in julia_edge_list
        @assert e in edges(g)
    end
    return julia_edge_list
end

function optimal_grid_edge_order_rows(g::SimpleGraph, r::Int, c::Int)
    x_max = c-1
    y_max = r-1
    python_edge_list = Tuple{Tuple{Int,Int}, Tuple{Int,Int}}[]
    if x_max >= y_max
        for x ∈ 0:x_max-1
            for y ∈ 0:y_max-1
                push!(python_edge_list, ((x,y),(x,y+1)))
            end
            for y ∈ 0:y_max
                push!(python_edge_list, ((x,y),(x+1,y)))
            end
        end
        for y ∈ 0:y_max-1
            push!(python_edge_list, ((x_max,y),(x_max,y+1)))
        end
    else
        for y ∈ 0:y_max-1
            for x ∈ 0:x_max-1
                push!(python_edge_list, ((x,y),(x+1,y)))
            end
            for x ∈ 0:x_max
                push!(python_edge_list, ((x,y),(x,y+1)))
            end
        end
        for x ∈ 0:x_max-1
            push!(python_edge_list, ((x,y_max),(x+1,y_max)))
        end
    end
    julia_edge_list = [convert_python_edges_to_julia(edge, r) for edge in python_edge_list]
    @assert length(julia_edge_list) == length(edges(g))
    for e ∈ julia_edge_list
        @assert e ∈ edges(g)
    end
    return julia_edge_list
end

function optimal_grid_edge_order_boxes_1(g::SimpleGraph, r::Int, c::Int)
    x_max = c-1
    y_max = r-1
    python_edge_list = Tuple{Tuple{Int,Int}, Tuple{Int,Int}}[]
    if x_max >= y_max
        for y ∈ 0:y_max-1
            push!(python_edge_list, ((0,y),(0,y+1)))
        end
        for x ∈ 0:x_max-1
            push!(python_edge_list, ((x,0),(x+1,0)))
            for y ∈ 0:y_max-1
                push!(python_edge_list, ((x,y+1),(x+1,y+1)))
                push!(python_edge_list, ((x+1,y),(x+1,y+1)))
            end
        end
    else
        for x ∈ 0:x_max-1
            push!(python_edge_list, ((x,0),(x+1,0)))
        end
        for y ∈ 0:y_max-1
            push!(python_edge_list, ((0,y),(0,y+1)))
            for x ∈ 0:x_max-1
                push!(python_edge_list, ((x+1,y),(x+1,y+1)))
                push!(python_edge_list, ((x,y+1),(x+1,y+1)))
            end
        end
    end
    julia_edge_list = [convert_python_edges_to_julia(edge, r) for edge in python_edge_list]
    @assert length(julia_edge_list) == length(edges(g))
    for e ∈ julia_edge_list
        @assert e ∈ edges(g)
    end
    return julia_edge_list
end

function optimal_grid_edge_order_boxes_2(g::SimpleGraph, r::Int, c::Int)
    x_max = c-1
    y_max = r-1
    python_edge_list = Tuple{Tuple{Int,Int}, Tuple{Int,Int}}[]
    if x_max >= y_max
        push!(python_edge_list, ((0,0),(1,0)))
        for y ∈ 0:y_max-1
            push!(python_edge_list, ((0,y),(0,y+1)))
            push!(python_edge_list, ((0,y+1),(1,y+1)))
            push!(python_edge_list, ((1,y),(1,y+1)))
        end
        for x ∈ 1:x_max-1
            push!(python_edge_list, ((x,0),(x+1,0)))
            for y ∈ 0:y_max-1
                push!(python_edge_list, ((x,y+1),(x+1,y+1)))
                push!(python_edge_list, ((x+1,y),(x+1,y+1)))
            end
        end
    else
        push!(python_edge_list, ((0,0),(0,1)))
        for x ∈ 0:x_max-1
            push!(python_edge_list, ((x,0),(x+1,0)))
            push!(python_edge_list, ((x+1,0),(x+1,1)))
            push!(python_edge_list, ((x,1),(x+1,1)))
        end
        for y ∈ 1:y_max-1
            push!(python_edge_list, ((0,y),(0,y+1)))
            for x ∈ 0:x_max-1
                push!(python_edge_list, ((x+1,y),(x+1,y+1)))
                push!(python_edge_list, ((x,y+1),(x+1,y+1)))
            end
        end
    end
    julia_edge_list = [convert_python_edges_to_julia(edge, r) for edge in python_edge_list]
    @assert length(julia_edge_list) == length(edges(g))
    for e ∈ julia_edge_list
        @assert e ∈ edges(g)
    end
    return julia_edge_list
end



function optimal_queen_grid_edge_order(g::SimpleGraph, r::Int, c::Int)
    x_max = c-1
    y_max = r-1
    python_edge_list = Tuple{Tuple{Int, Int}, Tuple{Int,Int}}[]
    if x_max >= y_max
        for x ∈ 0:x_max-1
            for y ∈ 0:y_max-1
                if x == 0
                    push!(python_edge_list, ((x,y),(x,y+1)))
                end
                push!(python_edge_list, ((x,y),(x+1,y)))
                push!(python_edge_list, ((x+1,y),(x,y+1)))
                push!(python_edge_list, ((x,y),(x+1,y+1)))
                push!(python_edge_list, ((x+1,y),(x+1,y+1)))
            end
            push!(python_edge_list, ((x,y_max),(x+1,y_max)))
        end
    else
        for y ∈ 0:y_max-1
            for x ∈ 0:x_max-1
                if y == 0
                    push!(python_edge_list, ((x,y),(x+1,y)))
                end
                push!(python_edge_list, ((x,y),(x,y+1)))
                push!(python_edge_list, ((x+1,y),(x,y+1)))
                push!(python_edge_list, ((x,y),(x+1,y+1)))
                push!(python_edge_list, ((x,y+1),(x+1,y+1)))
            end
            push!(python_edge_list, ((x_max,y),(x_max,y+1)))
        end
    end
    julia_edge_list = [convert_python_edges_to_julia(edge, r) for edge in python_edge_list]
    @assert length(julia_edge_list) == length(edges(g))
    for e in julia_edge_list
        @assert e in edges(g)
    end
    return julia_edge_list
end

function convert_python_vertices_to_julia(t::Tuple{Int,Int}, c::Int)
        x, y = t[1], t[2]
        return c*x+y+1
end

function convert_python_edges_to_julia(e::Tuple{Tuple{Int, Int}, Tuple{Int,Int}}, c::Int)
        vtx_1 = convert_python_vertices_to_julia(e[1], c)
        vtx_2 = convert_python_vertices_to_julia(e[2], c)
        return LightGraphs.SimpleGraphs.SimpleEdge(minmax(vtx_1, vtx_2))
end
