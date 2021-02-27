# node.jl

function copy_to_vec!(vec₁::Vector{T}, vec₂::Vector{T}) where T
    """ Copy items from vec₁ into vec₂.
        It is assumed that length(vec₂) >= length(vec₁)
    """
    for (i, item) in enumerate(vec₁)
        @inbounds vec₂[i] = item
    end
end

function copy_to_vec_from_idx!(vec₁::Vector{T}, vec₂::Vector{T}, idx::UInt8) where T
    """ Copy items from vec₁ into vec₂.
        It is assumed that length(vec₂) >= length(vec₁)
    """
    for i = idx:length(vec₁)
        @inbounds vec₂[i] = vec₁[i]
    end
end

###

function Base.isequal(node₁::Node, node₂::Node)
    node₁.hash == node₂.hash
end

function Base.hashindex(node::Node, sz)::Int
    (((node.hash %Int) & (sz-1)) + 1)
end

###

function readable(edge::NodeEdge)::String
    "NodeEdge(" * string(Int64(edge.edge₁)) * " -> " * string(Int64(edge.edge₂)) * ")"
end

function readable(arr::Vector{UInt8})::Array{Int64, 1}
    Array{Int, 1}([Int64(x) for x in arr])
end

function readable(cc::UInt8)::Int64
    Int64(cc)
end

function readable(fp::ForbiddenPair)::String
    "ForbiddenPair(" * string(Int64(fp.comp₁)) * " -> " * string(Int64(fp.comp₂)) * ")"
end

function readable(fps::Vector{ForbiddenPair})::Vector{String}
    readable_vec = Vector{String}([])
    for fp in fps
        push!(readable_vec, readable(fp))
    end
    readable_vec
end
