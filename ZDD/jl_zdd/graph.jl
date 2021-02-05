# graph.jl

"""
a graph which needs the following features

each node has two outgoing edges

can be implemented as a vector of immutable structs
the immutable struct has a zero node and one node
it has only two fields

when adding new node, simply push!
never need to remove node

when merging, that edge will simply refer to idx of previously created node
the actual node will still reside in the N set

what i could also do -
"""

struct ZDD_Node
    zero::Int
    one::Int
end
