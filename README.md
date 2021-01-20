# trees, leaves, fall foliage, etc

Checklist when pushing:
* When adding features/improvements, has the `weightless` ZDD codebase
also been updated?

## How to run the ZDD program:  
This project is in active development.

The main file is `zdd.jl` in `ZDD/jl_zdd/zdd.jl`. An example of a script that
runs the script is in `ZDD/jl_zdd/run_algo.jl`.

There is a sister `weightless_zdd.jl` that is very similar to `zdd.jl` but without weighting the components.
This implementation usually runs faster than `zdd.jl`, but can only produce all possible ways to partition a graph, whereas
the `zdd.jl` one produces all possible population balanced partitions of a graph.

### Testing
Testing is done by running `julia test_zdd.jl`, which tests for correctness for small rook and queen graphs.

### Visualization
The ZDDs can be visualized and plotted by calling `construct_zdd(..... , viz=true)`. Then, you can
plot the ZDD by calling `draw(zdd, graph, graph_edges)`. Constructing a ZDD with `viz=true` is more memory+time
intensive because we will be storing things specifically needed for visualization, and visualization gets
very complicated for large graphs anyway. It is recommended that *this option be turned off* when building ZDDs
larger than 4x4 grids, and that the user only attempt visualizing graphs less than the size of 4x4 for best performance.

### Ongoing + Planned efforts in the project
* Identify "dead" nodes (nodes that do not lead to a valid partition) early on in the tree.
* Identify best edge orderings
* Delete the previous layers of the ZDD from memory when we are done with them
* Use parallelization for speedups
* Make the size of the `Node` smaller
* Merge every `i` layers?
