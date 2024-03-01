"""
```julia
using VectorPlot

p1 = (0, 0)
p2 = (3, 4)

plot_vec(p1, p2, color=:red, linewidth=2)
```
"""
module VectorPlot

using Plots

export plot_vec

function plot_vec(p1::Tuple, p2::Tuple; kwargs...)
    # Extract coordinates of the points
    x1, y1 = p1
    x2, y2 = p2

    # Compute components of the vector
    dx = round(x2 - x1,digits=3)
    dy = round(y2 - y1,digits=3)

    # Create a plot with arrows representing the vector
    plot!([x1, x2], [y1, y2]; kwargs..., arrow=true)
    annotate!([(x2, y2, text("($dx, $dy)", 8, :left))])
end

end  # module
