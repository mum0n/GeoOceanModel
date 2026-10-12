"""
    visualization.jl

CairoMakie figures and a standalone Leaflet.js dashboard for the ocean model, assembled from
the families in `figures/`. Each family has its own file, so a figure cannot depend silently on
how a field was resolved and adding a family does not grow the whole.

`figures/hydrodynamic_access.jl` is included first because every other ocean family reads
through it. The larval figure family is not here: it belongs to `GeoLarvalTracking`, and this
package imports nothing from it (SPLIT_PLAN.md 1).
"""

using CairoMakie
import CairoMakie: record

include(joinpath(@__DIR__, "figures", "hydrodynamic_access.jl"))
include(joinpath(@__DIR__, "figures", "hydrodynamic_figures.jl"))
include(joinpath(@__DIR__, "figures", "interactive_html.jl"))
include(joinpath(@__DIR__, "figures", "animations.jl"))
include(joinpath(@__DIR__, "figures", "shared_figures.jl"))
