module GeoOceanModel

# ==============================================================================
#  GeoOceanModel
#
#  Phase 6 of the ParticleTracking split (SPLIT_PLAN.md 1). This package owns the physical
#  ocean model: grid, bathymetry, vertical coordinates, hydrodynamics, tides, the forcing
#  catalogue, and the ocean figures. It writes the handover described in
#  `src/model/simulation_result.jl` and imports nothing from GeoLarvalTracking.
#
#  Everything here was moved from `ParticleTracking/src` unchanged, apart from the module name
#  and the paths in `include`. Nothing was renamed: a reader looking for `build_shelf_grid` or
#  `run_hydrodynamic_simulation!` finds it here under the same name, because renaming on a move
#  is how a caller of the old name is told the function never existed.
#
#  The two packages share nothing but GeoData. Measured cross-domain call edges between the
#  ocean half and the biology half of the original source: zero (SPLIT_PLAN.md 7d).
# ==============================================================================

using
    Random,
    CairoMakie,
    CUDA,
    Adapt,
    NCDatasets,
    Downloads,
    DataFrames,
    Dates,
    Statistics,
    LinearAlgebra,
    TOML,
    JLD2,
    TaylorSeries,
    Interpolations,
    NumericalEarth,
    Distributions,
    GeoData

# The provenance registry, as a submodule so its names do not have to be re-exported one by
# one. Unchanged from ParticleTracking: this is the registry the run reports through.
include("data/manifest.jl")
using .Manifest

using Oceananigans.Units
using Oceananigans.Utils: prettytime
using Oceananigans
using JSON3

# ============================================================================
# Source layout, in dependency order. The figure families are included by
# `output/visualization.jl` itself, so they are NOT listed here: a double include
# redefines every method on the same line, which Julia refuses during precompilation.
# ============================================================================
include("config/configuration.jl")
include("data/open_data.jl")
include("utils/architecture.jl")
include("utils/tides.jl")
include("data/grid_bathymetry.jl")
include("data/vertical_grid.jl")
include("model/hydrodynamic_model.jl")
include("model/tidal_forcing.jl")
include("model/climate_scenarios.jl")
include("model/simulation.jl")
include("model/simulation_result.jl")
include("output/geodata_output.jl")
include("output/visualization.jl")
include("data/geodata/geodata_integration.jl")
# The ocean-half run segments, as library code. Included last so every name its
# bodies reach for is already defined; it adds the run driver the CLI dispatches to.
include("driver.jl")
end # module