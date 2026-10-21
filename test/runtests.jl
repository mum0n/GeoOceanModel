# ==============================================================================
# GeoOceanModel Comprehensive Test Suite Orchestrator
# ==============================================================================

import Pkg

const RUNNING_UNDER_PKG_TEST = basename(dirname(Base.active_project())) == "test"
if !RUNNING_UNDER_PKG_TEST
    Pkg.activate(joinpath(@__DIR__, ".."); io = devnull)
    Pkg.instantiate(; io = devnull)
end

# Usage:
#   julia --project=. test/runtests.jl                # Run every segment
#   julia --project=. test/runtests.jl store          # Run output store tests
#   julia --project=. test/runtests.jl grid           # Run grid & land mask tests
#   julia --project=. test/runtests.jl animation      # Run animation diagnostics
#
# Under `Pkg.test()` the whole suite runs; pass segments as test_args:
#   Pkg.test(test_args = ["grid", "animation"])
# ==============================================================================

using Test
using GeoOceanModel
using GeoData
using GeoDataSources
using DataFrames
using Oceananigans
using JSON3
using Zarr

const AVAILABLE_SEGMENTS = Dict{Symbol, @NamedTuple{file::String, desc::String}}(
    :store => (
        file = "test_store.jl",
        desc = "Output Store Callback, Re-run Isolation & Zarr Persistence"
    ),
    :grid => (
        file = "test_grid_mask.jl",
        desc = "Handover Contract, Grid Cell Centres & Land Mask Verification"
    ),
    :animation => (
        file = "test_animation.jl",
        desc = "Animation Diagnostics, Staggered Speed & Depth Invariants"
    ),
    :mosaic => (
        file = "test_mosaic.jl",
        desc = "Multi-Domain Stitching & Mosaic Blending"
    ),
    :sponge => (
        file = "test_sponge.jl",
        desc = "Lateral Sponge Relaxation"
    )
)

"""
    parse_requested_segments(args::Vector{String}) -> Vector{Symbol}

Resolves CLI segment arguments to segment keys.
"""
function parse_requested_segments(args::Vector{String})::Vector{Symbol}
    if isempty(args)
        return collect(keys(AVAILABLE_SEGMENTS))
    end

    selected = Symbol[]
    for a in args
        key = Symbol(lowercase(strip(a)))
        if haskey(AVAILABLE_SEGMENTS, key)
            push!(selected, key)
        else
            avail = sort(string.(collect(keys(AVAILABLE_SEGMENTS))))
            @warn "Unrecognized test segment: '$a'. Available segments: $(join(avail, ", "))"
        end
    end

    return isempty(selected) ? sort!(collect(keys(AVAILABLE_SEGMENTS))) : unique(selected)
end

requested_segments = parse_requested_segments(ARGS)

@testset "GeoOceanModel Comprehensive Suite" begin
    for seg_key in requested_segments
        info = AVAILABLE_SEGMENTS[seg_key]
        println("\n" * "="^80)
        println(">>> Running Segment: $(info.desc) ($(info.file))")
        println("="^80)
        include(joinpath(@__DIR__, info.file))
    end
end
