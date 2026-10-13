"""
    runtests.jl -- the ocean half's suite

Covers what `GeoOceanModel` owns: the output store, the handover contract, the grid and
immersed boundary, and the configuration.

What is deliberately NOT here: anything that needs a network. `preferred_bathymetry_source`
is tested because it is a decision function; the fetchers themselves are exercised by
`GeoDataSources`' own suite, and this package's `bathymetry_dataset` is a thin dispatcher over
them. A test that downloads is a test that fails on a plane.

`Pkg.test` can reach every function here, which is the point: the umbrella driver's functions
lived in a script and could not be tested.
"""

using Test
using GeoOceanModel
using Oceananigans
using JSON3
using Zarr

const OCEAN_TMP = joinpath("outputs", "_test_geoceanmodel")

if ispath(OCEAN_TMP)
    rm(OCEAN_TMP; recursive = true, force = true)
end
mkpath(OCEAN_TMP)

@testset "GeoOceanModel" begin

    @testset "output store: the run writes its own slices and axes" begin
        local mk = Oceananigans.RectilinearGrid(size = (4, 4, 2), extent = (4, 4, 2))
        local model = Oceananigans.HydrostaticFreeSurfaceModel(mk; tracers = (:T,))
        set!(model, T = 13.5)

        local store = GeoOceanModel.attach_output_callback!(
            Oceananigans.Simulation(model; Δt = 1.0, stop_time = 2.0), OCEAN_TMP;
            fields = ("u", "T"))
        @test store == GeoOceanModel.output_zarr_path(OCEAN_TMP)
        @test isdir(store)

        # The grid's cell centres travel inside the store, not in a sidecar. This is the
        # property that let the sidecar go.
        local a = JSON3.read(read(joinpath(store, ".zattrs"), String))
        @test haskey(a, "lon") && haskey(a, "lat") && haskey(a, "depth")
        @test length(a["lon"]) == 4
        @test haskey(a, "dlon")

        local sim = Oceananigans.Simulation(model; Δt = 1.0, stop_time = 2.0)
        GeoOceanModel.attach_output_callback!(sim, OCEAN_TMP; fields = ("u", "T"))
        Oceananigans.run!(sim)

        # One array per interval, and one time axis for the whole store.
        @test length(readdir(joinpath(store, "T"))) == length(readdir(joinpath(store, "u")))
        local times = JSON3.read(read(joinpath(store, "time.json"), String))
        @test !isempty(times)

        # A value read back is the value written.
        local T0 = collect(Zarr.zopen(joinpath(store, "T", "0")))
        @test size(T0) == (4, 4, 2)
        @test T0[1, 1, 1] == 13.5
    end

    @testset "output store: a re-run replaces rather than appends" begin
        local d = joinpath(OCEAN_TMP, "rerun")
        # A fresh model per run, because a model's clock survives `run!` and a second
        # Simulation on the same model starts where the first left off -- which would make
        # this test's interval counts meaningless rather than wrong.
        local function one_run(value)
            local mk = Oceananigans.RectilinearGrid(size = (4, 4, 2), extent = (4, 4, 2))
            local model = Oceananigans.HydrostaticFreeSurfaceModel(mk; tracers = (:T,))
            set!(model, T = value)
            local sim = Oceananigans.Simulation(model; Δt = 1.0, stop_time = 1.0)
            GeoOceanModel.attach_output_callback!(sim, d; fields = ("T",))
            Oceananigans.run!(sim)
            return length(readdir(joinpath(GeoOceanModel.output_zarr_path(d), "T")))
        end
        local first_n = one_run(10.0)
        @test first_n > 0
        # A second run into the same directory must not grow the first run's store: if it
        # appended, the count would be first_n + this run's own writes.
        @test one_run(20.0) == first_n
    end

    @testset "handover contract: absent and incomplete are both errors" begin
        local ck = joinpath(OCEAN_TMP, "hydro.jld2")
        local meta = Dict{Symbol, Any}(
            :domain_lon => (-68.0, -57.0), :domain_lat => (42.0, 47.5),
            :z_range => (-500.0, 0.0), :bathy_source => "scotian_shelf",
            :bathymetry_tier => "discretized", :realized_domain_lon => (-67.875, -57.125),
            :realized_domain_lat => (42.125, 47.375), :realized_grid_spacing_lon => 0.25,
            :realized_grid_spacing_lat => 0.25, :ocean_parameters_resolved => true,
            :simulation_start_time => "2023-06-01T00:00:00",
            :simulation_end_time => "2023-06-30T23:59:59", :schema_version => "5")

        local result = GeoOceanModel.write_simulation_result(ck, ck, meta)
        @test result isa GeoOceanModel.SimulationResult
        @test GeoOceanModel.ocean_parameters(result) == meta

        # A location that does not exist is an error, never a synthetic stand-in.
        @test_throws Exception GeoOceanModel.open_simulation_result(
            joinpath(OCEAN_TMP, "absent.jld2"))

        # A location whose parameters are incomplete errors naming the missing key.
        @test_throws Exception GeoOceanModel.open_simulation_result(
            ck; required = (:domain_lon, :nonsense))
    end

    @testset "configuration: a named source and the coverage rule" begin
        # An explicit source wins; an unnamed one is resolved by coverage.
        local scotian_lon = (-68.0, -57.0)
        local scotian_lat = (42.0, 47.5)
        @test GeoOceanModel.preferred_bathymetry_source(
            source = "etopo2022", lon_range = scotian_lon, lat_range = scotian_lat) == :etopo
        @test GeoOceanModel.preferred_bathymetry_source(
            lon_range = scotian_lon, lat_range = scotian_lat) == :nwa
        # Outside every regional box the global model is the failsafe.
        @test GeoOceanModel.preferred_bathymetry_source(
            lon_range = (-40.0, -30.0), lat_range = (50.0, 55.0)) == :etopo
        # A misspelling is an error, not a silent fallback to ETOPO.
        @test_throws Exception GeoOceanModel.preferred_bathymetry_source(source = "nonsense")
    end

    @testset "grid: the cell-centre fix has not regressed" begin
        # The property that was wrong before: sampling points must be the grid's interior
        # cell centres, not the face extent, or the sea floor lands half a cell off.
        local g = GeoOceanModel.build_shelf_grid(
            lon_range = (-65.0, -60.0), lat_range = (43.0, 46.0),
            z_range = (-500.0, 0.0), grid_size = (20, 20, 5))
        local tlons, tlats = GeoOceanModel._grid_interior_centres(g)
        @test length(tlons) == g.Nx && length(tlats) == g.Ny
        @test tlons == Float64.([g.λᶜᵃᵃ[i] for i in 1:g.Nx])
        # The old, wrong window: the interior FACE extent, which is half a cell displaced.
        @test first(tlons) != g.λᶠᵃᵃ[1]
        @test abs(first(tlons) - g.λᶠᵃᵃ[1]) > 0
    end
end

rm(OCEAN_TMP; recursive = true, force = true)
