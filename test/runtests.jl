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
using GeoData
using GeoDataSources
using DataFrames
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
    @testset "land mask: the run's own coastline outranks the bundled one" begin
        # A closed ring written as GeoParquet exactly the way the fetcher writes one, so
        # the reader path is exercised rather than mocked.
        local dir = joinpath(OCEAN_TMP, "coast")
        mkpath(dir)
        local ring_lons = Float64[-60.0, -60.0, -59.0, -59.0, -60.0]
        local ring_lats = Float64[44.0, 45.0, 45.0, 44.0, 44.0]
        local line = GeoData.GeoInterface.LineString(
            [(ring_lons[i], ring_lats[i]) for i in eachindex(ring_lons)])
        GeoData.GeoParquet.write(
            joinpath(dir, "coastline.parquet"),
            DataFrame(id = [1], name = ["test ring"], code = ["test_ring"], geometry = [line]),
            (:geometry,))

        # The rings the run fetched are the ones the mask is drawn against, and naming
        # the file instead of its directory resolves the same rings.
        local from_dir = GeoOceanModel.resolve_land_mask_coastline(nothing; coastline_path = dir)
        @test length(from_dir) == 1
        @test from_dir[1].lons == ring_lons
        @test from_dir[1].lats == ring_lats
        @test GeoOceanModel.resolve_land_mask_coastline(
            nothing; coastline_path = joinpath(dir, "coastline.parquet")) == from_dir

        # A directory with no coastline in it, and no path at all, are the two cases
        # where the region's bundled rings are used.
        local bare = joinpath(OCEAN_TMP, "no_coast")
        mkpath(bare)
        local bundled = GeoDataSources.SCOTIAN_SHELF_COASTLINE
        @test GeoOceanModel.resolve_land_mask_coastline(nothing; coastline_path = bare) == bundled
        @test GeoOceanModel.resolve_land_mask_coastline(nothing) == bundled

        # Explicit rings outrank every path.
        local explicit = NamedTuple[
            (name = "x", code = :x, lons = Float64[0.0, 1.0, 1.0, 0.0], lats = Float64[0.0, 0.0, 1.0, 1.0])
        ]
        @test GeoOceanModel.resolve_land_mask_coastline(explicit; coastline_path = dir) === explicit

        # The synthetic branch honours the same order: the run's ring masks its own
        # interior, and a caller holding an empty ring set is masked nowhere.
        local masked = GeoOceanModel.extract_hydrodynamic_dataset(
            nothing;
            domain_lon = (-61.0, -58.0), domain_lat = (43.0, 46.0),
            grid_size = (11, 11), coastline_path = dir)
        @test any(isnan, masked.speed)
        local unmasked = GeoOceanModel.extract_hydrodynamic_dataset(
            nothing;
            domain_lon = (-61.0, -58.0), domain_lat = (43.0, 46.0),
            grid_size = (11, 11), coastline = NamedTuple[])
        @test !any(isnan, unmasked.u)
    end

    @testset "animation: one variable, one depth, from a run's store" begin
        # The store is written by hand rather than by a run, so this checks the reading and
        # the depth choice without a simulation. `u` is shaped (n_x + 1, n_y, n_z) on purpose:
        # that is the shape the writer puts on disk, and the surface level is the LAST one in
        # the store's depth axis -- both of which the animation has to get right.
        local d = joinpath(OCEAN_TMP, "anim_store")
        local nx, ny, nz, nstep = 5, 4, 3, 4
        local store = GeoOceanModel.output_zarr_path(d)
        mkpath(store)
        open(joinpath(store, ".zgroup"), "w") do io
            write(io, "{\"zarr_format\": 2}")
        end
        open(joinpath(store, ".zattrs"), "w") do io
            JSON3.write(io, Dict{String, Any}(
                "fields" => ["u", "T"],
                "title" => "fixture",
                "lon" => collect(1.0:nx),
                "lat" => collect(1.0:ny),
                # Ascending magnitude means the SHALLOWEST level is last, as the run writes it.
                "depth" => [-30.0, -10.0, -2.5],
                "dlon" => 0.1, "dlat" => 0.1))
        end
        open(joinpath(store, "time.json"), "w") do io
            JSON3.write(io, Float64.(0:3600:(3600 * (nstep - 1))))
        end
        for (name, shp) in (("u", (nx + 1, ny, nz)), ("T", (nx, ny, nz)))
            for m in 0:(nstep - 1)
                local dir = joinpath(store, name, string(m))
                # The same call the writer makes, so a store written by `zcreate` and a store
                # written here are the same thing to a reader. Writing `.zarray` by hand does
                # not work: Zarr reverses the declared shape and, without its
                # `dimension_separator`, looks for a chunk key this layout does not use.
                local z = Zarr.zcreate(Float64, Zarr.DirectoryStore(dir), shp...;
                                       fill_value = 0.0)
                # Distinct along every axis, so each of the checks below can only pass for
                # the right interval, the right depth, and the right staggering. Interval 0
                # is offset by one as well, so no interval's value is zero: a zero-valued
                # field is impossible to tell apart from an unread one.
                for k in 1:shp[3], j in 1:shp[2], i in 1:shp[1]
                    z[i, j, k] = 1.0 + m + i + 10k
                end
            end
        end

        # A store-level variable is animated from the store, and the surface default is the
        # shallowest level, not index 1.
        local out = joinpath(OCEAN_TMP, "anim.gif")
        local path = GeoOceanModel.animate_variable(
            d; output_path = out, frames = 2, domain_lon = (0.5, nx + 0.5),
            domain_lat = (0.5, ny + 0.5))
        @test isfile(path) && filesize(path) > 0

        # The surface level of this store is -2.5 m, and it is the LAST of the three: the
        # depth axis ascends from the bottom, so index 1 is the seafloor. `:surface` must
        # resolve to the shallowest level, not to index 1 -- which is what the figure layer
        # does on its own and is the trap this default exists to avoid.
        local depths = collect(Float64, [-30.0, -10.0, -2.5])
        @test argmin(abs.(depths)) == nz
        local val_at_level(i, k) = 1.0 + 0 + i + 10 * k

        local src = GeoOceanModel._animation_source(d, :temperature, 4)
        local frame = GeoOceanModel.extract_hydrodynamic_dataset(src; time_index = 1)
        @test frame.depths == depths
        # The value is the one written for that level and that interval, not a fabricated
        # field: interval 0, shallowest level.
        @test frame.temperature[nx, ny, nz] == val_at_level(nx, nz)
        # And the deepest level is a different value, so the level choice is observable.
        @test frame.temperature[nx, ny, 1] != frame.temperature[nx, ny, nz]

        # Speed is built from u and v, and `u` is stored on x-faces: one column wider than
        # the grid. The centred value is the mean of the two faces that bracket it, not the
        # first face and not the last. The fixture's value is `1 + m + i + 10k`, so the mean
        # at centre i is exactly `i + 0.5` plus the constant.
        local speed_src = GeoOceanModel._animation_source(d, :speed, 4)
        local speed_frame = GeoOceanModel.extract_hydrodynamic_dataset(
            speed_src; time_index = 1)
        @test size(speed_frame.u) == (nx, ny, nz)
        @test speed_frame.u[1, 1, 1] == 0.5 * (val_at_level(1, 1) + val_at_level(2, 1))
        # The last centre brackets the last two faces, so the array's extra column is used
        # rather than dropped at the end.
        @test speed_frame.u[end, 1, 1] == 0.5 * (val_at_level(nx, 1) + val_at_level(nx + 1, 1))

        # A variable the store cannot supply is an error naming what it does carry, rather
        # than a frame of zeros.
        @test_throws Exception GeoOceanModel.animate_variable(
            d; variable = :salinity, output_path = out, frames = 2)

        # A path naming neither a store nor a file is refused. The figure layer's fallback
        # for an unrecognised source is a synthetic field, and animating one of those would
        # show a run that does not exist.
        @test_throws Exception GeoOceanModel.animate_variable(
            joinpath(OCEAN_TMP, "nope.zarr"); output_path = out, frames = 2)
    end
end
