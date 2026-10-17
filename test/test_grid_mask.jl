using Test
using GeoOceanModel
using GeoData
using GeoDataSources
using DataFrames

const OCEAN_TMP_ENV = joinpath(@__DIR__, "outputs", "_test_geoceanmodel_env")

@testset "Grid, Handover & Land Mask" begin
    if ispath(OCEAN_TMP_ENV)
        rm(OCEAN_TMP_ENV; recursive = true, force = true)
    end
    mkpath(OCEAN_TMP_ENV)

    # Handover contract
    local ck = joinpath(OCEAN_TMP_ENV, "hydro.jld2")
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
    @test_throws Exception GeoOceanModel.open_simulation_result(
        joinpath(OCEAN_TMP_ENV, "absent.jld2"))
    @test_throws Exception GeoOceanModel.open_simulation_result(
        ck; required = (:domain_lon, :nonsense))

    # Configuration: preferred bathymetry source
    local scotian_lon = (-68.0, -57.0)
    local scotian_lat = (42.0, 47.5)
    @test GeoOceanModel.preferred_bathymetry_source(
        source = "etopo2022", lon_range = scotian_lon, lat_range = scotian_lat) == :etopo
    @test GeoOceanModel.preferred_bathymetry_source(
        lon_range = scotian_lon, lat_range = scotian_lat) == :nwa
    @test GeoOceanModel.preferred_bathymetry_source(
        lon_range = (-40.0, -30.0), lat_range = (50.0, 55.0)) == :etopo
    @test_throws Exception GeoOceanModel.preferred_bathymetry_source(source = "nonsense")

    # Grid cell-centre fix
    local g = GeoOceanModel.build_shelf_grid(
        lon_range = (-65.0, -60.0), lat_range = (43.0, 46.0),
        z_range = (-500.0, 0.0), grid_size = (20, 20, 5))
    local tlons, tlats = GeoOceanModel._grid_interior_centres(g)
    @test length(tlons) == g.Nx && length(tlats) == g.Ny
    @test tlons == Float64.([g.λᶜᵃᵃ[i] for i in 1:g.Nx])
    @test first(tlons) != g.λᶠᵃᵃ[1]
    @test abs(first(tlons) - g.λᶠᵃᵃ[1]) > 0

    # Land mask coastline
    local dir = joinpath(OCEAN_TMP_ENV, "coast")
    mkpath(dir)
    local ring_lons = Float64[-60.0, -60.0, -59.0, -59.0, -60.0]
    local ring_lats = Float64[44.0, 45.0, 45.0, 44.0, 44.0]
    local line = GeoData.GeoInterface.LineString(
        [(ring_lons[i], ring_lats[i]) for i in eachindex(ring_lons)])
    GeoData.GeoParquet.write(
        joinpath(dir, "coastline.parquet"),
        DataFrame(id = [1], name = ["test ring"], code = ["test_ring"], geometry = [line]),
        (:geometry,))

    local from_dir = GeoOceanModel.resolve_land_mask_coastline(nothing; coastline_path = dir)
    @test length(from_dir) == 1
    @test from_dir[1].lons == ring_lons
    @test from_dir[1].lats == ring_lats
    @test GeoOceanModel.resolve_land_mask_coastline(
        nothing; coastline_path = joinpath(dir, "coastline.parquet")) == from_dir

    local bare = joinpath(OCEAN_TMP_ENV, "no_coast")
    mkpath(bare)
    local bundled = GeoDataSources.SCOTIAN_SHELF_COASTLINE
    @test GeoOceanModel.resolve_land_mask_coastline(nothing; coastline_path = bare) == bundled
    @test GeoOceanModel.resolve_land_mask_coastline(nothing) == bundled

    local explicit = NamedTuple[
        (name = "x", code = :x, lons = Float64[0.0, 1.0, 1.0, 0.0], lats = Float64[0.0, 0.0, 1.0, 1.0])
    ]
    @test GeoOceanModel.resolve_land_mask_coastline(explicit; coastline_path = dir) === explicit

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
