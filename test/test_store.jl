using Test
using GeoOceanModel
using GeoData
using Oceananigans

const OCEAN_TMP = joinpath(@__DIR__, "outputs", "_test_geoceanmodel")

@testset "Output Store" begin
    if ispath(OCEAN_TMP)
        rm(OCEAN_TMP; recursive = true, force = true)
    end
    mkpath(OCEAN_TMP)

    local mk = Oceananigans.RectilinearGrid(size = (4, 4, 2), extent = (4, 4, 2))
    local model = Oceananigans.HydrostaticFreeSurfaceModel(mk; tracers = (:T,))
    set!(model, T = 13.5)

    local store = GeoOceanModel.attach_output_callback!(
        Oceananigans.Simulation(model; Δt = 1.0, stop_time = 2.0), OCEAN_TMP;
        fields = ("u", "T"))
    @test store == GeoOceanModel.output_zarr_path(OCEAN_TMP)

    local sim = Oceananigans.Simulation(model; Δt = 1.0, stop_time = 2.0)
    GeoOceanModel.attach_output_callback!(sim, OCEAN_TMP; fields = ("u", "T"))
    Oceananigans.run!(sim)
    @test isdir(store)

    local ds = GeoData.geoload(store; backend = :zarr)
    @test "T" in keys(ds.variables) && "u" in keys(ds.variables)
    @test length(ds.dims[:lon].coords) == 4
    @test length(ds.dims[:lat].coords) == 4
    @test length(ds.dims[:depth].coords) == 2
    @test !isempty(ds.coords[:time].data)
    @test size(ds.variables["T"].data) == size(ds.variables["u"].data)

    local T0 = ds.variables["T"].data[:, :, :, 1]
    @test size(T0) == (4, 4, 2)
    @test T0[1, 1, 1] == 13.5

    # Re-run replaces rather than appends
    local d = joinpath(OCEAN_TMP, "rerun")
    local function one_run(value)
        local mk_r = Oceananigans.RectilinearGrid(size = (4, 4, 2), extent = (4, 4, 2))
        local model_r = Oceananigans.HydrostaticFreeSurfaceModel(mk_r; tracers = (:T,))
        set!(model_r, T = value)
        local sim_r = Oceananigans.Simulation(model_r; Δt = 1.0, stop_time = 1.0)
        GeoOceanModel.attach_output_callback!(sim_r, d; fields = ("T",))
        Oceananigans.run!(sim_r)
        return sum(length(GeoData.geoload(
            GeoOceanModel.output_zarr_path(d); backend = :zarr).coords[:time].data))
    end
    local first_n = one_run(10.0)
    @test first_n > 0
    @test one_run(20.0) == first_n
end
