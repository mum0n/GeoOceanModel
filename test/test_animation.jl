using Test
using GeoOceanModel
using GeoData
using Zarr

const OCEAN_TMP_ANIM = joinpath(@__DIR__, "outputs", "_test_geoceanmodel_anim")

@testset "Animation & Diagnostics" begin
    if ispath(OCEAN_TMP_ANIM)
        rm(OCEAN_TMP_ANIM; recursive = true, force = true)
    end
    mkpath(OCEAN_TMP_ANIM)

    local d = joinpath(OCEAN_TMP_ANIM, "anim_store")
    local nx, ny, nz, nstep = 5, 4, 3, 4
    local store = GeoOceanModel.output_zarr_path(d)
    local lons = collect(1.0:nx)
    local lats = collect(1.0:ny)
    local depths = [-30.0, -10.0, -2.5]
    local times = Float64.(0:3600:(3600 * (nstep - 1)))

    ds = GeoOceanModel._build_store_dataset(
        store, (x = lons, y = lats, z = depths), ["u", "T"], nstep, nothing)
    for m in 1:nstep
        for k in 1:nz, j in 1:ny, i in 1:nx
            ds.variables["T"].data[i, j, k, m] = 1.0 + m + i + 10k
            ds.variables["u"].data[i, j, k, m] = 1.0 + m + i + 10k
        end
    end
    Zarr.zopen(store, "w").arrays["time"][:] = times

    local out = joinpath(OCEAN_TMP_ANIM, "anim.gif")
    local path = GeoOceanModel.animate_variable(
        d; output_path = out, frames = 2, domain_lon = (0.5, nx + 0.5),
        domain_lat = (0.5, ny + 0.5))
    @test isfile(path) && filesize(path) > 0

    @test argmin(abs.(depths)) == nz
    local val_at_level(i, k) = 1.0 + 1 + i + 10 * k

    local src = GeoOceanModel._animation_source(d, :temperature, 4)
    local frame = GeoOceanModel.extract_hydrodynamic_dataset(src; time_index = 1)
    @test frame.depths == depths
    @test frame.temperature[nx, ny, nz] == val_at_level(nx, nz)
    @test frame.temperature[nx, ny, 1] != frame.temperature[nx, ny, nz]

    local speed_src = GeoOceanModel._animation_source(d, :speed, 4)
    local speed_frame = GeoOceanModel.extract_hydrodynamic_dataset(
        speed_src; time_index = 1)
    @test size(speed_frame.u) == (nx, ny, nz)
    @test speed_frame.u[1, 1, 1] == val_at_level(1, 1)
    @test speed_frame.u[end, 1, 1] == val_at_level(nx, 1)

    @test_throws Exception GeoOceanModel.animate_variable(
        d; variable = :salinity, output_path = out, frames = 2)

    @test_throws Exception GeoOceanModel.animate_variable(
        joinpath(OCEAN_TMP_ANIM, "nope.zarr"); output_path = out, frames = 2)
end
