# ==============================================================================
# test_mosaic.jl -- Multi-Domain Stitching & Mosaic Blending Verification
# ==============================================================================

using Test
using GeoOceanModel
using GeoData
using Zarr

@testset "Multi-Domain Mosaic Stitching" begin
    test_dir = mktempdir()
    tile_w_path = joinpath(test_dir, "tile_west.zarr")
    tile_e_path = joinpath(test_dir, "tile_east.zarr")
    mosaic_path = joinpath(test_dir, "mosaic.zarr")

    # Construct synthetic western store: lon in [-67.0, -63.0] (Nx=9), lat in [43.0, 45.0] (Ny=5), z in [-100.0, 0.0] (Nz=3)
    lon_w = collect(range(-67.0, -63.0, length = 9))
    lat_w = collect(range(43.0, 45.0, length = 5))
    z_w   = [-100.0, -50.0, 0.0]
    t_vals = [0.0, 3600.0]

    # Construct synthetic eastern store: lon in [-63.5, -57.0] (Nx=14), lat in [43.0, 45.0] (Ny=5), z in [-100.0, 0.0] (Nz=3)
    lon_e = collect(range(-63.5, -57.0, length = 14))
    lat_e = collect(range(43.0, 45.0, length = 5))
    z_e   = [-100.0, -50.0, 0.0]

    ax_w = (x = lon_w, y = lat_w, z = z_w)
    ax_e = (x = lon_e, y = lat_e, z = z_e)

    # Build stores using GeoOceanModel internal store dataset builder
    ds_w = GeoOceanModel._build_store_dataset(tile_w_path, ax_w, ["u", "T"], length(t_vals), nothing)
    gw = Zarr.zopen(tile_w_path, "w")
    for (m, t) in enumerate(t_vals)
        gw.arrays["time"][m] = t
    end

    ds_e = GeoOceanModel._build_store_dataset(tile_e_path, ax_e, ["u", "T"], length(t_vals), nothing)
    ge = Zarr.zopen(tile_e_path, "w")
    for (m, t) in enumerate(t_vals)
        ge.arrays["time"][m] = t
    end

    # Populate fields: Tile W has T = 5.0, Tile E has T = 15.0
    for m in 1:length(t_vals)
        ds_w.variables["T"].data[:, :, :, m] .= 5.0f0
        ds_w.variables["u"].data[:, :, :, m] .= -0.1f0
        ds_e.variables["T"].data[:, :, :, m] .= 15.0f0
        ds_e.variables["u"].data[:, :, :, m] .= 0.2f0
    end

    # Execute stitching
    res_path = stitch_shelf_domains(
        tile_w_path, tile_e_path;
        output_path = mosaic_path,
        blend_lon_range = (-63.5, -63.0)
    )

    @test isdir(mosaic_path)
    @test res_path == mosaic_path

    # Verify blended mosaic store
    ds_m = GeoData.geoload(mosaic_path; backend = :zarr)
    @test haskey(ds_m.variables, "T")
    @test haskey(ds_m.variables, "u")

    lon_m = ds_m.coords[:lon].data
    @test issorted(lon_m)
    @test minimum(lon_m) ≈ -67.0
    @test maximum(lon_m) ≈ -57.0

    T_data = ds_m.variables["T"].data
    @test all(!isnan, T_data)

    # In western domain (lon < -63.5), T must equal 5.0
    i_west = findall(x -> x < -63.5, lon_m)
    @test all(isapprox.(T_data[i_west, :, :, :], 5.0f0, atol = 1e-4))

    # In eastern domain (lon > -63.0), T must equal 15.0
    i_east = findall(x -> x > -63.0, lon_m)
    @test all(isapprox.(T_data[i_east, :, :, :], 15.0f0, atol = 1e-4))

    # In overlap blend zone, T must be strictly between 5.0 and 15.0
    i_blend = findall(x -> -63.5 <= x <= -63.0, lon_m)
    @test length(i_blend) >= 1
    @test all(5.0f0 .<= T_data[i_blend, :, :, :] .<= 15.0f0)

    # Clean up temporary test files
    rm(test_dir, recursive = true, force = true)
end
