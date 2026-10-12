"""
    open_data.jl

ParticleTracking-specific data acquisition functions not covered by GeoDataSources.

This module contains:
- Configuration-dependent domain resolution
- Source dispatchers that use ParticleTracking configuration

All generic geospatial data operations (bathymetry, winds, WOA23, coastlines, regridding)
are in GeoDataSources, as are Copernicus Marine and synthetic data functions.
"""

using Downloads
using TOML

"""
    resolve_domain_bounds(lon_range, lat_range; embedding = false) -> NamedTuple

Resolve geographic bounds for a data-retrieval function, reading them from the active TOML
configuration when not supplied explicitly.

No geographic domain is compiled into this package: the `[domain]` section defines the study
region and the `[boundaries]` section defines the larger embedding region used for boundary
and forcing downloads. Passing `nothing` for either range therefore defers to the
configuration file, which is the single source of truth.
"""
function resolve_domain_bounds(lon_range, lat_range; embedding::Bool = false)
    rngs = if isnothing(lon_range) || isnothing(lat_range)
        cfg = try
            load_configuration(find_default_config_path())
        catch err
            error(
                "No lon_range/lat_range supplied and the active configuration could not be " *
                "read ($(err)). Supply the bounds explicitly, or fix the TOML file."
            )
        end
        embedding ? embedding_domain_ranges(cfg) : study_domain_ranges(cfg)
    else
        (lon = (Float64(lon_range[1]), Float64(lon_range[2])),
         lat = (Float64(lat_range[1]), Float64(lat_range[2])))
    end
    return rngs
end

"""
    fetch_surface_winds(source; lon_range, lat_range, time_iso, output_path, verbose) -> String

Fetch surface winds from exactly one named source, and fail loudly if it is not one this
package can actually retrieve. Currently only `open_meteo` is implemented.
"""
function fetch_surface_winds(source::AbstractString; lon_range, lat_range, time_iso,
                            output_path, verbose::Bool = true)
    s = replace(lowercase(strip(source)), "-" => "_")
    if s in ("open_meteo", "era5", "open", "openmeteo")
        return fetch_open_meteo_winds(
            lon_range = lon_range, lat_range = lat_range,
            time_iso = time_iso, output_path = output_path,
            verbose = verbose
        )
    elseif s in ("cds", "copernicus", "cdsapi")
        return GeoDataSources.fetch_copernicus_surface_winds(
            lon_range = lon_range, lat_range = lat_range,
            time_iso = time_iso, output_path = output_path,
            verbose = verbose
        )
    end
    error(_unknown_source_message("wind_source", s,
        ["open_meteo" => "ERA5 via the keyless Open-Meteo archive API (default keyless source)",
         "cds"        => "ERA5 via Copernicus Climate Data Store (requires CDS API credentials)"]))
end

"""
    fetch_bathymetry(source; lon_range, lat_range, output_path, verbose) -> String

Fetch bathymetry from exactly one named source. Currently only `etopo2022` is implemented.
"""
function fetch_bathymetry(source::AbstractString; lon_range, lat_range, output_path,
                           verbose::Bool = true)
    s = replace(lowercase(strip(source)), "-" => "_")
    if s in ("etopo2022", "etopo", "open")
        return fetch_erddap_bathymetry(
            lon_range = lon_range, lat_range = lat_range,
            output_path = output_path, verbose = verbose
        )
    end
    error(_unknown_source_message("bathy_source", s,
        ["etopo2022" => "ETOPO 2022 15-arcsec global relief via the keyless NOAA ERDDAP"]))
end

"""
    _unknown_source_message(key, given, choices) -> String

Build the error for an unrecognised `[data] *_source` value.
"""
function _unknown_source_message(key::AbstractString, given::AbstractString,
                                choices::Vector{Pair{String, String}})
    io = IOBuffer()
    println(io, "[data] $key = \"$given\" is not a source this project can retrieve.")
    println(io, "Available sources:")
    for (name, note) in choices
        println(io, "  $name - ", note)
    end
    println(io, "Set [data] $key in the TOML and re-run. Nothing was downloaded, and no ",
            "substitute was written in its place.")
    return String(take!(io))
end

"""
    TimeSeriesSurfaceStress{N}

Bitstype surface kinematic stress time-series evaluator for atmospheric boundary forcing.

Evaluates surface kinematic wind stress \\tau(t) from a discrete sequence of N values
spaced by `tstep` seconds over a cyclic period of `horizon` seconds:
```math
\\tau(t) = \\tau_{\\text{values}}[k], \\quad k = 1 + \\left\\lfloor \\frac{t \\pmod H}{\\Delta t} \\right\\rfloor
```
where H is `horizon` and \\Delta t is `tstep`.
"""
struct TimeSeriesSurfaceStress{N} <: Function
    values::NTuple{N, Float64}
    horizon::Float64
    tstep::Float64
end

Adapt.adapt_structure(to, s::TimeSeriesSurfaceStress) = s

@inline function (s::TimeSeriesSurfaceStress{N})(x, y, t) where {N}
    if s.horizon <= 0.0 || N <= 1
        return @inbounds s.values[1]
    end
    t_mod = mod(Float64(t), s.horizon)
    k = Int(floor(t_mod / s.tstep)) + 1
    k_idx = k > N ? 1 : (k < 1 ? 1 : k)
    return @inbounds s.values[k_idx]
end

@inline (s::TimeSeriesSurfaceStress{N})(x, y, z, t) where {N} = s(x, y, t)

"""
    read_wind_stress(filepath::AbstractString) -> NamedTuple

Read kinematic surface stress from a wind file via GeoData with robust NetCDF fallback.
"""
function read_wind_stress(filepath::AbstractString)
    try
        return load_wind_stress_geodata(filepath)
    catch err
        # Fallback for NetCDF files where time coordinate is decoded as DateTime
        return NCDatasets.NCDataset(filepath) do ds
            tx = Array(ds["tau_x"])
            ty = Array(ds["tau_y"])
            tx_clean = filter(!isnan, tx)
            ty_clean = filter(!isnan, ty)
            tau_x = isempty(tx_clean) ? 0.0 : Float64(Statistics.mean(tx_clean))
            tau_y = isempty(ty_clean) ? 0.0 : Float64(Statistics.mean(ty_clean))
            tau_mag = sqrt.(tx.^2 .+ ty.^2)
            tau_clean = filter(!isnan, tau_mag)
            tau_max = isempty(tau_clean) ? 0.0 : Float64(maximum(tau_clean))
            cd_rho = 1.225 * 1.3e-3
            speed10_rms = isempty(tau_clean) ? 0.0 : sqrt(Float64(Statistics.mean(tau_clean)) / cd_rho)
            return (
                tau_x = tau_x,
                tau_y = tau_y,
                tau_max = tau_max,
                speed10_rms = speed10_rms
            )
        end
    end
end

"""
    build_bulk_surface_flux(wind_file::AbstractString; albedo::Real = 0.06)

Read a surface-wind file and return `(stress_x, stress_y, heat_flux)` callables.
Uses `build_bulk_surface_flux_geodata` and wraps the horizontal stresses in
`TimeSeriesSurfaceStress` for GPU and boundary condition compatibility.
"""
function build_bulk_surface_flux(wind_file::AbstractString; albedo::Real = 0.06)
    try
        tx_arr, ty_arr, heat_flux = build_bulk_surface_flux_geodata(
            wind_file; albedo = albedo
        )
        if !isnothing(tx_arr)
            ds = geoload(wind_file)
            tvec = first(GeoData.axis(ds, :time))
            nt = size(tx_arr, 3)
            tstep = nt > 1 ? (tvec[end] - tvec[1]) / (nt - 1) : 3600.0
            horizon = nt > 1 ? tvec[end] - tvec[1] : 3600.0

            tx_tuple = ntuple(k -> Float64(tx_arr[1, 1, k]), nt)
            ty_tuple = ntuple(k -> Float64(ty_arr[1, 1, k]), nt)
            stress_x = TimeSeriesSurfaceStress{nt}(tx_tuple, horizon, tstep)
            stress_y = TimeSeriesSurfaceStress{nt}(ty_tuple, horizon, tstep)

            return (stress_x, stress_y, heat_flux)
        end
    catch
        # Fallback reading directly via NCDatasets when DateTime decoding fails in GeoData
    end

    return NCDatasets.NCDataset(wind_file) do ds
        if !haskey(ds, "tau_x") || !haskey(ds, "tau_y")
            return (nothing, nothing, nothing)
        end
        tx_arr = Array(ds["tau_x"])
        ty_arr = Array(ds["tau_y"])
        nt = size(tx_arr, 3)
        tstep = 3600.0
        horizon = nt * tstep

        # Spatial average for each time slice
        tx_tuple = ntuple(k -> begin
            v = filter(!isnan, tx_arr[:, :, k])
            isempty(v) ? 0.0 : Float64(Statistics.mean(v))
        end, nt)
        ty_tuple = ntuple(k -> begin
            v = filter(!isnan, ty_arr[:, :, k])
            isempty(v) ? 0.0 : Float64(Statistics.mean(v))
        end, nt)

        stress_x = TimeSeriesSurfaceStress{nt}(tx_tuple, horizon, tstep)
        stress_y = TimeSeriesSurfaceStress{nt}(ty_tuple, horizon, tstep)
        heat_flux = nothing

        return (stress_x, stress_y, heat_flux)
    end
end

"""
    fetch_open_bathymetry(; kwargs...) -> GeoDataset

Fetch bathymetry from NOAA ERDDAP via GeoDataSources.
"""
function fetch_open_bathymetry(;
    lon_range::Tuple{Real, Real} = (-71.0, -53.0),
    lat_range::Tuple{Real, Real} = (40.0, 48.5),
    output_path::AbstractString = joinpath("inputs", "bathymetry.zarr"),
    dataset_id::AbstractString = "ETOPO_2022_v1_15s",
    stride::Int = 1,
    verbose::Bool = true
)
    return fetch_erddap_bathymetry(;
        lon_range = lon_range,
        lat_range = lat_range,
        output_path = output_path,
        dataset_id = dataset_id,
        stride = stride,
        verbose = verbose
    )
end

const fetch_etopo2022_bathymetry = fetch_open_bathymetry

"""
    fetch_open_surface_winds(; kwargs...) -> String

Fetch surface winds from Open-Meteo ERA5 via GeoDataSources.
"""
function fetch_open_surface_winds(;
    lon_range::Tuple{Real, Real} = (-71.0, -53.0),
    lat_range::Tuple{Real, Real} = (40.0, 48.5),
    time_iso::AbstractString = "2022-01-01T00:00:00Z",
    output_path::AbstractString = joinpath("inputs", "surface_winds.zarr"),
    verbose::Bool = true
)
    return fetch_open_meteo_winds(;
        lon_range = lon_range,
        lat_range = lat_range,
        time_iso = time_iso,
        output_path = output_path,
        verbose = verbose
    )
end

const fetch_open_meteo_surface_winds = fetch_open_surface_winds

"""
    fetch_natural_earth_coastline(; kwargs...) -> String

Fetch Natural Earth coastline vector via GeoDataSources.
"""
function fetch_natural_earth_coastline(;
    lon_range::Tuple{Real, Real} = (-71.0, -53.0),
    lat_range::Tuple{Real, Real} = (40.0, 48.5),
    resolution::AbstractString = "10m",
    output_path::AbstractString = joinpath("inputs", "coastline.dat"),
    margin_deg::Real = 1.0,
    verbose::Bool = true
)
    # Delegate to the GeoData abstraction layer, qualified. Unqualified, this call
    # resolves to THIS wrapper rather than to `GeoDataSources.fetch_natural_earth_coastline`
    # (both share the name), which is self-recursion and blows the stack.
    return GeoDataSources.fetch_natural_earth_coastline(;
        lon_range = lon_range,
        lat_range = lat_range,
        resolution = resolution,
        output_path = output_path,
        margin_deg = margin_deg,
        verbose = verbose
    )
end

"""
    fetch_open_woa_climatology(; kwargs...) -> NamedTuple

Fetch World Ocean Atlas 2023 climatology via GeoDataSources.
"""
function fetch_open_woa_climatology(;
    lon_range::Tuple{Real, Real} = (-71.0, -53.0),
    lat_range::Tuple{Real, Real} = (40.0, 48.5),
    month::Int = 0,
    output_dir::AbstractString = "inputs",
    include_o2::Bool = true,
    verbose::Bool = true
)
    return fetch_woa23(;
        lon_range = lon_range,
        lat_range = lat_range,
        month = month,
        output_dir = output_dir,
        include_o2 = include_o2,
        verbose = verbose
    )
end

"""
    fetch_boundary_hydrography(source::Symbol; kwargs...) -> NamedTuple

Acquire boundary hydrography via GeoDataSources.
"""
function fetch_boundary_hydrography(
    source::Symbol;
    lon_range::Tuple{Real, Real} = (-71.0, -53.0),
    lat_range::Tuple{Real, Real} = (40.0, 48.5),
    input_dir::AbstractString = "inputs",
    month::Int = 0,
    verbose::Bool = true
)
    return fetch_boundary_hydrography_geodata(
        source;
        lon_range = lon_range,
        lat_range = lat_range,
        input_dir = input_dir,
        month = month,
        verbose = verbose
    )
end

"""
    fetch_copernicus_physics_subset(; kwargs...) -> String

Download a regional subset of 3D temperature and salinity from Copernicus Marine (GLORYS).
"""
function fetch_copernicus_physics_subset(; kwargs...)
    return GeoDataSources.fetch_copernicus_physics_subset(; kwargs...)
end

"""
    fetch_copernicus_hydrography_with_fallback(; kwargs...) -> String

Fetch 3D temperature/salinity from Copernicus Marine with dataset fallback chain.
"""
function fetch_copernicus_hydrography_with_fallback(; kwargs...)
    return GeoDataSources.fetch_copernicus_hydrography_with_fallback(; kwargs...)
end

"""
    fetch_copernicus_surface_winds(; kwargs...) -> String

Fetch surface winds from Copernicus Climate Data Store (ERA5).
"""
function fetch_copernicus_surface_winds(; kwargs...)
    return GeoDataSources.fetch_copernicus_surface_winds(; kwargs...)
end
