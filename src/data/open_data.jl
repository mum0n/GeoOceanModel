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
        return GeoDataSources.fetch_open_meteo_winds(
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
            # Wind stress files carry dynamic stress in Pa (kg m^-1 s^-2). Oceananigans'
            # FluxBoundaryCondition expects kinematic stress (m^2 s^-2), so divide by
            # reference water density before returning.
            rho_water = 1025.0
            return (
                tau_x = tau_x / rho_water,
                tau_y = tau_y / rho_water,
                tau_max = tau_max / rho_water,
                speed10_rms = speed10_rms
            )
        end
    end
end

"""
    TimeSeriesBulkHeatFlux{N}

Bitstype surface net heat flux evaluator for atmospheric boundary forcing.

Evaluates bulk surface net heat flux Q(x, y, t, T_surf) in W m-2 from atmospheric state time series
(air temperature, surface pressure, relative humidity, cloud fraction, shortwave downwelling,
kinematic wind stress) using standard bulk aerodynamic and radiative formulas:
```math
Q_{\\text{net}} = Q_{\\text{sw}} - Q_{\\text{lw}} - Q_{\\text{sens}} - Q_{\\text{lat}}
```
All state time series are stored as NTuples of length N, ensuring the struct is strictly `isbits`
and directly executable on both CPU and GPU architectures.
"""
struct TimeSeriesBulkHeatFlux{N} <: Function
    ta::NTuple{N, Float64}
    ps::NTuple{N, Float64}
    rh::NTuple{N, Float64}
    cc::NTuple{N, Float64}
    sw::NTuple{N, Float64}
    tau_mag::NTuple{N, Float64}
    horizon::Float64
    tstep::Float64
    albedo::Float64
end

Adapt.adapt_structure(to, s::TimeSeriesBulkHeatFlux) = s

@inline function _wind_speed_from_stress_inline(tau::Float64)
    t = abs(tau)
    t <= 0.0 && return 0.0
    lo, hi = 0.0, 120.0
    # Inlined 30-iteration bisection for root-finding wind speed from kinematic stress
    for _ in 1:30
        mid = 0.5 * (lo + hi)
        cd = mid <= 11.0 ? 1.2e-3 : (0.49 + 0.065 * mid) * 1e-3
        s = (1.225 / 1025.0) * cd * (mid * mid)
        if s < t
            lo = mid
        else
            hi = mid
        end
    end
    return 0.5 * (lo + hi)
end

@inline function (s::TimeSeriesBulkHeatFlux{N})(x, y, t, T_surf) where {N}
    k = if s.horizon <= 0.0 || N <= 1
        1
    else
        k_calc = Int(floor(mod(Float64(t), s.horizon) / s.tstep)) + 1
        k_calc > N ? 1 : (k_calc < 1 ? 1 : k_calc)
    end

    Ta = @inbounds s.ta[k]
    p_s = @inbounds s.ps[k]
    r_h = (@inbounds s.rh[k]) / 100.0
    c_f = (@inbounds s.cc[k]) / 100.0
    c_f = c_f < 0.0 ? 0.0 : (c_f > 1.0 ? 1.0 : c_f)
    sw_d = @inbounds s.sw[k]
    tau = @inbounds s.tau_mag[k]

    sigma = 5.670374419e-8
    rho_a = 1.225
    cp_a = 1005.0
    Lv = 2.501e6
    Cd = 1.3e-3
    Ce = 1.5e-3

    # Vapor pressure and saturation formulas (August-Roche-Magnus approximation)
    # Ta arrives in Kelvin (ERA5). T_surf arrives in °C (model tracer). Convert
    # T_surf to Kelvin so all thermodynamic formulas are dimensionally consistent.
    T_surf_K = T_surf + 273.15
    esat_Ta = 611.2 * exp(17.67 * (Ta - 273.15) / (Ta - 29.65))
    e_a = r_h * esat_Ta
    esat_Ts = 611.2 * exp(17.67 * (T_surf_K - 273.15) / (T_surf_K - 29.65))
    qsat_Ts = 0.622 * esat_Ts / (p_s - 0.378 * esat_Ts)
    q_a = 0.622 * e_a / (p_s - 0.378 * e_a)

    U = _wind_speed_from_stress_inline(tau)
    sw_net = (1.0 - s.albedo) * (1.0 - 0.65 * c_f * c_f) * sw_d
    lw_factor = (0.34 - 0.14 * e_a / 1000.0) * (1.0 - 0.8 * c_f)
    lw_net = sigma * (T_surf_K^4 - Ta^4) * lw_factor
    sens = rho_a * cp_a * Cd * U * (T_surf_K - Ta)
    lat = rho_a * Lv * Ce * U * (qsat_Ts - q_a)

    return sw_net - lw_net - sens - lat
end

@inline (s::TimeSeriesBulkHeatFlux{N})(x, y, z, t, T_surf) where {N} = s(x, y, t, T_surf)

"""
    HaneyRelaxedHeatFlux{F}

Bitstype wrapper adding Haney sea-surface temperature relaxation to a base heat flux:
```math
Q_{\\text{applied}} = Q_{\\text{base}}(x, y, t, T_s) + \\lambda (T_{\\text{target}} - T_s)
```
"""
struct HaneyRelaxedHeatFlux{F} <: Function
    base_flux::F
    lambda::Float64
    t_target::Float64
end

Adapt.adapt_structure(to, h::HaneyRelaxedHeatFlux) =
    HaneyRelaxedHeatFlux(Adapt.adapt(to, h.base_flux), h.lambda, h.t_target)

@inline (h::HaneyRelaxedHeatFlux)(x, y, t, T_surf) =
    h.base_flux(x, y, t, T_surf) + h.lambda * (h.t_target - T_surf)

@inline (h::HaneyRelaxedHeatFlux)(x, y, z, t, T_surf) =
    h(x, y, t, T_surf)

"""
    KinematicSurfaceHeatFlux{F}

Bitstype wrapper converting a net heat flux Q (W m-2) to kinematic temperature flux (K m s-1):
```math
F_T = -\\frac{Q}{\\rho_0 c_p}
```
"""
struct KinematicSurfaceHeatFlux{F} <: Function
    flux::F
    rho0_cp::Float64
end

Adapt.adapt_structure(to, k::KinematicSurfaceHeatFlux) =
    KinematicSurfaceHeatFlux(Adapt.adapt(to, k.flux), k.rho0_cp)

@inline (k::KinematicSurfaceHeatFlux)(x, y, t, T) =
    -k.flux(x, y, t, T) / k.rho0_cp

@inline (k::KinematicSurfaceHeatFlux)(x, y, z, t, T) =
    k(x, y, t, T)

"""
    build_bulk_surface_flux(wind_file::AbstractString; albedo::Real = 0.06)

Read a surface-wind file and return `(stress_x, stress_y, heat_flux)` callables.
Returns `TimeSeriesSurfaceStress` and `TimeSeriesBulkHeatFlux` structs for GPU and
boundary condition compatibility.
"""
function build_bulk_surface_flux(wind_file::AbstractString; albedo::Real = 0.06)
    try
        ds = geoload(wind_file)
        needed = ("tau_x", "tau_y", "air_temperature", "surface_pressure",
                  "relative_humidity", "cloud_fraction", "sw_down")
        if all(v -> haskey(ds.variables, v), needed) && haskey(ds.coords, :time)
            tx = Float64.(ds.variables["tau_x"].data)
            ty = Float64.(ds.variables["tau_y"].data)
            ta = Float64.(ds.variables["air_temperature"].data)
            ps = Float64.(ds.variables["surface_pressure"].data)
            rh = Float64.(ds.variables["relative_humidity"].data)
            cc = Float64.(ds.variables["cloud_fraction"].data)
            sw = Float64.(ds.variables["sw_down"].data)
            tvec = first(GeoData.axis(ds, :time))

            nt = size(tx, 3)
            tstep = nt > 1 ? (tvec[end] - tvec[1]) / (nt - 1) : 3600.0
            horizon = nt > 1 ? tvec[end] - tvec[1] : 3600.0

            # Wind stress files carry dynamic stress in Pa (kg m^-1 s^-2).
            # Oceananigans FluxBoundaryCondition expects kinematic stress (m^2 s^-2),
            # so divide by reference water density before assigning.
            # Filter NaN/Inf values to prevent them from propagating into the model
            # and causing numerical divergence at later time steps.
            rho_water = 1025.0
            tx_tuple = ntuple(k -> begin
                v = Float64(tx[1, 1, k]) / rho_water
                isfinite(v) ? v : 0.0
            end, nt)
            ty_tuple = ntuple(k -> begin
                v = Float64(ty[1, 1, k]) / rho_water
                isfinite(v) ? v : 0.0
            end, nt)
            ta_tuple = ntuple(k -> Float64(ta[1, 1, k]), nt)
            ps_tuple = ntuple(k -> Float64(ps[1, 1, k]), nt)
            rh_tuple = ntuple(k -> Float64(rh[1, 1, k]), nt)
            cc_tuple = ntuple(k -> Float64(cc[1, 1, k]), nt)
            sw_tuple = ntuple(k -> Float64(sw[1, 1, k]), nt)
            tau_tuple = ntuple(k -> hypot(Float64(tx[1, 1, k]), Float64(ty[1, 1, k])), nt)

            stress_x = TimeSeriesSurfaceStress{nt}(tx_tuple, horizon, tstep)
            stress_y = TimeSeriesSurfaceStress{nt}(ty_tuple, horizon, tstep)
            heat_flux = TimeSeriesBulkHeatFlux{nt}(
                ta_tuple, ps_tuple, rh_tuple, cc_tuple, sw_tuple, tau_tuple,
                horizon, tstep, Float64(albedo)
            )

            return (stress_x, stress_y, heat_flux)
        end
    catch
        # Fallback reading directly via NCDatasets
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
        rho_water = 1025.0
        tx_tuple = ntuple(k -> begin
            v = filter(!isnan, tx_arr[:, :, k])
            val = isempty(v) ? 0.0 : Float64(Statistics.mean(v)) / rho_water
            isfinite(val) ? val : 0.0
        end, nt)
        ty_tuple = ntuple(k -> begin
            v = filter(!isnan, ty_arr[:, :, k])
            val = isempty(v) ? 0.0 : Float64(Statistics.mean(v)) / rho_water
            isfinite(val) ? val : 0.0
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
    return GeoDataSources.fetch_open_meteo_winds(;
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
    return GeoDataSources.fetch_woa23(;
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

With `with_velocities = true` the boundary zonal and meridional velocity are fetched
and returned as well, for a momentum sponge that relaxes toward an observed current
rather than toward a hand-entered constant.
"""
function fetch_boundary_hydrography(
    source::Symbol;
    lon_range::Tuple{Real, Real} = (-71.0, -53.0),
    lat_range::Tuple{Real, Real} = (40.0, 48.5),
    input_dir::AbstractString = "inputs",
    month::Int = 0,
    verbose::Bool = true,
    with_velocities::Bool = false
)
    return fetch_boundary_hydrography_geodata(
        source;
        lon_range = lon_range,
        lat_range = lat_range,
        input_dir = input_dir,
        month = month,
        verbose = verbose,
        with_velocities = with_velocities
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
