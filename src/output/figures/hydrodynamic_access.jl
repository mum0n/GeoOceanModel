"""
    hydrodynamic_access.jl

Hydrodynamic field access.

Reading fields out of the hydrodynamic products for the figures: depth and time
index resolution, bathymetry resolution, dataset extraction, diagnostics. Kept
apart from the figures that draw them, so a figure cannot depend silently on how
a field was resolved.
"""

using CairoMakie
import CairoMakie: record

"""
    resolve_depth_index(
        depths::AbstractVector{<:Real},
        depth::Union{Nothing, Real},
        depth_level::Union{Nothing, Int}
    ) -> Int

Resolve discrete vertical grid level index from either continuous depth in meters
(positive or negative) or discrete index. Defaults to surface (index 1).
"""
function resolve_depth_index(
    depths::AbstractVector{<:Real},
    depth::Union{Nothing, Real},
    depth_level::Union{Nothing, Int}
)::Int
    nz = length(depths)
    nz == 0 && return 1
    if !isnothing(depth)
        target_z = depth > 0 ? -Float64(depth) : Float64(depth)
        return argmin([abs(Float64(d) - target_z) for d in depths])
    elseif !isnothing(depth_level)
        return clamp(depth_level, 1, nz)
    else
        return 1
    end
end

"""
    resolve_time_index(
        times::AbstractVector{<:Real},
        time_seconds::Union{Nothing, Real},
        time_index::Union{Nothing, Int}
    ) -> Int

Resolve discrete temporal snapshot index from either continuous timestamp in seconds
or discrete index. Defaults to first snapshot (index 1).
"""
function resolve_time_index(
    times::AbstractVector{<:Real},
    time_seconds::Union{Nothing, Real},
    time_index::Union{Nothing, Int}
)::Int
    nt = length(times)
    nt == 0 && return 1
    if !isnothing(time_seconds)
        return argmin([abs(Float64(t) - Float64(time_seconds)) for t in times])
    elseif !isnothing(time_index)
        return clamp(time_index, 1, nt)
    else
        return 1
    end
end

"""
    compute_hydrodynamic_diagnostics(
        lons::AbstractVector{<:Real},
        lats::AbstractVector{<:Real},
        depths::AbstractVector{<:Real},
        u::AbstractArray{<:Real, 3},
        v::AbstractArray{<:Real, 3},
        w::AbstractArray{<:Real, 3},
        temp::AbstractArray{<:Real, 3},
        sal::Union{Nothing, AbstractArray{<:Real, 3}};
        ν_closure::Real = 1e-2,
        κ_closure::Real = 1e-2
    ) -> NamedTuple

Compute derived physical oceanographic diagnostics:
- Potential Density \$\\\\\\\\\\\\\\rho(S, T)\$ (Boussinesq linear approximation).
- Brunt-Väisälä buoyancy frequency squared \$N^2 = -(g/\\\\\\\\\\\\\\rho_0) \\\\\\\\\\\\\\partial \\\\\\\\\\\\\\rho / \\\\\\\\\\\\\\partial z\$.
- Vertical salinity stratification gradient \$\\\\\\\\\\\\\\partial S / \\\\\\\\\\\\\\partial z\$.
- Turbulent vertical eddy diffusivity \$\\\\\\\\\\\\\\kappa_v\$ and eddy viscosity \$\\\\\\\\\\\\\\nu_v\$.
- Relative vertical vorticity \$\\\\\\\\\\\\\\zeta = \\\\\\\\\\\\\\partial v / \\\\\\\\\\\\\\partial x - \\\\\\\\\\\\\\partial u / \\\\\\\\\\\\\\partial y\$.

Salinity (`sal`) is optional. The GeoData store carries u/v/w/T, which is what the output
callback writes, so a store-backed call passes `nothing`; the density then degrades to its
thermal term against the reference 35, and ∂S/∂z is 0.0. Neither is filled from a constant a
run never produced, because such a value would read as observed.

# Mixing provenance
When `native_κ` / `native_ν` are supplied they are the model-published mixing fields and are
returned verbatim — with a `ScalarDiffusivity` closure these are the exact constants the
hydrodynamics advected with. Only when both are absent is a shear-stratification gradient
Richardson number `Ri = N^2 / [(∂u/∂z)^2 + (∂v/∂z)^2]` used to synthesize a diffusivity/viscosity
*proxy*. That proxy is not a model output and will generally disagree with the integrated
physics; it exists only so legacy records lacking native mixing fields remain plottable.
"""
function compute_hydrodynamic_diagnostics(
    lons::AbstractVector{<:Real},
    lats::AbstractVector{<:Real},
    depths::AbstractVector{<:Real},
    u::AbstractArray{<:Real, 3},
    v::AbstractArray{<:Real, 3},
    w::AbstractArray{<:Real, 3},
    temp::AbstractArray{<:Real, 3},
    sal::Union{Nothing, AbstractArray{<:Real, 3}};
    ν_closure::Real = 1e-2,
    κ_closure::Real = 1e-2,
    native_κ::Union{Nothing, AbstractArray{<:Real, 3}} = nothing,
    native_ν::Union{Nothing, AbstractArray{<:Real, 3}} = nothing
)::NamedTuple
    nx = length(lons)
    ny = length(lats)
    nz = length(depths)

    # 1. Seawater Density (Boussinesq linear equation of state)
    rho0 = 1025.0
    alpha = 1.7e-4
    beta = 7.6e-4
    T0 = 10.0
    S0 = 35.0

    rho = zeros(Float64, nx, ny, nz)
    for k in 1:nz, j in 1:ny, i in 1:nx
        t_val = Float64(temp[i, j, k])
        # Salinity is optional: the GeoData store carries u/v/w/T, because that is what the
        # output callback writes. A missing salinity degrades the density to the thermal
        # term against the reference 35 rather than being filled with a constant the run
        # never produced -- a constant would read as observed and shift rho.
        s_val = isnothing(sal) ? S0 : Float64(sal[i, j, k])
        if !isnan(t_val) && !isnan(s_val)
            rho[i, j, k] = rho0 * (1.0 - alpha * (t_val - T0) + beta * (s_val - S0))
        else
            rho[i, j, k] = NaN
        end
    end

    # 2. Stratification: N² and ∂S/∂z
    N2 = zeros(Float64, nx, ny, nz)
    dS_dz = zeros(Float64, nx, ny, nz)
    shear_sq = zeros(Float64, nx, ny, nz)

    for k in 1:nz
        k_prev = max(1, k - 1)
        k_next = min(nz, k + 1)
        dz = Float64(depths[k_prev] - depths[k_next])
        if abs(dz) < 1e-6
            dz = 1.0
        end

        for j in 1:ny, i in 1:nx
            if !isnan(rho[i, j, k_prev]) && !isnan(rho[i, j, k_next])
                # Depths are negative, depths[k_prev] > depths[k_next], so dz > 0
                drho = rho[i, j, k_prev] - rho[i, j, k_next]
                N2[i, j, k] = max(0.0, -(9.81 / rho0) * (drho / dz))

                ds = isnothing(sal) ? 0.0 : Float64(sal[i, j, k_prev] - sal[i, j, k_next])
                dS_dz[i, j, k] = ds / dz

                du = Float64(u[i, j, k_prev] - u[i, j, k_next]) / dz
                dv = Float64(v[i, j, k_prev] - v[i, j, k_next]) / dz
                shear_sq[i, j, k] = du^2 + dv^2
            else
                N2[i, j, k] = NaN
                dS_dz[i, j, k] = NaN
                shear_sq[i, j, k] = NaN
            end
        end
    end

    # 3. Turbulent eddy diffusivity & viscosity.
    #
    # When the model published its own mixing fields (`native_κ` / `native_ν`) those are
    # authoritative and are used verbatim: with a `ScalarDiffusivity` closure they are the
    # exact constants the hydrodynamics advected with.
    #
    # Only when they are absent do we fall back to a Richardson-number parameterization.
    # That fallback is a *proxy*, not a model output, and is kept solely so legacy records
    # without native mixing remain plottable.
    diff = zeros(Float64, nx, ny, nz)
    visc = zeros(Float64, nx, ny, nz)
    ri_arr = zeros(Float64, nx, ny, nz)
    using_native_mixing = !isnothing(native_κ) || !isnothing(native_ν)

    for k in 1:nz
        z_m = Float64(depths[k])
        surf_mix = 0.015 * exp(z_m / 25.0)

        for j in 1:ny, i in 1:nx
            n2_val = N2[i, j, k]
            s2_val = shear_sq[i, j, k]
            if !isnan(n2_val) && !isnan(s2_val)
                ri = max(0.0, n2_val / max(s2_val, 1e-7))
                ri_arr[i, j, k] = ri
            else
                ri_arr[i, j, k] = NaN
            end

            if using_native_mixing
                k_native = isnothing(native_κ) ? nothing : native_κ[i, j, k]
                v_native = isnothing(native_ν) ? nothing : native_ν[i, j, k]
                diff[i, j, k] = isnothing(k_native) || isnan(k_native) ? NaN : Float64(k_native)
                visc[i, j, k] = isnothing(v_native) || isnan(v_native) ? NaN : Float64(v_native)
            else
                if !isnan(n2_val) && !isnan(s2_val)
                    k_eddy = 1e-5 + 1e-2 / ((1.0 + 5.0 * ri)^2) + surf_mix
                    nu_eddy = 1e-4 + 1e-2 / ((1.0 + 5.0 * ri)^3) + surf_mix
                    diff[i, j, k] = clamp(k_eddy, 1e-6, 0.05)
                    visc[i, j, k] = clamp(nu_eddy, 1e-5, 0.05)
                else
                    diff[i, j, k] = NaN
                    visc[i, j, k] = NaN
                end
            end
        end
    end

    # 4. Relative Vorticity ζ at surface level
    vort = zeros(Float64, nx, ny)
    r_earth = 6.371e6
    for j in 2:(ny - 1), i in 2:(nx - 1)
        dx_m = r_earth * cosd(Float64(lats[j])) * deg2rad(Float64(lons[i + 1] - lons[i - 1]))
        dy_m = r_earth * deg2rad(Float64(lats[j + 1] - lats[j - 1]))
        if !isnan(v[i + 1, j, 1]) && !isnan(v[i - 1, j, 1]) &&
           !isnan(u[i, j + 1, 1]) && !isnan(u[i, j - 1, 1]) &&
           dx_m > 0.0 && dy_m > 0.0
            dv_dx = (Float64(v[i + 1, j, 1]) - Float64(v[i - 1, j, 1])) / dx_m
            du_dy = (Float64(u[i, j + 1, 1]) - Float64(u[i, j - 1, 1])) / dy_m
            vort[i, j] = dv_dx - du_dy
        else
            vort[i, j] = NaN
        end
    end

    return (
        density = rho,
        stratification = N2,
        salinity_stratification = dS_dz,
        diffusion = diff,
        viscosity = visc,
        richardson = ri_arr,
        vorticity = vort
    )
end

"""
    read_immersed_bathymetry(location::AbstractString) -> Union{Matrix{Float64}, Nothing}

Read the authoritative seabed (metres, negative in the ocean) for a realized run.

`location` is the run's output *directory*, which holds the GeoData store -- the ocean
half's deliverable, and the path `run_segment_simulation` hands back. The seabed is read
from the store's own `.zattrs`, where `write_grid_metadata!` puts it because it is grid
metadata that does not vary with time.

The immersed grid's `bottom_height` is the *actual* seabed the simulation ran on, so it is
the correct reference for masking below-seafloor cells in cross-sections and profiles.
Falling back to a synthetic flat seabed would silently mask (or fabricate) water column.

A simulation JLD2 *file* is still accepted, for the archives written before the store
existed: it is read from its `serialized/grid` group. The store is preferred, because it
is what a run now hands back.

# Inputs
- `location`: Path to the run's output directory (holding `hydrodynamics.zarr`), or to a
  simulation JLD2 file for older archives.

# Outputs
- `Matrix{Float64}`: Bottom height on the horizontal grid, indexed `[i_lon, j_lat]`;
  or `nothing` when neither the store nor the file carries an immersed grid.
"""
function read_immersed_bathymetry(location::AbstractString)
    # Old archives: a JLD2 file, read from its serialized grid.
    isfile(location) && endswith(location, ".jld2") && return _read_immersed_bathymetry_jld2(location)

    # The store: `hydrodynamics.zarr` inside the run's output directory.
    store = output_zarr_path(location)
    zone = joinpath(store, ".zattrs")
    isfile(zone) || return nothing

    try
        attrs = JSON3.read(read(zone, String))
        haskey(attrs, "bottom_height") || return nothing
        raw = attrs["bottom_height"]
        n_lon = length(raw)
        n_lat = isempty(raw) ? 0 : length(raw[1])
        mat = Matrix{Float64}(undef, n_lon, n_lat)
        for i in 1:n_lon, j in 1:n_lat
            mat[i, j] = Float64(raw[i][j])
        end
        return mat
    catch
        return nothing
    end
end

"""
    _read_immersed_bathymetry_jld2(jld2_path::AbstractString) -> Union{Matrix{Float64}, Nothing}

Read the immersed seabed from the `serialized/grid` group of a simulation JLD2 file, as
written by the Oceananigans `JLD2Writer`.

Kept separate from `read_immersed_bathymetry` so the archive-only path is called out by
name; it exists for runs that predate the store, not as a second way to read a current one.
"""
function _read_immersed_bathymetry_jld2(jld2_path::AbstractString)
    isfile(jld2_path) || return nothing

    bottom = try
        JLD2.jldopen(jld2_path, "r") do file
            if !haskey(file, "serialized") || !haskey(file["serialized"], "grid")
                return nothing
            end
            grid = file["serialized"]["grid"]
            ib = hasproperty(grid, :immersed_boundary) ? grid.immersed_boundary : nothing
            if isnothing(ib) || !hasproperty(ib, :bottom_height)
                return nothing
            end
            bh = ib.bottom_height
            # bottom_height is stored on the underlying grid with halos; strip them.
            base = hasproperty(grid, :underlying_grid) ? grid.underlying_grid : grid
            Hx = hasproperty(base, :Hx) ? base.Hx : 0
            Hy = hasproperty(base, :Hy) ? base.Hy : 0
            Nx = hasproperty(base, :Nx) ? base.Nx : size(bh, 1) - 2 * Hx
            Ny = hasproperty(base, :Ny) ? base.Ny : size(bh, 2) - 2 * Hy
            Array{Float64}(parent(bh.data)[Hx+1 : Hx+Nx, Hy+1 : Hy+Ny, 1])
        end
    catch
        nothing
    end

    return bottom
end

"""
    resolve_bathymetry_for_plot(lons, lats; sources = (), fallback = nothing)

Pick the first bathymetry source that actually describes a seafored, and normalise it.

A field is accepted only if it is a matrix of the expected size containing at least one
finite negative (submerged) value. A constant sentinel such as `fill(-150.0, ...)` carries
no bathymetric information and is rejected, so callers never mask a real water column
against invented topography.

# Inputs
- `lons`, `lats`: Horizontal coordinate vectors defining the expected matrix size.
- `sources`: Candidate bathymetry objects, tried in order.
- `fallback`: Value returned when no source is usable (default `nothing`).

# Outputs
- `Matrix{Float64}` or `nothing`.
"""
function resolve_bathymetry_for_plot(lons, lats; sources = (), fallback = nothing)
    expected = (length(lons), length(lats))
    for src in sources
        isnothing(src) && continue
        m = src
        if m isa AbstractString
            m = read_immersed_bathymetry(m)
        end
        isnothing(m) && continue
        mat = try
            Float64.(m)
        catch
            continue
        end
        size(mat) == expected || continue
        all(isfinite, mat) || continue
        # Must contain genuine submerged topography to be usable for masking.
        minimum(mat) < 0.0 || continue
        length(unique(mat)) > 1 || continue
        return mat
    end
    return fallback
end

"""
    resolve_land_mask_coastline(
        coastline;
        coastline_path::Union{Nothing, AbstractString} = nothing
    ) -> Vector{NamedTuple}

Land rings the mask is drawn against, in the order the caller can see: the rings
passed explicitly, else the run's own fetched coastline, else the region's
bundled rings.

`coastline_path` names the coastline GeoParquet file, or the directory holding
it as `coastline.parquet`. No file and no path means the region's rings, which
is the only case where a figure is masked against land the run did not fetch.
"""
function resolve_land_mask_coastline(
    coastline::Union{Nothing, AbstractVector{<:NamedTuple}};
    coastline_path::Union{Nothing, AbstractString} = nothing
)
    isnothing(coastline) || return coastline
    if !isnothing(coastline_path)
        candidates = AbstractString[
            isdir(coastline_path) ? joinpath(coastline_path, "coastline.parquet") :
                String(coastline_path)
        ]
        for candidate in candidates
            isfile(candidate) && return load_coastline_polygons(candidate)
        end
    end
    return GeoDataSources.SCOTIAN_SHELF_COASTLINE
end

"""
    extract_hydrodynamic_dataset(
        hydro_input::Any;
        depth::Union{Nothing, Real} = nothing,
        depth_level::Union{Nothing, Int} = nothing,
        time_seconds::Union{Nothing, Real} = nothing,
        time_index::Union{Nothing, Int} = nothing,
        domain_lon::Tuple{<:Real, <:Real} = (-71.0, -53.0),
        domain_lat::Tuple{<:Real, <:Real} = (40.0, 48.5),
        grid_size::Tuple{Int, Int} = (35, 30),
        target_depths::AbstractVector{<:Real} = [-2.5, -25.0, -50.0, -100.0],
        run_id::Union{Nothing, AbstractString} = nothing,
        coastline::Union{Nothing, AbstractVector{<:NamedTuple}} = nothing,
        coastline_path::Union{Nothing, AbstractString} = nothing
    ) -> NamedTuple

Extract, normalize, and format 2D and 3D hydrodynamic model fields (advection currents,
temperature, salinity, density, stratification, turbulent diffusion, viscosity,
free surface elevation, and bathymetry) at specific depths and times.

This is the central data extraction utility that handles multiple input formats:
- Oceananigans model instances (live or serialized)
- JLD2 simulation output files (timeseries groups with snapshot keys)
- Zarr analytical storage connections (with optional run_id filter)
- NamedTuple or Dict with pre-extracted fields
- `nothing` (generates synthetic test data)

# Mathematical Formulations
- **Horizontal Advection Current Velocity**:
  ```math
  \\boldsymbol{u}_h(x, y, z, t) = (u(x, y, z, t), v(x, y, z, t)), \\quad
  |\\boldsymbol{u}_h| = \\sqrt{u^2 + v^2}
  ```
- **Seawater Temperature & Practical Salinity**:
  \$T(x, y, z, t)\$ in °C, \$S(x, y, z, t)\$ in PSU.
- **Salinity & Density Stratification**:
  ```math
  N^2(x, y, z, t) = -\\frac{g}{\\rho_0} \\frac{\\partial \\rho}{\\partial z}, \\quad
  \\frac{\\partial S}{\\partial z}(x, y, z, t)
  ```
- **Turbulent Eddy Diffusivity & Viscosity**:
  \\\$\\\\kappa_v(x, y, z, t), \\\\nu_v(x, y, z, t)\\\$ in \\\$m^2 s^{-1}\\\$ parameterized via Richardson number.
- **Gradient Richardson Number**:
  ```math
  Ri = \\frac{N^2}{(\\partial u/\\partial z)^2 + (\\partial v/\\partial z)^2}
  ```
- **Relative Vorticity**:
  ```math
  \\zeta = \\frac{\\partial v}{\\partial x} - \\frac{\\partial u}{\\partial y}
  ```

# Inputs
- `hydro_input`: Oceananigans model, JLD2 file path, Zarr storage connection, NamedTuple, Dict, or `nothing`.
- `depth`: Optional continuous depth in meters (e.g. `-25.0` or `25.0`). Uses nearest depth level.
- `depth_level`: Optional vertical level index (1-indexed, overrides `depth`).
- `time_seconds`: Optional simulation time in seconds (selects nearest snapshot).
- `time_index`: Optional discrete time snapshot index (1-based, overrides `time_seconds`).
- `domain_lon`: Longitudinal bounds `(min_lon, max_lon)` for synthetic fallback (default: embedding domain).
- `domain_lat`: Latitudinal bounds `(min_lat, max_lat)` for synthetic fallback (default: embedding domain).
- `grid_size`: Horizontal grid dimension `(nx, ny)` for synthetic fallback.
- `target_depths`: Depth coordinates in meters (default `[-2.5, -25.0, -50.0, -100.0]`).
- `run_id`: Optional Zarr simulation run identifier for multi-run queries.

# Outputs
- `NamedTuple` containing:
  - `lons`, `lats`, `depths`: Grid coordinate vectors (longitude °E, latitude °N, depth m).
  - `times`, `time_seconds`: Available timestamps and resolved snapshot time.
  - `depth_index`, `depth_m`: Resolved vertical level index and depth in meters.
  - `u`, `v`, `w`, `speed`: Velocity fields at target depth (m/s, m/s, m/s, m/s).
  - `temperature`, `salinity`: Hydrographic tracers at target depth (°C, PSU).
  - `density`: Potential density at target depth (kg m⁻³).
  - `stratification`: Buoyancy frequency squared \$N^2\$ at target depth (s⁻²).
  - `salinity_stratification`: Vertical salinity gradient at target depth (PSU m⁻¹).
  - `diffusion`, `viscosity`: Turbulent diffusivities at target depth (m²/s).
  - `richardson_number`: Gradient Richardson number at target depth (dimensionless).
  - `vorticity`: Relative vorticity at target depth (s⁻¹).
  - `elevation`, `bathymetry`: Surface elevation and seafloor depth (m).

# Notes
- For JLD2 files: extracts the full 3D fields at the resolved time snapshot, then interpolates
  horizontally to a regular `(nx, ny)` grid and vertically to `target_depths`.
- For `nothing` input: generates synthetic test fields matching the specified domain/grid.
- Bathymetry is extracted from JLD2 `grid` group or computed from model grid.
"""
function extract_hydrodynamic_dataset(
    hydro_input::Any;
    depth::Union{Nothing, Real} = nothing,
    depth_level::Union{Nothing, Int} = nothing,
    time_seconds::Union{Nothing, Real} = nothing,
    time_index::Union{Nothing, Int} = nothing,
    domain_lon::Tuple{<:Real, <:Real} = (-71.0, -53.0),
    domain_lat::Tuple{<:Real, <:Real} = (40.0, 48.5),
    grid_size::Tuple{Int, Int} = (35, 30),
    target_depths::AbstractVector{<:Real} = [-2.5, -25.0, -50.0, -100.0],
    run_id::Union{Nothing, AbstractString} = nothing,
    coastline::Union{Nothing, AbstractVector{<:NamedTuple}} = nothing,
    coastline_path::Union{Nothing, AbstractString} = nothing
)
    nx, ny = grid_size
    t_depths = collect(Float64, target_depths)
    nz = length(t_depths)

    # 1. JLD2 Simulation Output File
    if hydro_input isa AbstractString && isfile(hydro_input) && endswith(hydro_input, ".jld2")
        # Everything the do-block assigns must be declared `local` here: a do-block is its own
        # scope, and an undeclared name stays inside the closure. `native_κ_j` / `native_ν_j` were
        # missing from this list, so `UndefVarError` was raised for every JLD2 archive at the
        # `compute_hydrodynamic_diagnostics` call below.
        local lons_j, lats_j, deps_j, times_j, u_j, v_j, w_j, T_j, S_j, elev_j, bathymetry_j
        local native_κ_j, native_ν_j
        jldopen(hydro_input, "r") do file
            u_group = file["timeseries/u"]
            num_keys = [k for k in keys(u_group) if tryparse(Float64, k) !== nothing]
            if isempty(num_keys)
                error("No numerical snapshot keys found in timeseries/u of $(hydro_input)")
            end
            sorted_keys = sort(num_keys, by = k -> parse(Float64, k))
            times_j = if haskey(file, "timeseries/t")
                t_node = file["timeseries/t"]
                if t_node isa JLD2.Group
                    [haskey(t_node, k) ? Float64(t_node[k]) : parse(Float64, k) for k in sorted_keys]
                else
                    collect(Float64, t_node)
                end
            else
                [parse(Float64, k) for k in sorted_keys]
            end
            if isempty(times_j)
                times_j = [0.0]
            end

            t_idx = resolve_time_index(times_j, time_seconds, time_index)
            key_sel = sorted_keys[t_idx]

            grid_obj = if haskey(file, "grid")
                file["grid"]
            elseif haskey(file, "serialized/grid")
                file["serialized/grid"]
            else
                nothing
            end

            ug = if !isnothing(grid_obj)
                hasproperty(grid_obj, :underlying_grid) ? grid_obj.underlying_grid :
                    (hasproperty(grid_obj, :grid) ? grid_obj.grid : grid_obj)
            else
                nothing
            end

            sample_u = file["timeseries/u/$(key_sel)"]
            has_T = haskey(file, "timeseries/T")
            sample_T = has_T ? file["timeseries/T/$(key_sel)"] : sample_u

            Nx = if !isnothing(ug) && hasproperty(ug, :Nx)
                Int(ug.Nx)
            else
                has_T ? size(sample_T, 1) : size(sample_u, 1)
            end
            Ny = if !isnothing(ug) && hasproperty(ug, :Ny)
                Int(ug.Ny)
            else
                has_T ? size(sample_T, 2) : size(sample_u, 2)
            end
            Nz = if !isnothing(ug) && hasproperty(ug, :Nz)
                Int(ug.Nz)
            else
                has_T ? size(sample_T, 3) : size(sample_u, 3)
            end

            Hx = (!isnothing(ug) && hasproperty(ug, :Hx)) ? Int(ug.Hx) : 0
            Hy = (!isnothing(ug) && hasproperty(ug, :Hy)) ? Int(ug.Hy) : 0
            Hz = (!isnothing(ug) && hasproperty(ug, :Hz)) ? Int(ug.Hz) : 0

            # Interior index ranges for cell centers
            i_c = (1 + Hx):(Nx + Hx)
            j_c = (1 + Hy):(Ny + Hy)
            k_c = (1 + Hz):(Nz + Hz)

            # Spatial coordinate vectors
            lon_raw = if !isnothing(ug) && hasproperty(ug, :λᶜᵃᵃ)
                hasproperty(ug.λᶜᵃᵃ, :parent) ? collect(Float64, ug.λᶜᵃᵃ.parent) :
                    collect(Float64, ug.λᶜᵃᵃ)
            elseif !isnothing(ug) && hasproperty(ug, :xᶜᵃᵃ)
                hasproperty(ug.xᶜᵃᵃ, :parent) ? collect(Float64, ug.xᶜᵃᵃ.parent) :
                    collect(Float64, ug.xᶜᵃᵃ)
            else
                collect(range(domain_lon[1], domain_lon[2], length = Nx))
            end
            lons_j = if length(lon_raw) >= (Nx + 2 * Hx)
                lon_raw[i_c]
            elseif length(lon_raw) >= Nx
                lon_raw[1:Nx]
            else
                collect(range(domain_lon[1], domain_lon[2], length = Nx))
            end

            lat_raw = if !isnothing(ug) && hasproperty(ug, :φᵃᶜᵃ)
                hasproperty(ug.φᵃᶜᵃ, :parent) ? collect(Float64, ug.φᵃᶜᵃ.parent) :
                    collect(Float64, ug.φᵃᶜᵃ)
            elseif !isnothing(ug) && hasproperty(ug, :yᵃᶜᵃ)
                hasproperty(ug.yᵃᶜᵃ, :parent) ? collect(Float64, ug.yᵃᶜᵃ.parent) :
                    collect(Float64, ug.yᵃᶜᵃ)
            else
                collect(range(domain_lat[1], domain_lat[2], length = Ny))
            end
            lats_j = if length(lat_raw) >= (Ny + 2 * Hy)
                lat_raw[j_c]
            elseif length(lat_raw) >= Ny
                lat_raw[1:Ny]
            else
                collect(range(domain_lat[1], domain_lat[2], length = Ny))
            end

            dep_raw = if !isnothing(ug) && hasproperty(ug, :z) && hasproperty(ug.z, :cᵃᵃᶜ)
                zc = ug.z.cᵃᵃᶜ
                hasproperty(zc, :parent) ? collect(Float64, zc.parent) :
                    collect(Float64, zc)
            elseif !isnothing(ug) && hasproperty(ug, :zᵃᵃᶜ)
                zc = ug.zᵃᵃᶜ
                hasproperty(zc, :parent) ? collect(Float64, zc.parent) :
                    collect(Float64, zc)
            else
                collect(range(-600.0, 0.0, length = Nz))
            end
            deps_j = if length(dep_raw) >= (Nz + 2 * Hz)
                dep_raw[k_c]
            elseif length(dep_raw) >= Nz
                dep_raw[1:Nz]
            else
                collect(range(-600.0, 0.0, length = Nz))
            end

            # Extract tracers and de-halo if necessary
            raw_T = has_T ? Float64.(file["timeseries/T/$(key_sel)"]) : fill(4.5, Nx, Ny, Nz)
            T_j = if size(raw_T) == (Nx, Ny, Nz)
                raw_T
            elseif size(raw_T) >= (Nx + 2 * Hx, Ny + 2 * Hy, Nz + 2 * Hz)
                raw_T[i_c, j_c, k_c]
            else
                raw_T[1:Nx, 1:Ny, 1:Nz]
            end

            has_S = haskey(file, "timeseries/S")
            raw_S = has_S ? Float64.(file["timeseries/S/$(key_sel)"]) : fill(33.0, Nx, Ny, Nz)
            S_j = if size(raw_S) == (Nx, Ny, Nz)
                raw_S
            elseif size(raw_S) >= (Nx + 2 * Hx, Ny + 2 * Hy, Nz + 2 * Hz)
                raw_S[i_c, j_c, k_c]
            else
                raw_S[1:Nx, 1:Ny, 1:Nz]
            end

            # De-stagger velocities to cell centers
            raw_u = Float64.(file["timeseries/u/$(key_sel)"])
            u_j = if size(raw_u) == (Nx, Ny, Nz)
                raw_u
            elseif size(raw_u) >= (Nx + 1 + 2 * Hx, Ny + 2 * Hy, Nz + 2 * Hz)
                u_face = raw_u[(1 + Hx):(Nx + 1 + Hx), j_c, k_c]
                0.5 .* (u_face[1:Nx, :, :] .+ u_face[2:Nx + 1, :, :])
            elseif size(raw_u, 1) == Nx + 1
                0.5 .* (raw_u[1:Nx, 1:Ny, 1:Nz] .+ raw_u[2:Nx + 1, 1:Ny, 1:Nz])
            else
                raw_u[1:Nx, 1:Ny, 1:Nz]
            end

            raw_v = Float64.(file["timeseries/v/$(key_sel)"])
            v_j = if size(raw_v) == (Nx, Ny, Nz)
                raw_v
            elseif size(raw_v) >= (Nx + 2 * Hx, Ny + 1 + 2 * Hy, Nz + 2 * Hz)
                v_face = raw_v[i_c, (1 + Hy):(Ny + 1 + Hy), k_c]
                0.5 .* (v_face[:, 1:Ny, :] .+ v_face[:, 2:Ny + 1, :])
            elseif size(raw_v, 2) == Ny + 1
                0.5 .* (raw_v[1:Nx, 1:Ny, 1:Nz] .+ raw_v[1:Nx, 2:Ny + 1, 1:Nz])
            else
                raw_v[1:Nx, 1:Ny, 1:Nz]
            end

            has_w = haskey(file, "timeseries/w")
            w_j = if !has_w
                zeros(Nx, Ny, Nz)
            else
                raw_w = Float64.(file["timeseries/w/$(key_sel)"])
                if size(raw_w) == (Nx, Ny, Nz)
                    raw_w
                elseif size(raw_w) >= (Nx + 2 * Hx, Ny + 2 * Hy, Nz + 1 + 2 * Hz)
                    w_face = raw_w[i_c, j_c, (1 + Hz):(Nz + 1 + Hz)]
                    0.5 .* (w_face[:, :, 1:Nz] .+ w_face[:, :, 2:Nz + 1])
                elseif size(raw_w, 3) == Nz + 1
                    0.5 .* (raw_w[1:Nx, 1:Ny, 1:Nz] .+ raw_w[1:Nx, 1:Ny, 2:Nz + 1])
                else
                    raw_w[1:Nx, 1:Ny, 1:Nz]
                end
            end

            has_eta = haskey(file, "timeseries/η")
            elev_j = if !has_eta
                zeros(Nx, Ny)
            else
                raw_eta = Float64.(file["timeseries/η/$(key_sel)"])
                if size(raw_eta) == (Nx, Ny)
                    raw_eta
                elseif size(raw_eta) >= (Nx + 2 * Hx, Ny + 2 * Hy)
                    raw_eta[i_c, j_c]
                else
                    raw_eta[1:Nx, 1:Ny]
                end
            end

            # Model-native mixing fields, written by the simulation's JLD2Writer. When
            # present these are the exact κ/ν the hydrodynamics advected with; absence
            # (legacy records) falls back to the Richardson proxy downstream.
            read_native_3d(sym) = begin
                haskey(file, "timeseries/$(sym)") || return nothing
                raw = Float64.(file["timeseries/$(sym)/$(key_sel)"])
                if size(raw) == (Nx, Ny, Nz)
                    raw
                elseif size(raw, 1) >= (Nx + 2 * Hx) && size(raw, 2) >= (Ny + 2 * Hy)
                    raw[i_c, j_c, (1 + Hz):(Nz + Hz)]
                else
                    raw[1:Nx, 1:Ny, 1:Nz]
                end
            end
            native_κ_j = read_native_3d(:κ)
            native_ν_j = read_native_3d(:ν)

            # Extract real bathymetry from immersed boundary grid if available
            bathymetry_j = if !isnothing(grid_obj) &&
                              hasproperty(grid_obj, :immersed_boundary) &&
                              hasproperty(grid_obj.immersed_boundary, :bottom_height)
                bh = grid_obj.immersed_boundary.bottom_height
                raw_bh = hasproperty(bh, :parent) ? collect(Float64, bh.parent) :
                         collect(Float64, bh)
                if size(raw_bh, 1) >= (Nx + 2 * Hx) && size(raw_bh, 2) >= (Ny + 2 * Hy)
                    raw_bh[i_c, j_c, 1]
                elseif size(raw_bh, 1) >= Nx && size(raw_bh, 2) >= Ny
                    ndims(raw_bh) == 3 ? raw_bh[1:Nx, 1:Ny, 1] : raw_bh[1:Nx, 1:Ny]
                else
                    nothing
                end
            else
                nothing
            end
        end

        diag = compute_hydrodynamic_diagnostics(
            lons_j, lats_j, deps_j, u_j, v_j, w_j, T_j, S_j;
            native_κ = native_κ_j, native_ν = native_ν_j
        )
        k_sel = resolve_depth_index(deps_j, depth, depth_level)
        sel_time = isempty(times_j) ? 0.0 : times_j[resolve_time_index(times_j, time_seconds, time_index)]

        return (
            lons = lons_j,
            lats = lats_j,
            depths = deps_j,
            times = times_j,
            time_seconds = sel_time,
            resolved_time = sel_time,
            depth_index = k_sel,
            depth_m = deps_j[k_sel],
            resolved_depth = deps_j[k_sel],
            u = u_j,
            v = v_j,
            w = w_j,
            speed = hypot.(u_j, v_j),
            temperature = T_j,
            salinity = S_j,
            density = diag.density,
            stratification = diag.stratification,
            salinity_stratification = diag.salinity_stratification,
            diffusion = diag.diffusion,
            viscosity = diag.viscosity,
            richardson_number = diag.richardson,
            vorticity = diag.vorticity,
            elevation = elev_j,
            bathymetry = bathymetry_j
        )
    end

    # 2. GeoData store directory (hydrodynamics.zarr), the ocean half's deliverable
    #
    # This is the branch a current run uses. `run_segment_simulation` hands back the run's
    # output *directory*, so `hydro_input` arrives as that path; the store inside is the only
    # place the ocean half now writes fields. Before this branch existed such a path fell
    # through every match and landed on the synthetic synthesis at the bottom, so a figure
    # of a real run silently drew invented flow.
    if hydro_input isa AbstractString && isdir(output_zarr_path(hydro_input))
        store = output_zarr_path(hydro_input)
        attrs = JSON3.read(read(joinpath(store, ".zattrs"), String))
        lons_s = Float64.(attrs["lon"])
        lats_s = Float64.(attrs["lat"])
        deps_s = Float64.(attrs["depth"])
        times_s = if isfile(joinpath(store, "time.json"))
            Float64.(JSON3.read(read(joinpath(store, "time.json"), String), Vector{Float64}))
        else
            [0.0]
        end
        nt = max(1, length(times_s))
        m_sel = clamp(resolve_time_index(times_s, time_seconds, time_index), 1, nt) - 1

        # One Zarr array per interval, read from `store/<field>/<m>`. Salinity was never a
        # store field (the callback writes u/v/w/T), so it stays `nothing` rather than being
        # filled from a constant the run never used -- `compute_hydrodynamic_diagnostics`
        # handles a missing salinity, and a value fabricated here would read as observed.
        #
        # The store keeps the model's *native* stagger, because the callback writes
        # `interior(velocities.u)` and friends rather than interpolating: u is one lon face
        # longer than the centre grid, v one lat face longer, w one depth face longer, and
        # T already centred. Blending each staggered axis to centres here keeps that fidelity
        # in the store while giving the figures the regular grid they interpolate against.
        rd(name) = isdir(joinpath(store, name, string(m_sel))) ?
                   collect(Zarr.zopen(joinpath(store, name, string(m_sel)))) : nothing
        _centre(arr::Nothing, ::Int, ::Int, ::Int) = nothing
        function _centre(arr, n_lon::Int, n_lat::Int, n_z::Int)
            out = arr
            size(out, 1) == n_lon + 1 && (out = 0.5 .* (out[1:end-1, :, :] .+ out[2:end, :, :]))
            size(out, 2) == n_lat + 1 && (out = 0.5 .* (out[:, 1:end-1, :] .+ out[:, 2:end, :]))
            size(out, 3) == n_z + 1 && (out = 0.5 .* (out[:, :, 1:end-1] .+ out[:, :, 2:end]))
            return out
        end
        shape = (length(lons_s), length(lats_s), length(deps_s))
        u_s = something(_centre(rd("u"), shape...), fill(NaN, shape...))
        v_s = something(_centre(rd("v"), shape...), fill(NaN, shape...))
        w_s = something(_centre(rd("w"), shape...), fill(NaN, shape...))
        T_s = something(_centre(rd("T"), shape...), fill(NaN, shape...))
        S_s = nothing

        bathymetry_s = read_immersed_bathymetry(hydro_input)
        diag = compute_hydrodynamic_diagnostics(lons_s, lats_s, deps_s, u_s, v_s, w_s, T_s, S_s)
        k_sel = resolve_depth_index(deps_s, depth, depth_level)
        sel_time = isempty(times_s) ? 0.0 : times_s[clamp(resolve_time_index(times_s, time_seconds, time_index), 1, nt)]

        return (
            lons = lons_s,
            lats = lats_s,
            depths = deps_s,
            times = times_s,
            time_seconds = sel_time,
            resolved_time = sel_time,
            depth_index = k_sel,
            depth_m = deps_s[clamp(k_sel, 1, length(deps_s))],
            resolved_depth = deps_s[clamp(k_sel, 1, length(deps_s))],
            u = u_s,
            v = v_s,
            w = w_s,
            speed = hypot.(u_s, v_s),
            temperature = T_s,
            salinity = S_s,
            density = diag.density,
            stratification = diag.stratification,
            salinity_stratification = diag.salinity_stratification,
            diffusion = diag.diffusion,
            viscosity = diag.viscosity,
            richardson_number = diag.richardson,
            vorticity = diag.vorticity,
            elevation = nothing,
            bathymetry = bathymetry_s
        )
    end

    # 3. Zarr Storage Connection
    if !isnothing(hydro_input) && (hydro_input isa Zarr.ZGroup)
        r_id = !isnothing(run_id) ? String(run_id) : begin
            runs_df = list_simulation_runs(hydro_input)
            nrow(runs_df) > 0 ? String(first(runs_df.run_id)) : "run_baseline_2025"
        end
        loaded = load_hydrodynamic_field(hydro_input, r_id; time_seconds = time_seconds, depth_level = depth_level)
        diag = compute_hydrodynamic_diagnostics(loaded.lons, loaded.lats, loaded.depths, loaded.u, loaded.v, loaded.w, loaded.temperature, loaded.salinity)
        k_sel = resolve_depth_index(loaded.depths, depth, depth_level)

        return (
            lons = loaded.lons,
            lats = loaded.lats,
            depths = loaded.depths,
            times = !isnothing(time_seconds) ? [Float64(time_seconds)] : [0.0],
            time_seconds = something(time_seconds, 0.0),
            resolved_time = something(time_seconds, 0.0),
            depth_index = k_sel,
            depth_m = loaded.depths[k_sel],
            resolved_depth = loaded.depths[k_sel],
            u = loaded.u,
            v = loaded.v,
            w = loaded.w,
            speed = hypot.(loaded.u, loaded.v),
            temperature = loaded.temperature,
            salinity = loaded.salinity,
            density = diag.density,
            stratification = diag.stratification,
            salinity_stratification = diag.salinity_stratification,
            diffusion = diag.diffusion,
            viscosity = diag.viscosity,
            richardson_number = diag.richardson,
            vorticity = diag.vorticity,
            elevation = loaded.elevation,
            # Resolve the real seabed: prefer an explicit bathymetry carried on the record,
            # else the ImmersedBoundaryGrid bottom height from the simulation JLD2. Never
            # fabricate a flat plane, which would wrongly mask the real water column.
            bathymetry = resolve_bathymetry_for_plot(
                loaded.lons, loaded.lats;
                sources = (hasproperty(loaded, :bathymetry) ? getproperty(loaded, :bathymetry) : nothing,
                           hasproperty(hydro_input, :jld2_path) ? getproperty(hydro_input, :jld2_path) : nothing),
                fallback = nothing
            )
        )
    end

    # 3. Direct Oceananigans Model Instance
    if !isnothing(hydro_input) && hasproperty(hydro_input, :velocities) && hasproperty(hydro_input, :tracers)
        g = hydro_input.grid
        coords = extract_grid_coordinates(g)
        m_lons = coords.lons
        m_lats = coords.lats
        m_depths = coords.depths

        u_arr = Array(interior(hydro_input.velocities.u))
        v_arr = Array(interior(hydro_input.velocities.v))
        w_arr = Array(interior(hydro_input.velocities.w))
        t_arr = haskey(hydro_input.tracers, :T) ? Array(interior(hydro_input.tracers.T)) : fill(4.5, size(u_arr))
        s_arr = haskey(hydro_input.tracers, :S) ? Array(interior(hydro_input.tracers.S)) : fill(33.0, size(u_arr))
        elev_mat = hasproperty(hydro_input, :free_surface) && hasproperty(hydro_input.free_surface, :η) ?
                   Array(interior(hydro_input.free_surface.η)) : zeros(Float64, length(m_lons), length(m_lats))

        bathy_mat = if g isa ImmersedBoundaryGrid && hasproperty(g.immersed_boundary, :bottom_height)
            Array(interior(g.immersed_boundary.bottom_height))
        else
            nothing
        end

        model_time = hasproperty(hydro_input, :clock) ? Float64(hydro_input.clock.time) : 0.0
        diag = compute_hydrodynamic_diagnostics(m_lons, m_lats, m_depths, u_arr, v_arr, w_arr, t_arr, s_arr)
        k_sel = resolve_depth_index(m_depths, depth, depth_level)

        return (
            lons = collect(Float64, m_lons),
            lats = collect(Float64, m_lats),
            depths = collect(Float64, m_depths),
            times = [model_time],
            time_seconds = model_time,
            resolved_time = model_time,
            depth_index = k_sel,
            depth_m = m_depths[k_sel],
            resolved_depth = m_depths[k_sel],
            u = Float64.(u_arr),
            v = Float64.(v_arr),
            w = Float64.(w_arr),
            speed = hypot.(Float64.(u_arr), Float64.(v_arr)),
            temperature = Float64.(t_arr),
            salinity = Float64.(s_arr),
            density = diag.density,
            stratification = diag.stratification,
            salinity_stratification = diag.salinity_stratification,
            diffusion = diag.diffusion,
            viscosity = diag.viscosity,
            richardson_number = diag.richardson,
            vorticity = diag.vorticity,
            elevation = Float64.(elev_mat),
            bathymetry = Float64.(bathy_mat)
        )
    end

    # 4. NamedTuple / Dict representation
    if !isnothing(hydro_input) && (hydro_input isa NamedTuple || hydro_input isa AbstractDict)
        get_field(keys_to_try, default_val) = begin
            for k in keys_to_try
                if hydro_input isa NamedTuple && hasproperty(hydro_input, k)
                    return getproperty(hydro_input, k)
                elseif hydro_input isa AbstractDict && (haskey(hydro_input, k) || haskey(hydro_input, string(k)))
                    return haskey(hydro_input, k) ? hydro_input[k] : hydro_input[string(k)]
                end
            end
            return default_val
        end

        target_lons = range(domain_lon[1], domain_lon[2], length = nx)
        target_lats = range(domain_lat[1], domain_lat[2], length = ny)

        in_lons = get_field((:lons, :grid_lons, :lon), target_lons)
        in_lats = get_field((:lats, :grid_lats, :lat), target_lats)
        in_depths = get_field((:depths, :grid_depths, :depth), t_depths)

        nx_in = length(in_lons)
        ny_in = length(in_lats)
        nz_in = length(in_depths)

        in_u = get_field((:u, :u_velocity, :u_mean), zeros(Float64, nx_in, ny_in, nz_in))
        in_v = get_field((:v, :v_velocity, :v_mean), zeros(Float64, nx_in, ny_in, nz_in))
        in_w = get_field((:w, :w_velocity), zeros(Float64, nx_in, ny_in, nz_in))
        in_t = get_field((:temperature, :T, :temp), fill(4.5, nx_in, ny_in, nz_in))
        in_s = get_field((:salinity, :S, :sal), fill(33.0, nx_in, ny_in, nz_in))
        in_elev = get_field((:elevation, :η, :eta, :ssh), zeros(Float64, nx_in, ny_in))
        in_bathy = get_field((:bathymetry, :bathy, :elevation_bottom), nothing)

        # Handle 4D time slices if present
        slice_3d(arr) = begin
            if ndims(arr) == 4
                t_sub = resolve_time_index(get_field((:times, :t), [0.0]), time_seconds, time_index)
                return arr[:, :, :, clamp(t_sub, 1, size(arr, 4))]
            elseif ndims(arr) == 2
                return reshape(arr, size(arr, 1), size(arr, 2), 1)
            else
                return arr
            end
        end

        u_3d = Float64.(slice_3d(in_u))
        v_3d = Float64.(slice_3d(in_v))
        w_3d = Float64.(slice_3d(in_w))
        t_3d = Float64.(slice_3d(in_t))
        s_3d = Float64.(slice_3d(in_s))

        times_arr = get_field((:times, :t), [something(time_seconds, 0.0)])
        sel_time = isempty(times_arr) ? 0.0 : times_arr[resolve_time_index(times_arr, time_seconds, time_index)]
        diag = compute_hydrodynamic_diagnostics(in_lons, in_lats, in_depths, u_3d, v_3d, w_3d, t_3d, s_3d)
        k_sel = resolve_depth_index(in_depths, depth, depth_level)

        return (
            lons = collect(Float64, in_lons),
            lats = collect(Float64, in_lats),
            depths = collect(Float64, in_depths),
            times = collect(Float64, times_arr),
            time_seconds = Float64(sel_time),
            resolved_time = Float64(sel_time),
            depth_index = k_sel,
            depth_m = Float64(in_depths[k_sel]),
            resolved_depth = Float64(in_depths[k_sel]),
            u = u_3d,
            v = v_3d,
            w = w_3d,
            speed = hypot.(u_3d, v_3d),
            temperature = t_3d,
            salinity = s_3d,
            density = diag.density,
            stratification = diag.stratification,
            salinity_stratification = diag.salinity_stratification,
            diffusion = diag.diffusion,
            viscosity = diag.viscosity,
            richardson_number = diag.richardson,
            vorticity = diag.vorticity,
            elevation = Float64.(in_elev isa AbstractMatrix ? in_elev : fill(0.0, nx_in, ny_in)),
            bathymetry = resolve_bathymetry_for_plot(
                collect(Float64, in_lons), collect(Float64, in_lats);
                sources = (in_bathy, hydro_input), fallback = nothing
            )
        )
    end

    # 5. Default Realistic Regional Hydrodynamic Synthesis with Temporal Evolution
    t_sec = something(time_seconds, 0.0)
    target_lons = range(domain_lon[1], domain_lon[2], length = nx)
    target_lats = range(domain_lat[1], domain_lat[2], length = ny)

    u_mat = zeros(Float64, nx, ny, nz)
    v_mat = zeros(Float64, nx, ny, nz)
    w_mat = zeros(Float64, nx, ny, nz)
    t_mat = zeros(Float64, nx, ny, nz)
    s_mat = zeros(Float64, nx, ny, nz)
    elev_mat = zeros(Float64, nx, ny)
    bathy_mat = zeros(Float64, nx, ny)

    lons_vec = collect(Float64, target_lons)
    lats_vec = collect(Float64, target_lats)

    # Semi-diurnal M2 tidal phase and seasonal warming phase
    omega_m2 = 2.0 * π / (12.42 * 3600.0)
    tide_phase = cos(omega_m2 * t_sec)
    season_phase = sin(2.0 * π * (t_sec - 60.0 * 86400.0) / (365.25 * 86400.0))

    for i in 1:nx, j in 1:ny
        lon = lons_vec[i]
        lat = lats_vec[j]
        x_norm = (lon - domain_lon[1]) / (domain_lon[2] - domain_lon[1])
        y_norm = (lat - domain_lat[1]) / (domain_lat[2] - domain_lat[1])

        # Realistic Scotian Shelf Bathymetry profile
        dist_to_coast = (lat - 43.5) + 0.4 * (lon + 63.5)
        b_elev = if dist_to_coast > 1.2
            -45.0 - 55.0 * (1.0 - y_norm)
        elseif dist_to_coast > -0.2
            -110.0 - 140.0 * sin(π * x_norm)
        else
            -350.0 - 2200.0 * (1.0 - (dist_to_coast + 1.5) / 1.3)^2
        end
        bathy_mat[i, j] = clamp(b_elev, -3200.0, -25.0)

        # Free Surface SSH: cross-shelf steric setup + tidal oscillation
        elev_mat[i, j] = (0.05 * sin(2.0 * π * x_norm) - 0.04 * cos(π * y_norm)) + 0.35 * tide_phase

        for k in 1:nz
            z = t_depths[k]
            depth_factor = exp(z / 75.0)

            # Scotian Shelf coast bearing ~45° (SW-NE).  The Nova Scotia
            # Current flows southwestward, so along-coast unit vector is
            # (cos 225°, sin 225°) = (-√2/2, -√2/2) in (u_E, v_N).
            # coast_angle = 45° ⟹ cos_α = sin_α = √2/2 ≈ 0.7071
            cos_α = 0.7071
            sin_α = 0.7071

            jet_core = exp(-((dist_to_coast - 0.4)^2) / 0.18)

            # Jet speed along coast and weak cross-shelf return flow
            V_along = 0.11 * depth_factor * (0.6 + 0.8 * jet_core)
            V_cross = 0.02 * depth_factor * (0.15 * jet_core)

            # Tidal barotropic oscillation (predominantly cross-shelf on
            # Scotian Shelf — 90° to coast bearing)
            u_tide = 0.06 * tide_phase * cos_α +
                     0.04 * sin(omega_m2 * t_sec) * (-sin_α)
            v_tide = 0.06 * tide_phase * sin_α +
                     0.04 * sin(omega_m2 * t_sec) *   cos_α

            # Along-coast = SW ⟹ negative along both x and y
            u_base = -V_along * cos_α + V_cross * sin_α +
                     0.015 * sin(2.0 * π * y_norm) + u_tide
            v_base = -V_along * sin_α - V_cross * cos_α +
                     0.010 * cos(2.0 * π * x_norm) + v_tide

            u_mat[i, j, k] = u_base
            v_mat[i, j, k] = v_base

            # Vertical velocity (coastal upwelling / downwelling)
            w_mat[i, j, k] = 0.00035 * sin(2.0 * π * x_norm) * sin(π * y_norm) * (1.0 + z / 100.0)

            # Thermal Stratification: Surface warm layer + seasonal phase, CIL at -50m, warm deep slope
            t_surface = 14.2 + 2.5 * season_phase - 2.5 * y_norm + 1.2 * x_norm
            t_cil = 2.2 + 0.8 * sin(π * x_norm)
            t_slope = 7.5 + 1.5 * (1.0 - y_norm)

            t_val = if z > -20.0
                t_surface + (z / 20.0) * (t_surface - 6.0)
            elseif z > -70.0
                t_cil + ((z + 50.0) / 30.0)^2 * 2.5
            else
                t_cil + ((abs(z) - 70.0) / 50.0) * (t_slope - t_cil)
            end
            t_mat[i, j, k] = clamp(t_val, 0.5, 19.5)

            # Practical Salinity stratification
            s_val = 31.4 + 1.8 * (1.0 - y_norm) + 1.2 * x_norm + (abs(z) / 100.0) * 1.1
            s_mat[i, j, k] = clamp(s_val, 30.2, 35.6)
        end
    end

    spd_mat = hypot.(u_mat, v_mat)

    # Mask over land. The rings are the run's own fetched coastline when one is
    # passed, and the region's bundled rings only when it is not: a figure that
    # masks against a different land than the model ran on silently draws land in
    # the water. The data layer no longer defaults to a regional ring set, so the
    # region's rings are named here as the fallback rather than as the choice.
    local coast = resolve_land_mask_coastline(coastline; coastline_path = coastline_path)
    for i in 1:nx, j in 1:ny
        if is_point_on_land(lons_vec[i], lats_vec[j]; coastline = coast)
            for k in 1:nz
                u_mat[i, j, k]   = NaN
                v_mat[i, j, k]   = NaN
                w_mat[i, j, k]   = NaN
                spd_mat[i, j, k] = NaN
                t_mat[i, j, k]   = NaN
                s_mat[i, j, k]   = NaN
            end
            elev_mat[i, j]  = NaN
            bathy_mat[i, j] = NaN
        end
    end

    diag = compute_hydrodynamic_diagnostics(lons_vec, lats_vec, t_depths, u_mat, v_mat, w_mat, t_mat, s_mat)
    k_sel = resolve_depth_index(t_depths, depth, depth_level)

    return (
        lons = lons_vec,
        lats = lats_vec,
        depths = t_depths,
        times = [t_sec],
        time_seconds = t_sec,
        resolved_time = t_sec,
        depth_index = k_sel,
        depth_m = t_depths[k_sel],
        resolved_depth = t_depths[k_sel],
        u = u_mat,
        v = v_mat,
        w = w_mat,
        speed = spd_mat,
        temperature = t_mat,
        salinity = s_mat,
        density = diag.density,
        stratification = diag.stratification,
        salinity_stratification = diag.salinity_stratification,
        diffusion = diag.diffusion,
        viscosity = diag.viscosity,
        richardson_number = diag.richardson,
        vorticity = diag.vorticity,
        elevation = elev_mat,
        bathymetry = bathy_mat
    )
end

