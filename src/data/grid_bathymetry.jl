"""
    grid_bathymetry.jl

Grid construction and immersed boundary setup for shelf hydrodynamic modeling.

ParticleTracking-specific functions requiring Oceananigans grid types.
All generic geospatial operations (bathymetry I/O, regridding, coastline, marine cell
extraction, smoothing, buffering) are in GeoData.core.bathy. 
"""

using Oceananigans
using Oceananigans.Grids: Face, Center, znode
using Oceananigans.OrthogonalSphericalShellGrids: RotatedLatitudeLongitudeGrid, OrthogonalSphericalShellGrid
using Oceananigans.Architectures: architecture, on_architecture, CPU, GPU
using Oceananigans.ImmersedBoundaries: ImmersedBoundaryGrid, GridFittedBottom
using GeoData
using GeoDataSources
using Random
import Interpolations
using Interpolations: interpolate, Gridded, Linear, extrapolate

"""
    build_shelf_grid(;
        architecture = :cpu,
        lon_range=(-68.0, -57.0),
        lat_range=(42.0, 47.0),
        z_range=(-1000.0, 0.0),
        z_faces = nothing,
        grid_size=(50, 50, 10),
        halo = (7, 7, 5),
        topology=(Bounded, Bounded, Bounded),
        fallback_to_cpu::Bool = false,
        orientation::Symbol = :standard,
        rotation_azimuth::Real = 50.0,
        center_coords::Tuple{Real, Real} = (-63.0, 44.0),
        along_shelf_extent_km::Real = 850.0,
        cross_shelf_extent_km::Real = 220.0
    )

Construct a regional ocean model computational grid on the specified architecture.
Supports both standard geographic spherical grids (`LatitudeLongitudeGrid`) and
orthogonal rotated curvilinear grids (`RotatedLatitudeLongitudeGrid`) oriented along
the continental margin.

# Inputs
- `architecture`: Target device backend (`:cpu` or `:gpu`).
- `lon_range`: Zonal geographic bounds `(lon_min, lon_max)` in °E (used for `:standard`).
- `lat_range`: Meridional geographic bounds `(lat_min, lat_max)` in °N (used for `:standard`).
- `z_range`: Depth bounds `(z_min, z_max)` in meters.
- `z_faces`: Optional custom vector of vertical face coordinates.
- `grid_size`: Discretization tuple `(Nx, Ny, Nz)`.
- `halo`: Boundary ghost cell halo tuple `(Hx, Hy, Hz)`.
- `topology`: Spatial topology tuple (default `(Bounded, Bounded, Bounded)`).
- `fallback_to_cpu`: Whether to fall back to CPU if GPU initialization fails.
- `orientation`: `:standard` (geographic box) or `:rotated` (orthogonal along-shelf grid).
- `rotation_azimuth`: Shelf strike azimuth in degrees clockwise from North (default 50.0°).
- `center_coords`: Geographic center `(lon, lat)` for rotated grid (default `(-63.0, 44.0)`).
- `along_shelf_extent_km`: Domain dimension along shelf strike in kilometers (default 850.0 km).
- `cross_shelf_extent_km`: Domain dimension across shelf in kilometers (default 220.0 km).

# Outputs
- `Oceananigans.Grids.AbstractGrid`: Constructed base computational grid.
"""
function build_shelf_grid(;
    architecture = :cpu,
    lon_range::Tuple{Real, Real} = (-71.0, -53.0),
    lat_range::Tuple{Real, Real} = (40.0, 48.5),
    z_range::Tuple{Real, Real} = (-3500.0, 0.0),
    z_faces::Union{Nothing, AbstractVector, Function} = nothing,
    grid_size::Tuple{Int, Int, Int} = (50, 50, 10),
    halo::Tuple{Int, Int, Int} = (7, 7, 5),
    topology::Tuple = (Bounded, Bounded, Bounded),
    fallback_to_cpu::Bool = false,
    orientation::Symbol = :standard,
    rotation_azimuth::Real = 50.0,
    center_coords::Tuple{Real, Real} = (-63.0, 44.0),
    along_shelf_extent_km::Real = 850.0,
    cross_shelf_extent_km::Real = 220.0
)
    if orientation === :standard
        if lon_range[1] >= lon_range[2]
            error("Invalid longitude range: $(lon_range). lon_min must be < lon_max.")
        end
        if lat_range[1] >= lat_range[2]
            error("Invalid latitude range: $(lat_range). lat_min must be < lat_max.")
        end
    end
    if any(s <= 0 for s in grid_size)
        error("Grid size dimensions must all be positive integers: $(grid_size)")
    end

    z_specification = if !isnothing(z_faces)
        if z_faces isa AbstractVector
            if length(z_faces) != grid_size[3] + 1
                error("Custom z_faces length $(length(z_faces)) must equal Nz + 1 = $(grid_size[3] + 1)")
            end
            if !issorted(z_faces)
                error("Custom z_faces must be strictly monotonically increasing.")
            end
        end
        z_faces
    else
        if z_range[1] >= z_range[2]
            error("Invalid vertical range: $(z_range). z_min must be < z_max.")
        end
        z_range
    end

    arch = resolve_architecture(architecture; fallback_to_cpu = fallback_to_cpu)

    if orientation === :rotated
        rot = compute_rotated_pole(
            center_coords[1], center_coords[2], rotation_azimuth;
            along_shelf_extent_km = along_shelf_extent_km,
            cross_shelf_extent_km = cross_shelf_extent_km
        )
        grid = RotatedLatitudeLongitudeGrid(
            arch;
            size = grid_size,
            longitude = rot.lon_range,
            latitude = rot.lat_range,
            z = z_specification,
            north_pole = rot.north_pole,
            topology = topology,
            halo = halo
        )
        return grid
    else
        grid = LatitudeLongitudeGrid(
            arch;
            size = grid_size,
            longitude = lon_range,
            latitude = lat_range,
            z = z_specification,
            topology = topology,
            halo = halo
        )
        return grid
    end
end

"""
    compute_rotated_pole(
        lon_center::Real,
        lat_center::Real,
        azimuth::Real;
        along_shelf_extent_km::Real = 850.0,
        cross_shelf_extent_km::Real = 220.0,
        radius::Real = 6371000.0
    ) -> NamedTuple

Compute the Euler rotation pole `(lon_pole, lat_pole)` and the unrotated spherical
coordinate bounds `(lon_range, lat_range)` for an orthogonal curvilinear grid
oriented at bearing `azimuth` degrees clockwise from North at domain center
`(lon_center, lat_center)`.

# Mathematical Formulation
Let ``(\\lambda_0, \\phi_0)`` be the domain center in radians and ``\\theta`` be
the azimuth angle measured clockwise from true North. The local orthonormal
tangent vectors are:
```math
\\hat{e} = [-\\sin\\lambda_0, \\cos\\lambda_0, 0]^T, \\quad
\\hat{n} = [-\\sin\\phi_0 \\cos\\lambda_0, -\\sin\\phi_0 \\sin\\lambda_0, \\cos\\phi_0]^T
```
The along-shelf tangent vector is ``\\hat{t} = \\sin\\theta \\hat{e} + \\cos\\theta \\hat{n}``.
The pole ``\\hat{n}_p`` of the great circle passing through ``(\\lambda_0, \\phi_0)``
with tangent ``\\hat{t}`` is:
```math
\\hat{n}_p = \\hat{t} \\times \\hat{r}_0 = -\\cos\\theta \\hat{e} + \\sin\\theta \\hat{n}
```
In Oceananigans' `RotatedLatitudeLongitudeGrid`, the rotation aligns the rotated
equator (``\\phi' = 0``) along this great circle when the grid pole is placed at
``\\hat{n}_p``.

# Inputs
- `lon_center`: Central longitude in °E.
- `lat_center`: Central latitude in °N.
- `azimuth`: Shelf strike azimuth in degrees clockwise from North.
- `along_shelf_extent_km`: Domain extent along continental margin (km).
- `cross_shelf_extent_km`: Domain extent across continental margin (km).
- `radius`: Mean spherical planet radius in meters (default 6,371,000.0 m).

# Outputs
- `NamedTuple`: `(north_pole = (lon_p, lat_p), lon_range = (λ'_min, λ'_max), lat_range = (φ'_min, φ'_max))`
"""
function compute_rotated_pole(
    lon_center::Real,
    lat_center::Real,
    azimuth::Real;
    along_shelf_extent_km::Real = 850.0,
    cross_shelf_extent_km::Real = 220.0,
    radius::Real = 6371000.0
)
    λ0_rad = deg2rad(Float64(lon_center))
    φ0_rad = deg2rad(Float64(lat_center))
    θ_rad  = deg2rad(Float64(azimuth))

    e_hat = [-sin(λ0_rad), cos(λ0_rad), 0.0]
    n_hat = [-sin(φ0_rad) * cos(λ0_rad), -sin(φ0_rad) * sin(λ0_rad), cos(φ0_rad)]
    np = -cos(θ_rad) .* e_hat .+ sin(θ_rad) .* n_hat

    if np[3] < 0.0
        np = -np
    end

    lat_p = rad2deg(asin(np[3]))
    lon_p = rad2deg(atan(np[2], np[1]))

    dλ = deg2rad(lon_p)
    dφ = π / 2.0 - deg2rad(lat_p)

    Ry_inv = [ cos(dφ) 0.0 -sin(dφ); 0.0 1.0 0.0; sin(dφ) 0.0 cos(dφ)]
    Rz_inv = [ cos(dλ) sin(dλ) 0.0; -sin(dλ) cos(dλ) 0.0; 0.0 0.0 1.0]

    r_target = [cos(φ0_rad) * cos(λ0_rad), cos(φ0_rad) * sin(λ0_rad), sin(φ0_rad)]
    r_prime = Ry_inv * Rz_inv * r_target

    λ_prime_deg = rad2deg(atan(r_prime[2], r_prime[1]))

    dλ_deg = (Float64(along_shelf_extent_km) * 1000.0 / Float64(radius)) * (180.0 / π)
    dφ_deg = (Float64(cross_shelf_extent_km) * 1000.0 / Float64(radius)) * (180.0 / π)

    lon_range = (λ_prime_deg - 0.5 * dλ_deg, λ_prime_deg + 0.5 * dλ_deg)
    lat_range = (-0.5 * dφ_deg, 0.5 * dφ_deg)

    return (
        north_pole = (lon_p, lat_p),
        lon_range = lon_range,
        lat_range = lat_range
    )
end

"""
    grid_geographic_bounds(grid) -> (Tuple{Float64, Float64}, Tuple{Float64, Float64})

Compute the geographic bounding box `((lon_min, lon_max), (lat_min, lat_max))` in
degrees East and North enclosing `grid`'s interior domain.
"""
function grid_geographic_bounds(grid)
    base_g = grid isa ImmersedBoundaryGrid ? grid.underlying_grid : grid
    if base_g isa LatitudeLongitudeGrid
        lon_range = (Float64(base_g.λᶠᵃᵃ[1]), Float64(base_g.λᶠᵃᵃ[base_g.Nx + 1]))
        lat_range = (Float64(base_g.φᵃᶠᵃ[1]), Float64(base_g.φᵃᶠᵃ[base_g.Ny + 1]))
        return lon_range, lat_range
    else
        cpu_g = architecture(base_g) isa GPU ? on_architecture(CPU(), base_g) : base_g
        nx, ny = cpu_g.Nx, cpu_g.Ny
        lons = [Float64(cpu_g.λᶜᶜᵃ[i, j]) for i in 1:nx, j in 1:ny]
        lats = [Float64(cpu_g.φᶜᶜᵃ[i, j]) for i in 1:nx, j in 1:ny]
        lon_min, lon_max = extrema(lons)
        lat_min, lat_max = extrema(lats)
        return (lon_min - 0.2, lon_max + 0.2), (lat_min - 0.2, lat_max + 0.2)
    end
end

"""
    load_vertical_grid_csv(filepath::AbstractString) -> Vector{Float64}

Load vertical layer face coordinates from a CSV file (e.g. `scotian_shelf_vertical_grid.csv`).
"""
function load_vertical_grid_csv(filepath::AbstractString)::Vector{Float64}
    if !isfile(filepath)
        error("Vertical grid file does not exist: $(filepath)")
    end
    lines = readlines(filepath)
    if length(lines) < 2
        error("Vertical grid file $(filepath) is empty or missing data.")
    end
    faces = Float64[]
    for l in lines[2:end]
        parts = split(strip(l), ',')
        if length(parts) >= 3
            bot_val = parse(Float64, parts[2])
            top_val = parse(Float64, parts[3])
            if isempty(faces)
                push!(faces, bot_val)
            end
            push!(faces, top_val)
        end
    end
    if !issorted(faces)
        if issorted(faces; rev = true)
            error(
                "Vertical grid file $(filepath) is written positive-downward (first face " *
                "$(first(faces)) m, last $(last(faces)) m). Oceananigans requires z faces " *
                "NEGATIVE and ascending, ending at 0 m at the surface: rewrite so the first " *
                "row is the deepest layer and the last row's top is 0.0.")
        end
        error("Parsed vertical grid faces from $(filepath) are not monotonically sorted.")
    end
    abs(last(faces)) <= 1e-6 || error(
        "Vertical grid faces from $(filepath) must end at 0.0 m (the surface); got " *
        "$(last(faces)) m.")
    first(faces) < 0 || error(
        "Vertical grid faces from $(filepath) must be negative below the surface; got " *
        "$(first(faces)) m as the first (deepest) face.")
    return faces
end

"""
    build_immersed_grid(
        grid::LatitudeLongitudeGrid,
        bathymetry::Union{AbstractMatrix, GeoDataset};
        varname::AbstractString = "elevation"
    )

Wrap a base `LatitudeLongitudeGrid` with an `ImmersedBoundaryGrid` using
`GridFittedBottom` representing ocean seafloor topography.

`bathymetry` is either the `(Nx, Ny)` topography matrix itself, or a `GeoDataset` it is
read from through `GeoDataSources.bathymetry_arrays`, which selects the elevation variable
and resolves the `(lon, lat)` orientation from the dimensions the dataset declared. A file
path is deliberately not accepted: which file a source resolves to, and in what format, is
decided inside `GeoDataSources`, and a caller that passes a path is making that decision
itself.
"""
function build_immersed_grid(
    grid,
    bathymetry::Union{AbstractMatrix, GeoDataset};
    varname::AbstractString = "elevation"
)
    topo_matrix::Matrix{Float64} = if bathymetry isa GeoDataset
        elev_var = haskey(bathymetry.variables, varname) ? varname :
                   first(GeoData.variables_like(bathymetry, ["elevation", "altitude", "z", "topo"]))
        elev_ga = bathymetry.variables[elev_var]
        perm = GeoData.dim_permutation(elev_ga.dims, :lon, :lat)
        raw = Array{Float64}(elev_ga.data)
        ndims(raw) == 1 ? reshape(raw, :, 1) : permutedims(raw, perm)
    else
        Matrix{Float64}(bathymetry)
    end

    nx, ny, _ = size(grid)
    t_nx, t_ny = size(topo_matrix)

    if (nx != t_nx) || (ny != t_ny)
        error(
            "Bathymetry dimensions ($(t_nx), $(t_ny)) do not match grid horizontal " *
            "dimensions ($(nx), $(ny))."
        )
    end

    base_g = grid isa ImmersedBoundaryGrid ? grid.underlying_grid : grid
    arch = architecture(grid)
    z_max = arch isa GPU ? 0.0 : znode(base_g.Nz + 1, base_g, Face())
    max_topo = maximum(topo_matrix)
    if max_topo > z_max
        @warn "Maximum bathymetry elevation ($(max_topo) m) exceeds surface " *
              "height ($(z_max) m). Emerged land points present."
    end

    arch = architecture(grid)
    arch_topo = on_architecture(arch, topo_matrix)
    immersed_grid = ImmersedBoundaryGrid(grid, GridFittedBottom(arch_topo))
    return immersed_grid
end

"""
    is_in_bay_of_fundy(lon::Real, lat::Real) -> Bool

Determine whether geographic coordinates `(lon, lat)` in degrees East and North fall
within the hypertidal Bay of Fundy, Minas Basin, or Chignecto Bay exclusion zone.
"""
@inline function is_in_bay_of_fundy(lon::Real, lat::Real)::Bool
    if lat < 44.4 || lat > 46.0
        return false
    end
    if lon < -67.3 || lon > -64.0
        return false
    end
    if lat <= 44.8 && lon > -65.6
        return false
    end
    if lat <= 45.1 && lon > -65.2
        return false
    end
    if lat >= 45.7 && lon > -64.2
        return false
    end
    return true
end

"""
    _grid_interior_centres(grid) -> (Vector{Float64}, Vector{Float64})

Longitude and latitude of `grid`'s interior cell centres: `λᶜᵃᵃ[1 : Nx]` and `φᵃᶜᵃ[1 : Ny]`.

These are the coordinates `GridFittedBottom` and every spatial output field are indexed by,
so anything that wants "where on Earth is cell (i, j)" must use these. There was one
implementation per caller, and they disagreed: one sliced `[Hx+1 : Hx+Nx]`, which is mostly
halo, and one rebuilt them from the face bounds with a `range` of length `Nx`, which is the
face extent rather than the centres and so lands half a cell off. Both read as plausible and
both were wrong, which is why this is now one function.
"""
function _grid_interior_centres(grid)
    base_g = grid isa ImmersedBoundaryGrid ? grid.underlying_grid : grid
    cpu_g = architecture(base_g) isa GPU ? on_architecture(CPU(), base_g) : base_g
    if cpu_g isa LatitudeLongitudeGrid
        return ([Float64(cpu_g.λᶜᵃᵃ[i]) for i in 1:cpu_g.Nx],
                [Float64(cpu_g.φᵃᶜᵃ[j]) for j in 1:cpu_g.Ny])
    else
        nx, ny = cpu_g.Nx, cpu_g.Ny
        lons = [Float64(cpu_g.λᶜᶜᵃ[i, j]) for i in 1:nx, j in 1:ny]
        lats = [Float64(cpu_g.φᶜᶜᵃ[i, j]) for i in 1:nx, j in 1:ny]
        return (lons, lats)
    end
end

"""
    _regrid_curvilinear_2d(
        src_lon::AbstractVector,
        src_lat::AbstractVector,
        src_data::AbstractMatrix,
        target_lons::AbstractMatrix,
        target_lats::AbstractMatrix
    ) -> Matrix{Float64}

Bilinear interpolation of gridded bathymetric elevation onto a 2D curvilinear coordinate
mesh `(target_lons, target_lats)`.
"""
function _regrid_curvilinear_2d(
    src_lon::AbstractVector,
    src_lat::AbstractVector,
    src_data::AbstractMatrix,
    target_lons::AbstractMatrix,
    target_lats::AbstractMatrix
)
    lons = Float64.(collect(src_lon))
    lats = Float64.(collect(src_lat))
    field = Array{Float64}(src_data)
    if size(field) == (length(lats), length(lons))
        field = permutedims(field, (2, 1))
    end
    if !issorted(lons)
        p = sortperm(lons)
        lons = lons[p]
        field = field[p, :]
    end
    if !issorted(lats)
        p = sortperm(lats)
        lats = lats[p]
        field = field[:, p]
    end
    itp = interpolate((lons, lats), field, Gridded(Linear()))
    itp_flat = extrapolate(itp, Interpolations.Flat())

    nx, ny = size(target_lons)
    out = Matrix{Float64}(undef, nx, ny)
    for j in 1:ny, i in 1:nx
        out[i, j] = itp_flat(Float64(target_lons[i, j]), Float64(target_lats[i, j]))
    end
    return out
end

"""
    _fit_bathymetry_to_grid(
        grid,
        elevation::AbstractMatrix,
        lon::AbstractVector,
        lat::AbstractVector;
        min_water_depth::Real = 10.0,
        mask_bay_of_fundy::Bool = false
    ) -> Matrix{Float64}

Regrid `(elevation, lon, lat)` onto `grid`'s interior cell centres and condition the result
for use as a `GridFittedBottom`.

This is the array-level half of `build_immersed_grid_from_real_data` -- bilinear regridding,
the Bay of Fundy mask, smoothing, and the depth floor -- and it is separate from that
function so it can be exercised without fetching anything: acquisition is the only part of
`build_immersed_grid_from_real_data` that leaves the process.

The target coordinates are the grid's interior cell centres, from `_grid_interior_centres`,
because `GridFittedBottom` indexes its array by interior cell. The previous form was
`range(λᶠᵃᵃ[1], λᶠᵃᵃ[Nx+1], length = Nx)`, which is a *different* set of points: it spans the
interior face extent rather than the centres, so every value landed half a cell away from the
cell it was stored in. A half-cell offset is invisible to any test that only checks that an
`ImmersedBoundaryGrid` came back, which is why it survived.

`min_water_depth` is applied as a floor on the *smoothed* topography, so a cell whose
smoothing pulled it below the floor is raised to it.
"""
function _fit_bathymetry_to_grid(
    grid,
    elevation::AbstractMatrix,
    lon::AbstractVector,
    lat::AbstractVector;
    min_water_depth::Real = 10.0,
    max_water_depth::Real = 0.0,
    mask_bay_of_fundy::Bool = false
)
    nx, ny, _ = size(grid)
    target_lons, target_lats = _grid_interior_centres(grid)

    regridded_topo = if target_lons isa AbstractVector && target_lats isa AbstractVector
        regrid_2d_field(lon, lat, elevation, target_lons, target_lats)
    else
        _regrid_curvilinear_2d(lon, lat, elevation, target_lons, target_lats)
    end

    if mask_bay_of_fundy
        n_masked = 0
        for j in 1:ny, i in 1:nx
            cell_lon = target_lons isa AbstractMatrix ? target_lons[i, j] : target_lons[i]
            cell_lat = target_lats isa AbstractMatrix ? target_lats[i, j] : target_lats[j]
            if is_in_bay_of_fundy(cell_lon, cell_lat)
                regridded_topo[i, j] = 0.0
                n_masked += 1
            end
        end
        @info "Bay of Fundy masking active: set $(n_masked) cells to solid land (0.0 m)."
    end

    # Condition bathymetry
    h_floor = Float64(min_water_depth)
    h_ceiling = Float64(max_water_depth)
    conditioned_topo = smooth_bathymetry(
        regridded_topo,
        passes = 4,
        alpha = 0.5,
        h_min = h_floor,
        h_max = h_ceiling
    )

    if mask_bay_of_fundy
        for j in 1:ny, i in 1:nx
            cell_lon = target_lons isa AbstractMatrix ? target_lons[i, j] : target_lons[i]
            cell_lat = target_lats isa AbstractMatrix ? target_lats[i, j] : target_lats[j]
            if is_in_bay_of_fundy(cell_lon, cell_lat)
                conditioned_topo[i, j] = 0.0
            end
        end
    end

    return conditioned_topo
end

"""
    build_immersed_grid_from_real_data(
        grid::LatitudeLongitudeGrid;
        source = nothing,
        tier = nothing,
        varname::AbstractString = "elevation",
        output_path::Union{Nothing, AbstractString} = nothing,
        force::Bool = false,
        min_water_depth::Real = 10.0,
        mask_bay_of_fundy::Bool = false,
        kwargs...
    ) -> ImmersedBoundaryGrid

Construct an `ImmersedBoundaryGrid` by interpolating real-world bathymetry onto the target
model grid coordinates.

The bathymetry is requested from `GeoDataSources` by `source`, over the domain taken from
`grid` itself. Nothing here names a file, a format, or a backend: which product the source
resolves to, at which `tier`, and whether it is fetched or already cached are all decided
inside `GeoDataSources`. `source = nothing` asks it to choose, which it does by coverage --
the CHS 15-arc-second soundings where the domain lies inside them, the global ETOPO 2022
model everywhere else. See `GeoDataSources.preferred_bathymetry_source`.

`output_path`, when given, is where that resolution is made durable, so this grid and every
later reader in a run see one sea floor rather than one per fetch. `force` re-fetches past a
cache.

# Example
```julia
# The CHS soundings, because the domain asked for is the Scotian Shelf.
grid = build_immersed_grid_from_real_data(
    base_grid; source = "scotian_shelf",
    output_path = joinpath("inputs", "bathymetry_active.nc"))
```
"""
function build_immersed_grid_from_real_data(
    grid;
    source = nothing,
    tier = nothing,
    varname::AbstractString = "elevation",
    output_path::Union{Nothing, AbstractString} = nothing,
    force::Bool = false,
    min_water_depth::Real = 10.0,
    max_water_depth::Real = 0.0,
    mask_bay_of_fundy::Bool = false,
    kwargs...
)
    base_g = grid isa ImmersedBoundaryGrid ? grid.underlying_grid : grid
    lon_range, lat_range = grid_geographic_bounds(base_g)

    bathy_elevation, bathy_lon, bathy_lat = if !force && !isnothing(output_path) && (isfile(output_path) || isdir(output_path))
        ds = GeoData.geoload(output_path)
        elev_var = haskey(ds.variables, varname) ? varname :
                   first(GeoData.variables_like(ds, ["elevation", "altitude", "z", "topo"]))
        elev_ga = ds.variables[elev_var]
        perm = GeoData.dim_permutation(elev_ga.dims, :lon, :lat)
        lon_c, _ = GeoData.axis(ds, :lon)
        lat_c, _ = GeoData.axis(ds, :lat)
        raw = Array{Float64}(elev_ga.data)
        elev = ndims(raw) == 1 ? reshape(raw, :, 1) : permutedims(raw, perm)
        (elev, lon_c, lat_c)
    else
        bathy = GeoDataSources.bathymetry_field(;
            source = source,
            lon_range = lon_range,
            lat_range = lat_range,
            tier = tier,
            varname = varname,
            output_path = output_path,
            force = force,
            kwargs...
        )
        (bathy.elevation, bathy.lon, bathy.lat)
    end

    conditioned_topo = _fit_bathymetry_to_grid(
        base_g, bathy_elevation, bathy_lon, bathy_lat;
        min_water_depth = min_water_depth,
        max_water_depth = max_water_depth,
        mask_bay_of_fundy = mask_bay_of_fundy
    )

    return build_immersed_grid(grid, conditioned_topo)
end

"""
    extract_grid_coordinates(grid) -> NamedTuple

Extract 1D cell-center spatial coordinates `(lons, lats, depths)` from an
Oceananigans computational grid, returning geographic coordinates in **degrees**
and the vertical coordinate in **metres**.
"""
function extract_grid_coordinates(grid)
    base_g = grid isa ImmersedBoundaryGrid ? grid.underlying_grid : grid
    cpu_g = architecture(base_g) isa GPU ? on_architecture(CPU(), base_g) : base_g
    nx, ny, nz = cpu_g.Nx, cpu_g.Ny, cpu_g.Nz

    # Geographic horizontal coordinates (degrees), halo region stripped (interior indices 1:nx,
    # 1:ny). Through `_grid_interior_centres`, so the fit and the output fields cannot disagree
    # about where a cell is.
    lons, lats = _grid_interior_centres(cpu_g)

    # Vertical coordinate (metres) is genuinely metric, so `znode` is appropriate.
    depths = [Float64(Oceananigans.Grids.znode(k, cpu_g, Oceananigans.Grids.Center()))
              for k in 1:nz]

    return (lons = lons, lats = lats, depths = depths)
end

"""
    get_strata_buffered_envelope(
        polygons::AbstractVector{<:NamedTuple};
        buffer_km::Real = 100.0
    ) -> NamedTuple

Compute the collective bounding envelope across a set of administrative stratum polygons
(e.g., loaded CFAs) expanded by a user-defined buffer distance (default 100.0 km).
"""
function get_strata_buffered_envelope(
    polygons::AbstractVector{<:NamedTuple};
    buffer_km::Real = 100.0
)
    if isempty(polygons)
        error("Cannot compute buffered envelope for empty polygon list.")
    end

    all_lons = Float64[]
    all_lats = Float64[]

    for poly in polygons
        append!(all_lons, poly.lons)
        append!(all_lats, poly.lats)
    end

    raw_lon = extrema(all_lons)
    raw_lat = extrema(all_lats)

    buf_lon, buf_lat = expand_domain_with_buffer(raw_lon, raw_lat, buffer_km = buffer_km)
    dlon, dlat = buffer_distance_to_degrees(buffer_km, 0.5 * (raw_lat[1] + raw_lat[2]))

    return (
        lon_range = buf_lon,
        lat_range = buf_lat,
        raw_lon_range = raw_lon,
        raw_lat_range = raw_lat,
        buffer_km = Float64(buffer_km),
        dlon = dlon,
        dlat = dlat
    )
end

"""
    point_in_polygon(x::Real, y::Real, poly_lons::AbstractVector, poly_lats::AbstractVector) -> Bool

Determine whether point (x, y) is inside polygon (poly_lons, poly_lats).

A re-export of `GeoData.point_in_polygon`, kept because the geometry is a pure data-layer
concern and a caller that has rings should not have to import this package to test them.
"""
const point_in_polygon = GeoData.point_in_polygon

"""
    is_point_on_land(lon::Real, lat::Real; coastline = nothing) -> Bool

Determine whether geographic point (lon, lat) falls on land.

A re-export of `GeoData.is_point_on_land`, which resolves the coastline file when the caller
does not name one. The default location is that function's to decide, not this package's.

There is no bundled regional fallback here. This package is an application layer for one
region, and the rings for that region live in `GeoDataSources.SCOTIAN_SHELF_COASTLINE`; a
caller that wants a regional default passes them, which is a decision it can see.
"""
const is_point_on_land = GeoData.is_point_on_land

"""
    load_coastline_polygons(; coastline_path = nothing) -> Vector{NamedTuple}

Load coastline rings from `coastline_path`, or from the default location when none is named.

A re-export of `GeoData.load_coastline_polygons`.
"""
const load_coastline_polygons = GeoData.load_coastline_polygons

