"""
    grid_bathymetry.jl

Grid construction and immersed boundary setup for shelf hydrodynamic modeling.

ParticleTracking-specific functions requiring Oceananigans grid types.
All generic geospatial operations (bathymetry I/O, regridding, coastline, marine cell
extraction, smoothing, buffering) are in GeoData.core.bathy. 
"""

using Oceananigans
using Oceananigans.Grids: Face, Center, znode
using Oceananigans.Architectures: architecture, on_architecture, CPU, GPU
using Oceananigans.ImmersedBoundaries: ImmersedBoundaryGrid, GridFittedBottom
using GeoData
using GeoDataSources
using Random

"""
    build_shelf_grid(;
        architecture = :cpu,
        lon_range=(-68.0, -57.0),
        lat_range=(42.0, 47.0),
        z_range=(-1000.0, 0.0),
        grid_size=(50, 50, 10),
        topology=(Bounded, Bounded, Bounded),
        fallback_to_cpu::Bool = false
    )

Construct a `LatitudeLongitudeGrid` for regional shelf hydrodynamics with specified architecture.
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
    fallback_to_cpu::Bool = false
)
    if lon_range[1] >= lon_range[2]
        error("Invalid longitude range: $(lon_range). lon_min must be < lon_max.")
    end
    if lat_range[1] >= lat_range[2]
        error("Invalid latitude range: $(lat_range). lat_min must be < lat_max.")
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
    grid::LatitudeLongitudeGrid,
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
    return ([Float64(grid.λᶜᵃᵃ[i]) for i in 1:grid.Nx],
            [Float64(grid.φᵃᶜᵃ[j]) for j in 1:grid.Ny])
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

    # Regridding 2D bilinear onto the target coordinates. The source axes are handled by
    # `GeoData.regrid_2d_field`, which sorts them internally so a latitude axis stored
    # south-to-north is mirrored rather than misread.
    regridded_topo = regrid_2d_field(lon, lat, elevation, target_lons, target_lats)

    if mask_bay_of_fundy
        n_masked = 0
        for j in 1:ny, i in 1:nx
            if is_in_bay_of_fundy(target_lons[i], target_lats[j])
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
            if is_in_bay_of_fundy(target_lons[i], target_lats[j])
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
    grid::LatitudeLongitudeGrid;
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

    # The domain comes from the grid: interior faces 1 : Nx+1, which is exactly the extent
    # `build_shelf_grid` was asked for. It is requested as a box rather than as the cell
    # centres so the source covers the cells it is about to be sampled onto.
    lon_range = (Float64(base_g.λᶠᵃᵃ[1]), Float64(base_g.λᶠᵃᵃ[base_g.Nx + 1]))
    lat_range = (Float64(base_g.φᵃᶠᵃ[1]), Float64(base_g.φᵃᶠᵃ[base_g.Ny + 1]))

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

    if !(cpu_g isa LatitudeLongitudeGrid)
        error(
            "extract_grid_coordinates requires a LatitudeLongitudeGrid (or an " *
            "ImmersedBoundaryGrid wrapping one) to report geographic coordinates in " *
            "degrees; received $(nameof(typeof(cpu_g))). Metric-distance grids must be " *
            "converted to geographic coordinates before extraction."
        )
    end

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

