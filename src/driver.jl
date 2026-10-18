# ==============================================================================
#  driver.jl -- the ocean half's run segments, as library code
# ==============================================================================
#
#  The segment functions `ParticleTrackingRun.jl` held as a script, moved here so
#  `GeoOceanModelRun.jl` can dispatch to them and `Pkg.test` can reach them: a
#  script that defines the behaviour is a script nobody can test.
#
#  Included LAST in `GeoOceanModel.jl`, after the model builders and the store
#  writer, so every name these bodies reach for (`build_shelf_grid`,
#  `build_hydrodynamic_model`, the tidal helpers, `attach_output_callback!`, ...)
#  is already defined when a call happens.
#
#  What is NOT here: the lar**val** segments and the larval results/analytics
#  API. `run_production_pipeline` reaches `run_segment_tracking` and its
#  siblings on the path past the hand-over, and under the split each is a stub
#  that names where it really lives (`GeoLarvalTracking`), so the coupling is at
#  the call site rather than a `MethodError` later.
#
#  NO `using` statements. The module imports Oceananigans, `prettytime`, the
#  manifest and `GeoDataSources`; a file included into a package relies on the
#  module scope. `data_source`/`fetch_input` are written `GeoDataSources.`-qualified
#  because they are not exported from `GeoDataSources`.
# ==============================================================================

"""
    resolve_hydro_model_path(opts::HydrodynamicOptions, default_filename::String) -> Tuple{String, String}

Resolve this run's model-file base path and name -- the Oceananigans restart checkpoint,
not a deliverable. The fields a consumer reads live in the GeoData store at
`dirname(...)/hydrodynamics.zarr`, written by the output callback; this file is what
`pickup` resumes from, so `dirname` is what anchors the store and the checkpoints.

If `opts.hydro_model_file` is specified it is used directly (or resolved relative to
`opts.output_dir` when a bare name is given); if empty it is
`joinpath(opts.output_dir, default_filename)`. Its directory is the run's output directory,
where the GeoData store is written, and the base names the JLD2 restart checkpoint; the store
itself is `output_zarr_path(dirname(...))`.

# Inputs
- `opts::HydrodynamicOptions`: Configuration parameters.
- `default_filename::String`: Default fallback name when none is specified.

# Outputs
- `Tuple{String, String}`: `(full_path, file_basename)`
"""
function resolve_hydro_model_path(opts::HydrodynamicOptions, default_filename::String)
    if isempty(opts.hydro_model_file)
        full_path = joinpath(opts.output_dir, default_filename)
        return (full_path, default_filename)
    else
        raw = opts.hydro_model_file
        full_path = isabspath(raw) || dirname(raw) != "" ? raw : joinpath(opts.output_dir, raw)
        return (full_path, basename(raw))
    end
end

"""
    _woa_store_locations(opts::HydrodynamicOptions) -> Vector{String}

Where WOA23 already lives in the GeoData store. Each variable is ~900 MB, so a store copy is
reused, never re-downloaded: this run's input dir first, then the package's own `inputs/`,
then every `work/*/inputs` and `outputs/*/inputs` a prior run left, then the stores of the
other halves. `woa23_in_store` picks the first copy that exists.
"""
function _woa_store_locations(opts::HydrodynamicOptions)
    root = dirname(@__DIR__)   # the GeoOceanModel package directory
    # The central GeoData store first (projects/lakehouse/GeoData/woa23), then the run and
    # its siblings, then the umbrella's. Reading the central copy means a ~1.8 GB
    # climatology is fetched once per machine, not once per run.
    dirs = AbstractString[GeoDataSources.store_path("woa23"), opts.input_dir, joinpath(root, "inputs")]
    for scan in (joinpath(root, "work"), joinpath(root, "outputs"),
                 joinpath(root, "..", "ParticleTracking", "outputs"))
        isdir(scan) && for d in readdir(scan; join = true)
            push!(dirs, joinpath(d, "inputs"))
        end
    end
    push!(dirs, joinpath(root, "..", "ParticleTracking", "inputs"))
    push!(dirs, joinpath(root, "..", "ParticleTracking", "work", "snowcrab", "inputs"))
    return String[dirs...]
end

"""
    _woa_in_store(opts, prefix, month_str) -> String

Path to the store's WOA23 file for `prefix` and month, or "" if the store has none. `.nc` in
preference to `.zarr`, at the first directory that has either.
"""
function _woa_in_store(opts::HydrodynamicOptions, prefix::AbstractString, month_str::AbstractString)
    for d in _woa_store_locations(opts)
        nc = joinpath(d, "woa23_$(prefix)_$(month_str)_0.25deg.nc")
        isfile(nc) && return nc
        zr = joinpath(d, "woa23_$(prefix)_$(month_str)_0.25deg.zarr")
        isdir(zr) && return zr
    end
    return ""
end

"""
    _store_sim_info(store_path; stop_time) -> (; exists, is_complete, last_time, n_timesteps)

Resume/reuse state of a run's GeoData store, read from the store's own `time` coordinate. This
is a `geoload` read of the same axis the writer filled, so the record of progress is the store
itself and cannot drift from what it recorded -- which a `time.json` sidecar could, and did:
a sidecar written per interval and a coordinate written per interval are two writes of the same
number.

`last_time` is the newest, `n_timesteps` the count, `is_complete` whether it reached
`stop_time`. A missing store, or one with no slices, is `exists = false`. This replaced
inspecting a JLD2 archive, which the run no longer writes: the store is the deliverable, so it
is also the record of progress.
"""
function _store_sim_info(store_path::AbstractString; stop_time::Real)
    stored = if isdir(store_path)
        try
            ds = GeoData.geoload(store_path; backend = :zarr)
            isempty(ds.variables) ? Float64[] :
                collect(Float64, ds.coords[:time].data)
        catch
            Float64[]
        end
    else
        Float64[]
    end
    n = length(stored)
    last = n > 0 ? maximum(stored) : 0.0
    return (exists = n > 0,
            is_complete = n > 0 && last >= stop_time - 1e-3,
            last_time = last,
            n_timesteps = n)
end

"""
    store_interval_count(store_path) -> Int

How many intervals a run's store holds, from its own `time` coordinate. 0 for a store that
does not exist or has no slices, so a caller resuming a fresh directory starts at 1.
"""
function store_interval_count(store_path::AbstractString)
    isdir(store_path) || return 0
    try
        ds = GeoData.geoload(store_path; backend = :zarr)
        return isempty(ds.variables) ? 0 : length(ds.coords[:time].data)
    catch
        return 0
    end
end

"""
    _run_bathymetry_source(opts::HydrodynamicOptions) -> Symbol

Which bathymetry product this run uses.

`opts.bathy_source` is the name the operator wrote in the TOML; the domain is
`opts.domain_lon`/`opts.domain_lat`. The choice itself is `GeoDataSources`'s: a named
source stands, and an unnamed one is resolved by coverage -- the CHS 15-arc-second
Northwest Atlantic soundings where the domain lies inside them, the global ETOPO 2022 model
everywhere else. Every segment asks through here, so they cannot disagree about the sea
floor.
"""
function _run_bathymetry_source(opts::HydrodynamicOptions)
    return GeoDataSources.preferred_bathymetry_source(;
        source = opts.bathy_source,
        lon_range = opts.domain_lon,
        lat_range = opts.domain_lat
    )
end

"""
    _run_bathymetry_path(opts::HydrodynamicOptions) -> String

Where this run's bathymetry artefact lives: `inputs/bathymetry_active.nc` when there is no
store yet, otherwise whichever of the `.nc` and `.zarr` spellings is already on disk.
"""
function _run_bathymetry_path(opts::HydrodynamicOptions)
    bathy_nc = joinpath(opts.input_dir, "bathymetry_active.nc")
    bathy_zarr = joinpath(opts.input_dir, "bathymetry_active.zarr")
    return isfile(bathy_nc) ? bathy_nc : (isdir(bathy_zarr) ? bathy_zarr : bathy_nc)
end

"""
    _run_bathymetry(; opts::HydrodynamicOptions, force::Bool = false) -> NamedTuple

Resolve this run's bathymetry to its source, its durable artefact, and its arrays.

One call site per segment rather than one per use, so the grid, the larval spawn filter, the
tracking bathymetry function and the figure backgrounds are all one sea floor. Format and
file location are `GeoDataSources`'s decision; this only states the source, the range, and
where the artefact is kept.
"""
function _run_bathymetry(; opts::HydrodynamicOptions, force::Bool = false)
    source = _run_bathymetry_source(opts)
    output_path = _run_bathymetry_path(opts)
    
    # Fast path: If the bathymetry dataset has already been fetched and cached,
    # load it directly via GeoData.geoload without invoking GeoDataSources.
    if !force && (isfile(output_path) || isdir(output_path))
        ds = GeoData.geoload(output_path)
        elev_var = haskey(ds.variables, "elevation") ? "elevation" :
                   first(GeoData.variables_like(ds, ["elevation", "altitude", "z", "topo"]))
        elev_ga = ds.variables[elev_var]
        perm = GeoData.dim_permutation(elev_ga.dims, :lon, :lat)
        lon, _ = GeoData.axis(ds, :lon)
        lat, _ = GeoData.axis(ds, :lat)
        raw = Array{Float64}(elev_ga.data)
        elevation = ndims(raw) == 1 ? reshape(raw, :, 1) : permutedims(raw, perm)
        return (source = source, output_path = output_path, elevation = elevation,
                lon = lon, lat = lat)
    end

    bathy = GeoDataSources.bathymetry_field(;
        source = source,
        lon_range = opts.domain_lon,
        lat_range = opts.domain_lat,
        output_path = output_path,
        force = force
    )
    println("Bathymetry source: $(source) (", GeoDataSources.BATHYMETRY_SOURCES[source], ")")
    return (source = source, output_path = output_path, elevation = bathy.elevation,
            lon = bathy.lon, lat = bathy.lat)
end

# ─────────────────────────────────────────────────────────────────────────────
# Segment 1: Data Ingestion & Benchmark Generation
# ─────────────────────────────────────────────────────────────────────────────

"""
    run_segment_data(; opts::HydrodynamicOptions) -> NamedTuple

Fetch real-world bathymetry and surface winds from NOAA ERDDAP or generate synthetic
analogs matching the model bounding box. Computes kinematic wind stress components.

# Inputs
- `opts::HydrodynamicOptions`: Workflow options container.

# Outputs
- `NamedTuple`: `(bathy_file, wind_file, tau_x, tau_y, bathy_info)`
"""
function run_segment_data(; opts::HydrodynamicOptions = HydrodynamicOptions())
    println("\n=================================================================")
    println(" [Segment 1/8] Environmental Data Ingestion & Drag Processing")
    println("=================================================================")
    mkpath(opts.input_dir)

    # The coastline is a fetched input like any other, and is listed in the provenance
    # registry, so it is acquired rather than seeded from a file in this repository. A stale
    # repository copy would let two scenarios mask against different land.
    # `.parquet`, not `.dat`: the data layer's `load_coastline_polygons` reads GeoParquet and
    # rejects any other suffix. The `.dat` name it used to carry was Parquet bytes under a
    # mislabel, which let the fetch succeed and the later read fail.
    coastline_file = joinpath(opts.input_dir, "coastline.parquet")
    if !isfile(coastline_file)
        src = opts.coastline_source
        println("Retrieving coastline ($(src))...")
        fetch_natural_earth_coastline(
            lon_range = opts.domain_lon,
            lat_range = opts.domain_lat,
            resolution = replace(src, "natural_earth_" => ""),
            output_path = coastline_file
        )
    end

    bathy_nc   = joinpath(opts.input_dir, "bathymetry_active.nc")
    bathy_zarr = joinpath(opts.input_dir, "bathymetry_active.zarr")
    wind_nc    = joinpath(opts.input_dir, "wind_active.nc")
    wind_zarr  = joinpath(opts.input_dir, "wind_active.zarr")

    wind_file  = isfile(wind_nc)  ? wind_nc  : (isdir(wind_zarr) ? wind_zarr : wind_nc)

    # Data acquisition always reads observed products. There is no "synthetic mode": the product
    # is named directly by `[data] bathy_source` / `wind_source` in the TOML, and a failed
    # download aborts the run rather than substituting an analytic field, because a run must
    # never claim a dataset-forced state it did not read.
    #
    # The source is passed through `preferred_bathymetry_source` rather than to a fetcher
    # directly, so a `bathy_source` that names no single product ("auto", or absent) is
    # resolved by coverage instead of falling back to one by default. Which file that product
    # is stored in, and in what format, is decided inside `GeoDataSources`; a cached artefact
    # is reused rather than re-fetched, which is what keeps this segment and the grid segment
    # on the same sea floor.
    bathy = _run_bathymetry(; opts = opts)
    bathy_file = bathy.output_path

    if !isfile(wind_file) && !isdir(wind_file)
        println("Retrieving surface winds ($(opts.wind_source)) for $(opts.wind_time_iso)...")
        # One named source, no mirror swap, no synthetic substitute. `fetch_surface_winds`
        # stops with a message listing the sources this project can actually retrieve.
        wind_res = fetch_surface_winds(opts.wind_source; lon_range = opts.domain_lon,
                                       lat_range = opts.domain_lat, time_iso = opts.wind_time_iso,
                                       output_path = wind_file)
        if wind_res isa AbstractString && (isfile(wind_res) || isdir(wind_res))
            wind_file = wind_res
        end
    else
        println("Using existing real surface winds file/store: $wind_file")
    end

    if opts.enable_tides
        src_str = lowercase(strip(String(opts.tides_source)))
        if !(src_str in ("analytic", "synthetic", "none", "off", "false"))
            # `resolved_cache_path`, not the raw `.cache` template: the template names
            # `<store>`, and a test that substitutes only `<output_dir>` looks at a
            # `<store>/tpxo9/...` directory that never exists, so it re-fetches every run.
            tides_file = GeoDataSources.resolved_cache_path(:tides, opts.output_dir)
            if !isfile(tides_file)
                println("Retrieving tidal solution ($(opts.tides_source))...")
                tides_res = GeoDataSources.fetch_input(:tides, opts.output_dir)
                tides_file = tides_res isa AbstractString ? String(tides_res) : tides_file
            end
            println("Tidal solution: $tides_file")
        end
    end

    # The stress is read back out of the file that was actually fetched. Deriving it from
    # hard-coded reference winds would make the fetch decorative: the numbers driving the
    # model would not come from the ingested product at all.
    w = read_wind_stress(wind_file)
    tau_x, tau_y = w.tau_x, w.tau_y
    bathy_info = (file = bathy_file, exists = true)
    println("Large & Pond (1981) kinematic wind stress, read from $(basename(wind_file)):")
    println("  Domain-mean 10m wind speed: $(round(w.speed10_rms, digits=2)) m/s")
    println("  Domain-mean stress:   tau_x = $(round(tau_x, digits=8)), " *
            "tau_y = $(round(tau_y, digits=8)) m^2/s^2")
    println("  Peak stress magnitude: $(round(w.tau_max, digits=8)) m^2/s^2")
    return (bathy_file = bathy_file, wind_file = wind_file, tau_x = tau_x, tau_y = tau_y, bathy_info = bathy_info)
end



# ─────────────────────────────────────────────────────────────────────────────
# Segment 2: Computational Grid & Immersed Seafloor Boundary
# ─────────────────────────────────────────────────────────────────────────────

"""
    run_segment_grid(; opts::HydrodynamicOptions) -> NamedTuple

Construct the base spherical `LatitudeLongitudeGrid` and wrap it with an
`ImmersedBoundaryGrid` using 2D bilinear interpolation from bathymetry data.

# Inputs
- `opts::HydrodynamicOptions`: Workflow options.

# Outputs
- `NamedTuple`: `(base_grid, immersed_grid)`

The bathymetry is requested through `_run_bathymetry_source`/`_run_bathymetry_path`, so this
segment resolves the same source, over the same domain, to the same artefact as segment 1.
"""
function run_segment_grid(; opts::HydrodynamicOptions = HydrodynamicOptions())
    println("\n=================================================================")
    println(" [Segment 2/8] Spherical Grid & Immersed Boundary Construction")
    println("=================================================================")

    arch_label = opts.use_gpu ? "GPU (CUDA)" : "CPU"
    println("Building base spherical grid on $(arch_label): $(opts.grid_size) cells...")

    z_faces = if !isempty(opts.vertical_depths)
        # `[grid] depths` is written positive-downward from the surface (0, 10, 20, ...);
        # Oceananigans wants z faces negative and ascending, so negate and reverse. The config
        # layer has already checked monotonicity, the 0 m surface, and the nz+1 face count.
        f = sort(-Float64.(opts.vertical_depths))
        # Precedence: an explicit depth list DEFINES the column, so nz must agree with it
        # rather than being applied on top. Two numbers for one axis is a configuration
        # disagreement, and picking either silently is how a run ends up with a column nobody
        # asked for.
        nz_from_depths = length(f) - 1
        nz_from_depths != opts.grid_size[3] && error(
            "[grid] nz = $(opts.grid_size[3]) conflicts with [grid] depths, which define " *
            "$(length(f)) faces and therefore Nz = $(nz_from_depths). Set nz = $(nz_from_depths), " *
            "or remove the depth list to let nz alone define the column. Refusing to pick one.")
        println("Using $(length(f)) vertical faces from [grid] depths " *
                "(shallowest dz=$(round(f[end] - f[end-1], digits=1))m, " *
                "deepest dz=$(round(f[2] - f[1], digits=1))m)...")
        f
    elseif !isempty(opts.vertical_grid_file)
        # A named `vertical_grid_file` is the operator's own file. `configuration_to_options`
        # already resolved it to a copy inside this run's input directory and refused to
        # proceed if it was missing, so reaching here means the file is present and staged.
        Lz = abs(opts.domain_z[1] - opts.domain_z[2])
        println("Using vertical layer faces from $(basename(opts.vertical_grid_file)) (Lz=$(Lz)m)...")
        stretched_tanh_z_faces(opts.grid_size[3], Lz; csv_path = opts.vertical_grid_file, strict = true)
    elseif opts.vertical_stretching_mode === :csv
        error(
            "[grid] vertical_stretching_mode = \"csv\" but neither vertical_grid_file nor " *
            "depths is set. Name a file, give a depth list, or use a mode that derives faces " *
            "from the depth range."
        )
    elseif opts.vertical_stretching_mode in (:tanh, :stretched)
        # Two-segment grid: a surface-refined upper segment that actually resolves the shallow
        # active layer, plus a coarse lower segment so the deep basin is still represented.
        # A single tanh over 0 to -5000 m puts its *coarsest* cell at the surface, which left the
        # shallowest cell centre at -258 m and no cell centre at all in the 0-50 m nursery band.
        Lz = abs(opts.domain_z[1] - opts.domain_z[2])
        nz_tot = opts.grid_size[3]
        # The segment split is a second, independent way of saying how many layers there are,
        # so it has to agree with nz rather than quietly dividing it differently.
        if opts.nz_above > 0
            nz_above = opts.nz_above
            nz_below_implied = nz_tot - nz_above
            nz_below_implied < 1 && error(
                "[grid] nz_above = $nz_above leaves only $nz_below_implied layers below it " *
                "given nz = $nz_tot. Need nz_above < nz.")
        else
            nz_above = round(Int, 0.6 * nz_tot)
        end
        nz_below = nz_tot - nz_above
        nz_below < 1 && error(
            "vertical grid split leaves nz_below = $nz_below; need nz_above < nz = $nz_tot.")
        break_depth = abs(opts.vertical_break_depth)
        z_break = break_depth > 0.0 ? -break_depth : -400.0
        println("Applying two-segment vertical coordinates (break=$(z_break)m, " *
                "nz_above=$nz_above, nz_below=$nz_below, Lz=$(Lz)m)...")
        two_segment_z_faces(z_break = z_break, z_min = -Lz,
                            nz_above = nz_above, nz_below = nz_below,
                            scaling_above = 2.0, scaling_below = 0.0)
    else
        nothing
    end

    base_grid = build_shelf_grid(
        architecture = opts.use_gpu ? :gpu : :cpu,
        lon_range = opts.domain_lon,
        lat_range = opts.domain_lat,
        z_range = opts.domain_z,
        z_faces = z_faces,
        grid_size = opts.grid_size,
        fallback_to_cpu = opts.fallback_to_cpu,
        orientation = opts.grid_orientation,
        rotation_azimuth = opts.rotation_azimuth,
        center_coords = (opts.center_lon, opts.center_lat),
        along_shelf_extent_km = opts.along_shelf_extent_km,
        cross_shelf_extent_km = opts.cross_shelf_extent_km
    )

    println("Constructing immersed boundary by 2D bilinear regridding...")
    immersed_grid = build_immersed_grid_from_real_data(
        base_grid;
        source = _run_bathymetry_source(opts),
        output_path = _run_bathymetry_path(opts),
        min_water_depth = abs(opts.inshore_depth),
        max_water_depth = 1500.0,
        mask_bay_of_fundy = opts.mask_bay_of_fundy
    )

    println("Grid summary:")
    if opts.grid_orientation === :rotated
        println("  Orientation: Rotated orthogonal grid at $(opts.rotation_azimuth)° azimuth")
        println("  Center:      ($(opts.center_lon)°E, $(opts.center_lat)°N)")
        println("  Extent:      $(opts.along_shelf_extent_km) km along-shelf × $(opts.cross_shelf_extent_km) km cross-shelf")
        println("  Horizontal:  Nx=$(opts.grid_size[1]) × Ny=$(opts.grid_size[2]) ($(opts.grid_size[1]*opts.grid_size[2]) total cells)")
    else
        println("  Orientation: Standard geographic grid")
        println("  Longitude:   $(opts.domain_lon[1])°E to $(opts.domain_lon[2])°E (Nx=$(opts.grid_size[1]))")
        println("  Latitude:    $(opts.domain_lat[1])°N to $(opts.domain_lat[2])°N (Ny=$(opts.grid_size[2]))")
    end
    println("  Depth:     $(opts.domain_z[1]) m to $(opts.domain_z[2]) m (Nz=$(opts.grid_size[3]))")
    if !isnothing(z_faces)
        src = isempty(opts.vertical_grid_file) ? "stretched" : basename(opts.vertical_grid_file)
        println("  Vertical:  $src (surface dz=$(round(z_faces[end]-z_faces[end-1], digits=1))m, bed dz=$(round(z_faces[2]-z_faces[1], digits=1))m)")
    else
        println("  Vertical:  Oceananigans default (nz=$(opts.grid_size[3]) evenly spaced faces)")
    end

    return (base_grid = base_grid, immersed_grid = immersed_grid)
end

# ─────────────────────────────────────────────────────────────────────────────
# Segment 3: Hydrodynamic Model Instantiation & Stratification
# ─────────────────────────────────────────────────────────────────────────────

"""
    run_segment_model(;
        opts::HydrodynamicOptions,
        immersed_grid=nothing,
        tau_x::Real=0.0001,
        tau_y::Real=0.0
    ) -> NamedTuple

Instantiate the Oceananigans `HydrostaticFreeSurfaceModel` with Coriolis rotation,
buoyancy, surface wind boundary conditions, and semi-diurnal tidal body forcing (M2).

# Inputs
- `opts::HydrodynamicOptions`: Workflow options.
- `immersed_grid`: Pre-constructed `ImmersedBoundaryGrid`.
- `tau_x, tau_y`: Surface kinematic momentum fluxes.

# Outputs
- `NamedTuple`: `(model, tidal_forcing, simpson_hunter_bank, simpson_hunter_shelf)`
"""
function run_segment_model(;
    opts::HydrodynamicOptions = HydrodynamicOptions(),
    immersed_grid = nothing,
    tau_x::Real = 1e-4,
    tau_y::Real = 0.0,
    wind_file::AbstractString = ""
)
    target_grid = if isnothing(immersed_grid)
        grid_res = run_segment_grid(opts = opts)
        grid_res.immersed_grid
    else
        immersed_grid
    end

    println("\n=================================================================")
    println(" [Segment 3/8] Hydrodynamic Model & Tidal Forcing Setup")
    println("=================================================================")

    # Astronomical tides, read from the global TPXO9 solution rather than imposed as a uniform
    # oscillation. The previous form applied `tidal_u_amp` (0.25 m/s) uniformly in x, y AND z;
    # on this shelf that configuration diverged to 8-13 m/s within six simulated hours, and
    # disabling tides returned the same setup to 0.4 m/s. The amplitudes in [tides] are
    # therefore no longer used to scale a body force: they describe a uniform tide that is not
    # what the shelf actually experiences, and the real field varies from 0.006 m/s on the
    # southern boundary to 0.713 m/s in the Bay of Fundy.
    #
    # The atlas is read from this run's containerised inputs directory; see `fetch_input` and
    # the :tides entry in DATA_SOURCES for how to obtain it and what the alternatives are.
    tidal_forcing = if opts.enable_tides
        src_str = lowercase(strip(String(opts.tides_source)))
        if !(src_str in ("analytic", "synthetic", "none", "off", "false"))
            tides_file = GeoDataSources.resolved_cache_path(:tides, opts.output_dir)
            if !isfile(tides_file)
                println("Tidal solution missing at $(tides_file); retrieving via fetch_input(:tides)...")
                tides_res = GeoDataSources.fetch_input(:tides, opts.output_dir)
                tides_file = tides_res isa AbstractString ? String(tides_res) : tides_file
            end
            if !isfile(tides_file)
                error(
                    "[hydrodynamics] enable_tides is true and tides_source is '$(opts.tides_source)', " *
                    "but the tidal solution could not be retrieved:\n" *
                    "  $(tides_file)\n" *
                    "Obtain it with `fetch_input(:tides, \"$(opts.output_dir)\")` or `--fetch-tides`, " *
                    "or set enable_tides = false (or pass `--no-tides`). There is deliberately no fallback to a " *
                    "uniform body-force approximation: that was what made this configuration diverge, " *
                    "and a run that silently used a different tidal solution than its provenance " *
                    "file records is the failure this project is trying to prevent.")
            end
            # Honour `[tides] constituents`. An empty list in a config that disables tides is not an
            # error; it simply means no constituents were requested, and the branch that reads
            # harmonics is not reached.
            cons = opts.tidal_constituents
            println("Reading tidal harmonics from $(basename(tides_file)) " *
                    "(constituents: $(join(cons, ", ")))...")
            h = read_tidal_velocity_harmonics(
                tides_file;
                constituents = cons,
                lon_range = opts.domain_lon,
                lat_range = opts.domain_lat,
            )
            # The forcing is reduced to a small `isbits` coefficient grid BEFORE it reaches the
            # model. A closure that searches the atlas arrays at call time is not GPU-safe and
            # fails to compile in gpu_compute_hydrostatic_free_surface_Gu!.
            # Grid size is bounded by GPU kernel *parameter* memory: the coefficient tuple is
            # passed by value into every invocation, so 16x12x2x4 doubles (12 KB) is rejected
            # with "Kernel invocation uses too much parameter memory". 6x4 gives 192 doubles
            # (~1.5 KB), which fits comfortably. The tidal field varies on scales of degrees and
            # the domain is ~18 deg across, so 3 deg spacing still resolves the 0.007 m/s southern
            # boundary from the 0.45 m/s Fundy -- the contrast that caused the original blowup.
            coeff = TidalCoefficientGrid(h, (opts.domain_lon[1], opts.domain_lon[2]),
                                         (opts.domain_lat[1], opts.domain_lat[2]);
                                         nx = 6, ny = 4)
            f = tidal_velocity_coefficients(coeff)
            edge = boundary_tidal_forcing(h; edges = (:east, :south, :west), sample_stride = 4)
            amps = vcat(edge[:east].uamp[1, :], edge[:south].uamp[1, :], edge[:west].uamp[1, :])
            println("  M2 boundary speed amplitude: ", round(minimum(amps), digits = 4), " - ",
                    round(maximum(amps), digits = 4), " m/s")
            f
        else
            nothing
        end
    else
        nothing
    end

    coriolis_lat = 0.5 * (opts.domain_lat[1] + opts.domain_lat[2])

"""
    fetch_boundary_tracers(source::Symbol, opts) -> NamedTuple

Thin wrapper binding a configured source to the bounding box and month from the TOML.

Everything else -- acquisition, caching, which file holds which variable, what units the
temperature is in, and whether the file is actually usable data -- lives in
`fetch_boundary_hydrography`. The driver deliberately does not know how any source is
packaged, because that knowledge is what made two WOA23 readers, a truncated boundary file
and a units error possible in the first place.

The bounding box is the `[boundaries] embedding_*` region rather than the study domain. It
has to be larger: the model needs the water just *outside* its own edges, and a subset cut
to the study domain would put the interpolation's edge exactly on the sponge.
"""
function fetch_boundary_tracers(source::Symbol, opts)
    return fetch_boundary_hydrography(source;
        lon_range = opts.embedding_lon, lat_range = opts.embedding_lat,
        input_dir = opts.input_dir,
        month = opts.hydrography_month)
end

    # Configure open boundary conditions (lateral sponge) if requested.
    #
    # Two independent things are decided here and they were previously conflated into one
    # boolean:
    #
    #   * whether the sponge band is active at all, and
    #   * whether observed boundary temperature and salinity are available to relax toward.
    #
    # The second is what the source name actually selects. Previously *any* non-GLORYS value
    # fell through to `nothing`, which meant a keyless config asking for observed boundary
    # hydrography silently got an unconstrained edge -- the request was accepted and ignored.
    # That is the worst possible failure for this input, because an unconstrained edge still
    # produces plausible-looking output.
    obc_src = opts.ocean_boundary_source
    boundary_tracer_data = nothing
    if obc_src in (:woa23, :glorys12v1, :glorys_climatology, :hycom)
        boundary_tracer_data = fetch_boundary_tracers(obc_src, opts)
        println("Boundary hydrography: observed temperature and salinity from $(obc_src).")
        println("  This constrains what the incoming water IS, not how fast it is GOING -- " *
                "the inflow speed still comes from [boundaries] u_inflow/v_inflow.")
    elseif obc_src == :synthetic
        println("Boundary hydrography: none. The open edges are unconstrained.")
    else
        error(
            "Unknown [boundaries] ocean_boundary_source \"$(obc_src)\". Choose \"synthetic\" " *
            "(unconstrained edges), \"woa23\" (keyless climatology, no currents), " *
            "\"glorys12v1\" (reanalysis with currents; needs a Copernicus Marine account), " *
            "or \"hycom\" (keyless with currents, but its server currently refuses data " *
            "requests). Nothing was substituted."
        )
    end

    obc_craft = if obc_src in (:glorys12v1, :glorys_climatology, :woa23, :hycom)
        println("Enabling lateral boundary relaxation...")
        true
    else
        nothing
    end

    # Atmospheric surface forcing, computed from the ingested ERA5 surface state rather than
    # carried as a constant. `surface_heat_flux` in the config is what decides whether the
    # flux is applied; the stress is always applied, since without it there is no momentum
    # input at all.
    #
    # `wind_file` is a parameter of this function now, not a free variable left over from the
    # segment-1 scope. It was referenced here but never in scope, so `--all` died with
    # "UndefVarError: wind_file not defined in Main" before ever reaching the flux.
    wind_path = isempty(wind_file) ? joinpath(opts.input_dir, "wind_active.nc") : wind_file
    atmo_craft = if opts.atmospheric_source in (:era5, :era5_climatology, :mhw)
        if !isfile(wind_path)
            error(
                "Atmospheric forcing is requested but the surface-wind file $(wind_path) is " *
                "missing. Run the data-ingestion segment, or disable atmospheric forcing.")
        end
        sx, sy, hx = build_bulk_surface_flux(wind_path)
        if isnothing(sx)
            error(
                "Atmospheric forcing is requested but $(basename(wind_path)) has no bulk " *
                "flux inputs. Re-fetch the wind file so the surface state is included.")
        end
        # Which heat flux to apply is decided by `[hydrodynamics] bulk_heat_flux`:
        #   true  -> the bulk flux `hx` computed from the ERA5 surface state
        #   false -> the constant `surface_heat_flux` (W m-2)
        #
        # This was previously decided by `surface_heat_flux != 0.0`, which was wrong twice
        # over. `surface_heat_flux` is a VALUE in W m-2, so treating it as a switch conflates
        # "no flux" with "a small real flux"; and the resulting `want_flux` was then used only
        # to pick a printed message, while `heat_flux = hx` was returned unconditionally. So
        # setting `surface_heat_flux = 0.0` did not disable the bulk flux -- it still drove
        # the run -- and `bulk_heat_flux`, which is parsed and stored in `HydrodynamicOptions` and
        # documented in the TOML, was never read at all. Both keys now do what they say.
        want_bulk = opts.bulk_heat_flux
        if want_bulk && isnothing(hx)
            error(
                "[hydrodynamics] bulk_heat_flux = true but the ingested surface state in " *
                "$(basename(wind_path)) is unavailable, so the bulk flux cannot be computed. " *
                "Re-fetch the wind file, or set bulk_heat_flux = false to apply the constant " *
                "`surface_heat_flux` instead."
            )
        end
        raw_flux = want_bulk ? hx : opts.surface_heat_flux
        applied_flux = if opts.sst_damping_coefficient > 0.0
            λ = opts.sst_damping_coefficient
            T_target = opts.sst_target
            if raw_flux isa Function
                HaneyRelaxedHeatFlux(raw_flux, λ, T_target)
            else
                HaneyRelaxedHeatFlux((x, y, t, T_surf) -> Float64(raw_flux), λ, T_target)
            end
        else
            raw_flux
        end
        if want_bulk
            if opts.sst_damping_coefficient > 0.0
                println("Atmospheric forcing: time-varying stress from ERA5 surface state, " *
                        "with bulk net heat flux + Haney SST relaxation (λ = $(opts.sst_damping_coefficient) W m-2 K-1).")
            else
                println("Atmospheric forcing: time-varying stress from ERA5 surface state, " *
                        "with a bulk net heat flux.")
            end
        elseif opts.surface_heat_flux != 0.0
            if opts.sst_damping_coefficient > 0.0
                println("Atmospheric forcing: time-varying stress from ERA5; constant heat flux " *
                        "of $(opts.surface_heat_flux) W m-2 + Haney SST relaxation (λ = $(opts.sst_damping_coefficient) W m-2 K-1).")
            else
                println("Atmospheric forcing: time-varying stress from ERA5; constant heat flux " *
                        "of $(opts.surface_heat_flux) W m-2 (bulk_heat_flux = false).")
            end
        else
            println("Atmospheric forcing: time-varying stress from ERA5; heat flux disabled.")
        end
        (stress_x = sx, stress_y = sy, heat_flux = applied_flux)
    else
        nothing
    end

    wind_x = isnothing(atmo_craft) ? tau_x : atmo_craft.stress_x
    wind_y = isnothing(atmo_craft) ? tau_y : atmo_craft.stress_y
    surface_heat_flux_val = if isnothing(atmo_craft) || isnothing(atmo_craft.heat_flux)
        0.0
    else
        atmo_craft.heat_flux
    end

    # SplitExplicitFreeSurface with explicit substepping (substeps = 45).
    # Provides an averaging filter for fast barotropic gravity waves over steep
    # bathymetry, suppressing 2-delta numerical resonance near open boundaries.
    free_surf = SplitExplicitFreeSurface(target_grid; substeps = 45)
    closure_choice = hasproperty(opts, :turbulence_closure) ?
        Symbol(lowercase(string(opts.turbulence_closure))) : :smagorinsky
    # Model assembly is delegated entirely to NumericalEarth.ocean_simulation, which builds
    # a `HydrostaticFreeSurfaceModel` with a spherical Coriolis (retaining the beta term),
    # a full TEOS-10 equation of state, and the split-explicit free-surface solver.
    println("Assembling model via NumericalEarth.ocean_simulation " *
            "(spherical Coriolis, TEOS-10 EOS, split-explicit free surface)...")
    model = build_hydrodynamic_model(
        target_grid,
        Δt = opts.sim_dt,
        coriolis_latitude = coriolis_lat,
        surface_wind_stress_x = wind_x,
        surface_wind_stress_y = wind_y,
        surface_heat_flux = surface_heat_flux_val,
        tidal_forcing = tidal_forcing,
        open_boundary_conditions = obc_craft,
        lateral_boundary_relaxation = opts.boundary_method != :none,
        sponge_width = opts.sponge_layer_width,
        sponge_tau = opts.sponge_timescale,
    # The prescribed tide is a VELOCITY and is relaxed toward, not forced as an acceleration.
    # See the note in `build_hydrodynamic_model`. The keyword is the model's
    # `tidal_relaxation_timescale` (the umbrella driver's older `tidal_tau` spelling was
    # renamed in `src/model/hydrodynamic_model.jl`).
    tidal_relaxation_timescale = opts.tidal_relaxation_timescale,
        u_inflow = opts.u_inflow,
        v_inflow = opts.v_inflow,
        boundary_tracers = boundary_tracer_data,
        closure = closure_choice in (:nemotke, :catke) ? :catke : nothing,
        ν = 1e-2,
        κ = 1e-2,
        tracers = (:T, :S),
        free_surface = free_surf,
        momentum_advection = WENO(),
        tracer_advection = WENO(),
        lon_range = opts.domain_lon,
        lat_range = opts.domain_lat,
        mask_bay_of_fundy = opts.mask_bay_of_fundy,
        active_boundaries = opts.active_boundaries,
        tidal_placement = :boundary
    )

    # Initial thermal and haline stratification.
    #
    # The source of the initial water column is `hydrography_source`, read from `[data]`.
    # It is deliberately independent of `ocean_boundary_source`: the lateral boundary
    # provider and the initial state are separate choices, and conflating them is what left
    # every "real data" run silently initialising from the constants `T = 10.0, S = 35.0`.
    hydro_src = opts.hydrography_source
    if hydro_src === :woa23

        month_str = lpad(string(opts.hydrography_month), 2, '0')

        # Resolve each variable from the store first (a store copy is reused, never
        # re-downloaded), then this run's own input dir; a download is the last resort.
        t_file = _woa_in_store(opts, "temperature", month_str)
        s_file = _woa_in_store(opts, "salinity", month_str)
        o_file = _woa_in_store(opts, "oxygen", month_str)
        if !isempty(t_file) || !isempty(s_file)
            println("WOA23 resolved $(isempty(t_file) ? "?" : basename(t_file)) / " *
                    "$(isempty(s_file) ? "?" : basename(s_file)) from the GeoData store.")
        end

        if isempty(t_file) || isempty(s_file)
            println("Fetching WOA23 climatology via GeoDataSources...")
            woa_res = fetch_open_woa_climatology(
                lon_range = opts.domain_lon,
                lat_range = opts.domain_lat,
                month = opts.hydrography_month,
                # WOA23 is a shared, machine-level product; it lands in the central GeoData
                # store (GeoDataSources.store_path("woa23")) rather than this run's inputs.
                output_dir = GeoDataSources.store_path("woa23")
            )
            # Re-resolve through the same resolver the check above used, not through this
            # run's input directory. The fetch writes to the store, so looking for the result
            # only in `inputs/` finds nothing on a machine whose store had no copy yet --
            # which is every fresh installation. The files are on disk, in the store, and the
            # run reports them as missing. `_woa_in_store` searches the store first and every
            # other place a copy may be, so the order is "wherever it is" rather than one
            # directory that this code path happens to write to.
            t_file = _woa_in_store(opts, "temperature", month_str)
            s_file = _woa_in_store(opts, "salinity", month_str)
            o_file = isempty(o_file) ? _woa_in_store(opts, "oxygen", month_str) : o_file
            if !isempty(t_file) || !isempty(s_file)
                println("WOA23 resolved from $(isdir(dirname(t_file)) ? dirname(t_file) : "the store") " *
                        "after acquisition: $(isempty(t_file) ? "?" : basename(t_file)) / " *
                        "$(isempty(s_file) ? "?" : basename(s_file))")
            end
        end

        if !isempty(t_file) && !isempty(s_file)
            println("Applying WOA23 climatological hydrography " *
                    "(month=$(opts.hydrography_month == 0 ? "annual" : opts.hydrography_month), source=$(basename(t_file)))...")
            woa = woa23_regridded_tracers(
                model.grid, t_file, s_file;
                oxygen_file = !isempty(o_file) ? o_file : nothing
            )
            # Separate `set!` calls: Oceananigans has no NamedTuple form of `set!`, and the
            # closures are only accepted as keyword values for the tracer fields.
            set!(model, T = woa.T, S = woa.S)
            set!(model, u = 0.0, v = 0.0)
            if haskey(woa, :O_2) && hasproperty(model.tracers, :O_2)
                set!(model, O_2 = woa.O_2)
            end
        else
            error(
                "hydrography_source = \"woa23\" but WOA23 files are missing from " *
                "$(opts.input_dir) (expected NetCDF or Zarr format for temperature and salinity). " *
                "Download them via GeoDataSources or check network access to NOAA NCEI. " *
                "To use the analytic profile instead, set [data] hydrography_source = \"synthetic\" in your config."
            )
        end
    elseif hydro_src in (:glorys12v1, :glorys_climatology, :copernicus, :cmesms)
        if boundary_tracer_data !== nothing && hasproperty(boundary_tracer_data, :T) && hasproperty(boundary_tracer_data, :S)
            println("Initializing hydrography from GeoData boundary tracer field ($hydro_src)...")
            set!(model, T = boundary_tracer_data.T, S = boundary_tracer_data.S)
            set!(model, u = 0.0, v = 0.0)
        else
            copernicus_file = joinpath(opts.input_dir, "copernicus_ts.nc")
            start_yr = hasproperty(opts, :start_year) ? opts.start_year : opts.projection_year
            try
                if !isfile(copernicus_file)
                    println("Fetching Copernicus Marine hydrography (trying GLOPHY-2 → GLO12 → GLORYS12 → NRT)...")
                    fetch_copernicus_hydrography_with_fallback(
                        lon_range = opts.domain_lon,
                        lat_range = opts.domain_lat,
                        start_date = "$(start_yr)-01-01",
                        end_date = "$(start_yr)-12-31",
                        output_path = copernicus_file,
                        verbose = true
                    )
                else
                    println("Using cached Copernicus Marine hydrography: $copernicus_file")
                end
                println("Applying Copernicus Marine hydrography...")
                hydro_month = opts.hydrography_month == 0 ? 1 : clamp(opts.hydrography_month, 1, 12)
                cop = copernicus_regridded_tracers(model.grid, copernicus_file; month = hydro_month)
                set!(model, T = cop.T, S = cop.S)
                set!(model, u = 0.0, v = 0.0)
            catch err
                error(
                    "hydrography_source = \"$(hydro_src)\" requires Copernicus Marine download " *
                    "which failed: $err. " *
                    "Ensure 'copernicusmarine' is installed and credentials configured. " *
                    "To use the analytic profile instead, set [data] hydrography_source = \"synthetic\"."
                )
            end
        end
    else
        println("Applying baseline thermal stratification (T_surf=15°C, dT/dz=0.01°C/m)...")
        set_initial_stratification!(
            model,
            surface_temperature = 15.0,
            temperature_gradient = 0.01,
            salinity = 35.0
        )
    end

    # Calculate Simpson-Hunter tidal mixing front parameters
    chi_bank  = simpson_hunter_parameter(40.0, 1.1)  # Shallow bank (mixed)
    chi_shelf = simpson_hunter_parameter(150.0, 0.2) # Deep shelf (stratified)
    println("Simpson-Hunter Tidal Mixing Diagnostics:")
    println("  Shallow Bank (h=40m,  U=1.1m/s): χ = $(round(chi_bank, digits=2)) (well-mixed if < 1.5)")
    println("  Deep Shelf   (h=150m, U=0.2m/s): χ = $(round(chi_shelf, digits=2)) (stratified if > 2.0)")

    return (
        model = model,
        tidal_forcing = tidal_forcing,
        simpson_hunter_bank = chi_bank,
        simpson_hunter_shelf = chi_shelf
    )
end

# ─────────────────────────────────────────────────────────────────────────────
# Segment 4: Climate Forcing Scenarios & Larval Thermal Ecology
# ─────────────────────────────────────────────────────────────────────────────

"""
    run_segment_climate(; opts::HydrodynamicOptions, model=nothing) -> NamedTuple

Inspect CMIP6 climate scenario deltas, apply warming and freshening anomalies to
the model stratification, and evaluate temperature-dependent PLD and mortality.

# Inputs
- `opts::HydrodynamicOptions`: Workflow options.
- `model`: Optional `HydrostaticFreeSurfaceModel` to update in-place.

# Outputs
- `NamedTuple`: `(deltas,)` -- the scenario anomaly deltas. Pelagic larval duration and
  thermal mortality are larval biology and are computed by `GeoLarvalTracking`, not here.
"""
function run_segment_climate(;
    opts::HydrodynamicOptions = HydrodynamicOptions(),
    model = nothing
)
    println("\n=================================================================")
    println(" [Segment 4/8] Climate Scenario Integration & Thermal Ecology")
    println("=================================================================")

    deltas = get_climate_scenario_deltas(opts.scenario, year = opts.projection_year)
    println("Climate Scenario: $(deltas.description) [Year $(opts.projection_year)]")
    println("  Surface Temperature Anomaly: +$(round(deltas.ΔT_surface, digits=2)) °C")
    println("  CIL Temperature Anomaly:     +$(round(deltas.ΔT_cil, digits=2)) °C")
    println("  Deep Temperature Anomaly:    +$(round(deltas.ΔT_deep, digits=2)) °C")
    println("  Surface Salinity Anomaly:    $(round(deltas.ΔS_surface, digits=2)) PSU")
    println("  Atmospheric Wind Factor:     x$(round(deltas.Δwind_factor, digits=2))")

    if !isnothing(model) && opts.scenario ∉ (:baseline, :historical, :climatology) &&
       opts.ocean_boundary_source ∉ (:glorys12v1, :glorys_climatology)
        println("Applying climate anomalies to model stratification...")
        apply_climate_scenario!(model, scenario = opts.scenario, year = opts.projection_year)
    end

    # Pelagic larval duration and thermal mortality are NOT computed here: they are larval
    # biology, owned by the larval half (GeoLarvalTracking). This segment ends at the ocean
    # effect of a scenario -- the warming/freshening anomaly applied to the stratification.
    return (deltas = deltas,)
end

# ─────────────────────────────────────────────────────────────────────────────
# Segment 5: Hydrodynamic Simulation Execution
# ─────────────────────────────────────────────────────────────────────────────

"""
    run_segment_simulation(; opts::HydrodynamicOptions, model=nothing) -> NamedTuple

Configure Oceananigans time stepping, adaptive CFL wizard, stability watchdogs,
and execute the regional hydrodynamic integration.

# Inputs
- `opts::HydrodynamicOptions`: Workflow options.
- `model`: Configured `HydrostaticFreeSurfaceModel`.

# Outputs
- `NamedTuple`: `(simulation, output_path, checkpoint_path)` — `output_path` is the GeoData
  output store, `checkpoint_path` the JLD2 restart checkpoint.
"""
function run_segment_simulation(;
    opts::HydrodynamicOptions = HydrodynamicOptions(),
    model = nothing
)
    default_jld2 = "hydrodynamics_$(opts.scenario)_$(opts.projection_year).jld2"
    jld2_path, _ = resolve_hydro_model_path(opts, default_jld2)
    # The store is where this run's output lives and is what resume/reuse act on; the JLD2 is
    # only the Oceananigans restart checkpoint (state, not a deliverable).
    store_path = output_zarr_path(dirname(jld2_path))

    # Inspect the store's own record: how far it has run and how many slices it holds. This
    # replaced inspecting a JLD2 timeseries archive, which is no longer written.
    file_info = _store_sim_info(store_path; stop_time = opts.sim_duration)
    if file_info.exists
        println("Existing GeoData store at $(store_path): reaching " *
                "$(round(file_info.last_time / 3600.0, digits = 2)) h over " *
                "$(file_info.n_timesteps) snapshot(s).")
    end

    # If track-only: require an existing store with snapshots to track through
    if opts.track_only
        println("\n=================================================================")
        println(" [Segment 5/8] Hydrodynamic Simulation Time Stepping")
        println("=================================================================")
        if file_info.exists && file_info.n_timesteps > 0
            println("Using existing GeoData store from: $(store_path)")
            return (simulation = nothing, output_path = store_path)
        else
            error("Cannot run in --track-only mode: no GeoData store with snapshots at $(store_path)\n" *
                  "Please run with --hydro-only first or specify an existing store with --hydro-model=<path>.")
        end
    end

    # If reuse-hydro requested, or animate-hydro on an existing simulation: reuse the store.
    can_reuse = opts.reuse_hydro && file_info.exists && file_info.n_timesteps > 0
    if can_reuse
        println("\n=================================================================")
        println(" [Segment 5/8] Hydrodynamic Simulation Time Stepping")
        println("=================================================================")
        if file_info.is_complete
            println("Reusing completed GeoData store from: $(store_path)")
        else
            println("Reusing GeoData store from: $(store_path)")
            println("  Notice: $(file_info.n_timesteps) snapshot(s) present (up to $(round(file_info.last_time / 3600.0, digits=2)) h).")
        end
        return (simulation = nothing, output_path = store_path)
    end

    target_model = if isnothing(model)
        model_res = run_segment_model(opts = opts)
        run_segment_climate(opts = opts, model = model_res.model)
        model_res.model
    else
        model
    end

    println("\n=================================================================")
    println(" [Segment 5/8] Hydrodynamic Simulation Time Stepping")
    println("=================================================================")
    mkpath(opts.output_dir)

    # Existing output: resume, extend, or discard.
    #
    # `sim_duration` is the END TIME measured from t = 0, not the length of the work still to
    # do. That single decision makes extending a run require nothing but editing the number:
    # the inputs are already cached under this scenario's input directory and are not
    # re-downloaded, the model state is picked up from wherever it stopped, and raising
    # `sim_duration` runs the extra time. Resuming an interrupted run and extending a finished
    # one are therefore the same operation, and neither needs a separate flag.
    #
    # Two conditions must not loop, and they are different:
    #
    #   * the archive holds NO progress (stored time still zero) -- resuming cannot help, so
    #     it is discarded immediately;
    #   * the archive holds progress at T, but a previous attempt to continue from T already
    #     failed. The file is unchanged by that attempt, so resuming from it reproduces the
    #     same failure indefinitely. A small sidecar records the stored time at the moment each
    #     resume was attempted; if it still matches, that continuation has been tried and the
    #     archive is discarded.
    #
    # The sidecar needs no clearing. A successful continuation advances the archive's stored
    # time past the recorded value, so the match simply stops happening; the next extension
    # overwrites it. That is why a plain "last attempted time" is sufficient state -- the
    # archive itself is the record of whether progress was made.
    out_dir_target = dirname(jld2_path)
    attempt_path = string(jld2_path, ".attempt")

    discard_run!(reason) = begin
        println(reason)
        println("  Discarding the GeoData store and its checkpoints; starting from t = 0.")
        ispath(store_path) && rm(store_path; recursive = true, force = true)
        cpdir = joinpath(out_dir_target, "checkpoints")
        isdir(cpdir) && for f in readdir(cpdir; join = true)
            endswith(f, ".jld2") && rm(f; force = true)
        end
        isfile(attempt_path) && rm(attempt_path; force = true)
        nothing
    end

    # `attempted_at` is the stored time of the last resume attempt, or NaN if none was recorded
    # or the sidecar is unreadable. An unreadable sidecar is treated as "never attempted",
    # which resumes rather than discards: losing an archive on the strength of a corrupt
    # sidecar would be the worse failure.
    attempted_at = NaN
    if isfile(attempt_path)
        attempted_at = try
            parse(Float64, strip(read(attempt_path, String)))
        catch
            NaN
        end
    end
    failed_before = !isnan(attempted_at) && attempted_at == file_info.last_time

    resume = false
    if opts.overwrite
        discard_run!("Flag --overwrite requested: purging existing GeoData store and checkpoints.")
    elseif file_info.exists && !file_info.is_complete && file_info.last_time <= 0
        discard_run!("Existing store $(basename(store_path)) is incomplete with no recorded progress.")
    elseif file_info.exists && failed_before && opts.auto_restart
        discard_run!(
            "A previous attempt to continue this run from t = " *
            "$(round(file_info.last_time / 3600.0, digits=2)) h did not get any further " *
            "(the store's stored time is unchanged).")
    elseif file_info.exists && !file_info.is_complete && opts.auto_restart
        println("Found interrupted simulation on the store at: $(store_path)")
        println("  (progress: $(round(file_info.last_time / 3600.0, digits=2)) of " *
                "$(round(opts.sim_duration / 3600.0, digits=2)) hours)")
        file_info.last_time >= opts.sim_duration &&
            println("  Resuming to the configured end time $(round(opts.sim_duration / 3600.0, digits=2)) h " *
                    "-- raise sim_duration to extend this run further.")
        println("Resuming simulation gracefully from latest checkpoint...")
        resume = true
    elseif file_info.exists && !file_info.is_complete
        println("Found incomplete simulation on $(store_path), but auto-restart is disabled.")
        println("Starting fresh simulation from t = 0...")
    elseif file_info.exists && file_info.is_complete && opts.auto_restart
        if file_info.last_time >= opts.sim_duration
            println("Existing run already reached $(round(file_info.last_time / 3600.0, digits=2)) h, " *
                    "which is at or beyond the configured end time " *
                    "$(round(opts.sim_duration / 3600.0, digits=2)) h.")
            println("Raise sim_duration in the TOML to extend this run; nothing to do at the " *
                    "current length.")
        else
            println("Extending the existing run from $(round(file_info.last_time / 3600.0, digits=2)) h " *
                    "to $(round(opts.sim_duration / 3600.0, digits=2)) h.")
            resume = true
        end
    end

    # Record the attempt only when we are about to continue an existing archive, and only
    # after the decision to resume has been made: a run that starts from t = 0 leaves the
    # archive behind, and a stale sidecar must not be able to discard it on the next call.
    if resume
        mkpath(out_dir_target)
        write(attempt_path, string(file_info.last_time))
    end

    out_dir_target = dirname(jld2_path)
    mkpath(out_dir_target)

    # Checkpoint storage directory
    cp_dir_target = isempty(opts.checkpoint_dir) ?
        joinpath(out_dir_target, "checkpoints") : opts.checkpoint_dir

    # Where this invocation writes: the GeoData store beside this run's outputs. A resume
    # appends to it (the writer keeps the interval already written); a fresh run replaces it.
    # There are no `_partN` JLD2 archives any more -- the store is the deliverable, and its
    # own `time` coordinate is the record of progress the resume/reuse logic reads back.

    # Initial time step: start at 10% of the configured sim_dt to allow the
    # CFL wizard to safely ramp up from rest without violating barotropic CFL.
    init_Δt  = Float64(opts.sim_dt) * 0.10
    max_Δt   = Float64(opts.max_dt)

    spinup_sec = opts.spinup_hours * 3600.0
    total_sim_duration = opts.sim_duration + spinup_sec

    println("Setting up simulation (target_duration=$(opts.sim_duration)s, spinup=$(spinup_sec)s, total_stop_time=$(total_sim_duration)s, Δt₀=$(init_Δt)s, max_Δt=$(max_Δt)s, min_Δt=$(opts.min_dt_seconds)s)...")
    sim = setup_hydrodynamic_simulation(
        target_model,
        Δt = init_Δt,
        stop_time = total_sim_duration,
        adaptive_time_step = opts.adaptive_cfl,
        target_cfl = opts.target_cfl,
        max_Δt = max_Δt,
        min_Δt = opts.min_dt_seconds,
        enable_output = false,
        output_schedule = (opts.output_schedule_seconds > 0.0) ?
            opts.output_schedule_seconds : 21600.0,
        progress_schedule = 100,
        enable_checkpoint = opts.enable_checkpoint,
        checkpoint_dir = cp_dir_target,
        checkpoint_prefix = opts.checkpoint_prefix,
        checkpoint_schedule = opts.checkpoint_schedule > 0 ? opts.checkpoint_schedule : nothing,
        cleanup_checkpoints = opts.checkpoint_cleanup,
        pickup = resume ? :auto : false,
        divergence_velocity_limit = opts.divergence_velocity_limit
    )

    out_interval = Oceananigans.TimeInterval(
        (opts.output_schedule_seconds > 0.0) ? opts.output_schedule_seconds : 21600.0)
    output_store = attach_output_callback!(sim, out_dir_target; fields = ("u", "v", "w", "T"),
                                           schedule = out_interval, resume = resume,
                                           spinup_seconds = spinup_sec,
                                           max_save_depth = opts.max_save_depth)
    @assert output_store == store_path "output store drifted from the resume/reuse target"
    println("Ocean output is a GeoData store: $(output_store)")

    println("Integrating hydrostatic primitive equations...")
    run_hydrodynamic_simulation!(
        sim,
        pickup = resume ? :auto : false,
        verbose = true
    )

    completed_sim = sim.model.clock.time >= (sim.stop_time - 1e-3)
    if completed_sim
        println("Simulation completed successfully; output written to the GeoData store: $(output_store)")
    else
        println("Simulation paused/interrupted at $(prettytime(sim.model.clock.time)); state saved to checkpoints.")
    end

    return (simulation = sim, output_path = output_store, checkpoint_path = jld2_path)
end

# ─────────────────────────────────────────────────────────────────────────────
# Hydrodynamic Animation
# ─────────────────────────────────────────────────────────────────────────────

function render_hydrodynamic_animation_if_requested(
    opts::HydrodynamicOptions,
    hydro_source::AbstractString
)
    if !opts.animate_hydro
        return nothing
    end
    # The run's own bathymetry, resolved the same way the grid was, so the animation and the
    # model it shows describe one sea floor.
    bathy_data = _run_bathymetry(; opts = opts)
    # The run's own coastline, fetched in Segment 1 and kept in the run's input directory.
    # Read here rather than defaulting to the bundled regional rings: a figure that masks
    # against a different land than the model ran on silently draws land in the water.
    # Absent file means no rings, and the caller falls back to the region's rings.
    coastline_rings = let path = joinpath(opts.input_dir, "coastline.parquet")
        isfile(path) ? load_coastline_polygons(path) : nothing
    end
    anim_fmt = lowercase(opts.anim_format) == "gif" ? "gif" : "mp4"
    anim_file_path = if !isempty(strip(opts.anim_output_path))
        opts.anim_output_path
    elseif opts.anim_variable == :dashboard
        joinpath(opts.output_dir, "hydrodynamic_dashboard_animation.$(anim_fmt)")
    else
        joinpath(opts.output_dir, "hydrodynamic_$(opts.anim_variable)_animation.$(anim_fmt)")
    end

    mkpath(dirname(anim_file_path))
    if opts.anim_variable == :dashboard
        println("Rendering hydrodynamic dashboard -> $(anim_file_path)...")
        animate_hydrodynamic_dashboard(
            hydro_source;
            depth = opts.anim_depth,
            trajectories = nothing,
            bathymetry_data = bathy_data,
            output_path = anim_file_path,
            framerate = opts.anim_fps,
            show_trajectories = false,
            domain_lon = opts.domain_lon,
            domain_lat = opts.domain_lat,
            coastline = coastline_rings
        )
    else
        println("Rendering hydrodynamic $(opts.anim_variable) field -> $(anim_file_path)...")
        animate_hydrodynamic_field(
            hydro_source;
            variable = opts.anim_variable,
            depth = opts.anim_depth,
            trajectories = nothing,
            bathymetry_data = bathy_data,
            output_path = anim_file_path,
            framerate = opts.anim_fps,
            show_trajectories = false,
            domain_lon = opts.domain_lon,
            domain_lat = opts.domain_lat,
            coastline = coastline_rings
        )
    end
    println("Hydrodynamic animation saved to: $(anim_file_path)")
    return anim_file_path
end

# ─────────────────────────────────────────────────────────────────────────────
# Full End-to-End Production Pipeline (ocean path)
# ─────────────────────────────────────────────────────────────────────────────

"""
    run_production_pipeline(; opts::HydrodynamicOptions) -> NamedTuple

Execute the complete end-to-end regional modeling, hydrodynamic simulation,
snow crab larval tracking, metric extraction, and visualization workflow.

The ocean half runs Steps 1-5 and, with `--hydro-only`, stops at the handover. Reaching
Step 6 onwards is the larval half's job: `run_segment_tracking`, `run_segment_metrics` and
`run_segment_visualize` are stubs that say so, so a call that continues past the handover
fails with that message rather than silently running one entry point's two halves' work.
The pipeline is two commands with the handover file between them
(`GeoOceanModelRun.jl` then `GeoLarvalTrackingRun.jl`).

# Inputs
- `opts::HydrodynamicOptions`: Workflow options.

# Outputs
- `NamedTuple`: Summary of all workflow artifacts.
"""
function run_production_pipeline(; opts::HydrodynamicOptions = HydrodynamicOptions())
    println("\n=================================================================")
    println(" Executing Full Hydrodynamic & Particle Tracking Pipeline")
    println("=================================================================")
    t_start = time()

    # Step 1: Data Ingestion
    data_res = run_segment_data(opts = opts)

    # Step 2: Grid Construction
    grid_res = run_segment_grid(opts = opts)

    # Decoupled hydrodynamics: skip model and simulation if --track-only is active
    model_res = nothing
    climate_res = nothing
    sim_res = nothing

    if !opts.track_only
        # Step 3: Model Setup
        model_res = run_segment_model(
            opts = opts,
            immersed_grid = grid_res.immersed_grid,
            tau_x = data_res.tau_x,
            tau_y = data_res.tau_y
        )

        # Step 4: Climate Scenarios
        climate_res = run_segment_climate(opts = opts, model = model_res.model)

        # Step 5: Hydrodynamic Simulation
        sim_res = run_segment_simulation(opts = opts, model = model_res.model)
    else
        println("[Pipeline Notice] --track-only active: Skipping hydrodynamic simulation (Segments 3-5).")
    end

    # If --hydro-only is active, exit early after saving hydrodynamic solution
    if opts.hydro_only
        t_elapsed = round(time() - t_start, digits = 2)
        println("\n=================================================================")
        println(" Hydrodynamics-only execution complete in $(t_elapsed) s (--hydro-only specified).")
        println(" Flow solution saved to: $(isnothing(sim_res) ? store_path : sim_res.output_path)")
        println(" Skipping larval particle tracking and metrics.")

        viz_res = nothing
        if opts.animate_hydro
            println(" Generating requested hydrodynamic animations...")
            # The store the run just wrote, which is the only data source the figures take.
            # Before the store existed this read the simulation JLD2, which is now the
            # restart checkpoint alone -- the fields a figure draws are not in it.
            hydro_source = isnothing(sim_res) ? store_path : sim_res.output_path
            anim_file = render_hydrodynamic_animation_if_requested(opts, hydro_source)
            viz_res = (animation = anim_file,)
        end

        # No storage session to close: the ocean output is the GeoData Zarr store written
        # incrementally by the attached callback (`sim_res.output_path`), which a reader
        # opens by path whenever it needs it.
        println("=================================================================")
        return (
            data = data_res,
            grid = grid_res,
            model = model_res,
            climate = climate_res,
            simulation = sim_res,
            tracking = nothing,
            metrics = nothing,
            visualizations = viz_res,
            elapsed_seconds = t_elapsed
        )
    end

    # Step 6: Lagrangian Particle Tracking
    track_res = run_segment_tracking(opts = opts)

    # Step 7: Metrics & Connectivity
    metrics_res = run_segment_metrics(opts = opts, trajectories = track_res.trajectories)

    # Step 8: Visualizations
    viz_res = run_segment_visualize(opts = opts, trajectories = track_res.trajectories)

    t_elapsed = round(time() - t_start, digits = 2)
    println("\n=================================================================")
    println(" Complete Workflow Pipeline Successfully Finished in $(t_elapsed) s")
    println(" Outputs written to: $(opts.output_dir)/")
    if !isnothing(sim_res)
        println(" Ocean output (GeoData store): $(sim_res.output_path)")
    end
    println("=================================================================")

    return (
        data = data_res,
        grid = grid_res,
        model = model_res,
        climate = climate_res,
        simulation = sim_res,
        tracking = track_res,
        metrics = metrics_res,
        visualizations = viz_res,
        elapsed_seconds = t_elapsed
    )
end


# ==============================================================================
#  Boundary stubs
#
#  Names the larval half and the old storage/analytics API own. Two ocean functions
#  reach them: `run_production_pipeline` past the handover, and `run_segment_climate`'s
#  thermal-ecology lines and `run_segment_simulation`'s legacy archive block. Each is a
#  stub that errors naming where it lives, so the coupling is at the call site and a run
#  that crosses the split fails with that message rather than a `MethodError` later, or
#  worse, a silent substitution.
#
#  Each stub is a docstring immediately followed by one definition; an unterminated
#  docstring here is what made the earlier generated file fail `include` while passing
#  `Meta.parseall`, so the streaming parser in `scratch_parse_probe/joiner.jl` checks it.
# ==============================================================================

"""
    run_segment_tracking(args...; kwargs...)

Owned by the LARVAL half (`GeoLarvalTracking`). Run it from `GeoLarvalTrackingRun.jl`
against the store this package writes. `GeoOceanModel` ends at the handover; calling this
here would be one entry point doing two halves' work.
"""
run_segment_tracking(args...; kwargs...) = error(
    "`run_segment_tracking` is owned by GeoLarvalTracking, not by this package: run it " *
    "from `GeoLarvalTrackingRun.jl` against the store this one wrote.")

"""
    run_segment_metrics(args...; kwargs...)

Owned by the LARVAL half (`GeoLarvalTracking`). See `run_segment_tracking`.
"""
run_segment_metrics(args...; kwargs...) = error(
    "`run_segment_metrics` is owned by GeoLarvalTracking, not by this package: run it " *
    "from `GeoLarvalTrackingRun.jl` against the store this one wrote.")

"""
    run_segment_visualize(args...; kwargs...)

Owned by the LARVAL half (`GeoLarvalTracking`). See `run_segment_tracking`.
"""
run_segment_visualize(args...; kwargs...) = error(
    "`run_segment_visualize` is owned by GeoLarvalTracking, not by this package: run it " *
    "from `GeoLarvalTrackingRun.jl` against the store this one wrote.")
