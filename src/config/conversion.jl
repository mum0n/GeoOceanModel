"""
    conversion.jl

Configuration conversion.

Turning the nested dictionary into runtime options and into the three pure config
structs, and back. `get_val` lives here because it is how a dict may be missing
keys during conversion.
"""


"""
    configuration_to_options(config_dict::AbstractDict; overrides...) -> HydrodynamicOptions

Convert a configuration dictionary into a validated `HydrodynamicOptions` instance,
allowing optional keyword parameter overrides.

# Inputs
- `config_dict::AbstractDict`: Parsed configuration dictionary.
- `overrides...`: Additional keyword arguments to override dictionary entries.

# Outputs
- `HydrodynamicOptions`: Constructed runtime options instance.
"""
function configuration_to_options(config_dict::AbstractDict; overrides...)
    # Helper to safely extract nested values with defaults
    function get_val(section::String, key::String, default_val)
        if haskey(config_dict, section) && haskey(config_dict[section], key)
            return config_dict[section][key]
        end
        return default_val
    end

    lon_min = Float64(get_val("domain", "lon_min", -71.0))
    lon_max = Float64(get_val("domain", "lon_max", -53.0))
    lat_min = Float64(get_val("domain", "lat_min", 40.0))
    lat_max = Float64(get_val("domain", "lat_max", 48.5))
    z_min   = Float64(get_val("domain", "z_min", -3500.0))
    z_max   = Float64(get_val("domain", "z_max", 0.0))

    buffer_km = Float64(get_val("domain", "buffer_km", get_val("biology", "buffer_km", 100.0)))

    nx = Int(get_val("grid", "nx", 50))
    ny = Int(get_val("grid", "ny", 50))
    nz = Int(get_val("grid", "nz", 10))

    grid_orientation = Symbol(lowercase(String(get_val("grid", "orientation", "standard"))))
    rotation_azimuth = Float64(get_val("grid", "rotation_azimuth", 50.0))
    center_lon = Float64(get_val("grid", "center_lon", -63.0))
    center_lat = Float64(get_val("grid", "center_lat", 44.0))
    along_shelf_extent_km = Float64(get_val("grid", "along_shelf_extent_km", 850.0))
    cross_shelf_extent_km = Float64(get_val("grid", "cross_shelf_extent_km", 220.0))

    bathy_source = String(get_val("data", "bathy_source", "etopo2022"))
    coastline_source = String(get_val("data", "coastline_source", "natural_earth_10m"))
    wind_source = String(get_val("data", "wind_source", "open_meteo"))
    wind_time_iso = String(get_val("data", "wind_time_iso", "2020-06-01T00:00:00Z"))

    # Shelf geometry, from [bathymetry]. `inshore_depth` is the water depth at the landward
    # edge of the domain (positive, m); `shelf_slope` is the total rise across the shelf (m).
    # Both are read from here rather than hardcoded, so a config can actually shape its seabed.
    inshore_depth = abs(Float64(get_val("bathymetry", "inshore_depth", 20.0)))
    shelf_slope   = abs(Float64(get_val("bathymetry", "shelf_slope", 180.0)))
    mask_bay_of_fundy = Bool(get_val("bathymetry", "mask_bay_of_fundy", false))

    # Tidal body forcing. `constituents` is an empty list in a config that disables tides, so
    # an empty read is not an error; it simply means no constituents were requested.
    tides_src = String(get_val("tides", "tides_source", "tpxo9"))
    raw_cons = get_val("tides", "constituents", Any[:M2, :S2])
    # Normalised to upper case because `get_tidal_frequency` dispatches on `:M2`/`:S2`
    # case-sensitively, while the atlas file lookup is case-insensitive. Normalising here
    # means a TOML writing "m2" and one writing "M2" behave identically.
    tidal_cons = Symbol[Symbol(uppercase(String(c))) for c in raw_cons]

    enable_tides = Bool(get_val("tides", "enable_tides", true))
    tides_source = tides_src
    tidal_constituents = tidal_cons
    tidal_u = Float64(get_val("tides", "tidal_u_amp", 0.25))
    tidal_v = Float64(get_val("tides", "tidal_v_amp", 0.12))
    s2_u = Float64(get_val("tides", "s2_u_amp", 0.11))
    s2_v = Float64(get_val("tides", "s2_v_amp", 0.05))

    scenario = Symbol(get_val("climate", "scenario", "ssp245"))
    proj_year = Int(get_val("climate", "projection_year", 2050))

    sim_hours = Float64(get_val("hydrodynamics", "sim_duration_hours", 12.0))
    sim_dt    = Float64(get_val("hydrodynamics", "sim_dt_seconds", 120.0))
    # Floor on the adaptive time step, forwarded to the simulation as `min_Δt`. The surface cell
    # sets the admissible step (Δt ≈ target_cfl × Δz_min / w), so a floor above that value stops
    # the wizard stabilising the surface layer.
    min_dt_seconds = Float64(get_val("hydrodynamics", "min_dt_seconds", 2.0))
    adapt_cfl = Bool(get_val("hydrodynamics", "adaptive_cfl", true))
    tgt_cfl   = Float64(get_val("hydrodynamics", "target_cfl", 0.2))
    heat_flux = Float64(get_val("hydrodynamics", "surface_heat_flux", 50.0))
    # `bulk_heat_flux` selects WHICH heat flux is applied: true computes it from the ingested
    # atmospheric surface state, false applies the constant `surface_heat_flux` above. It was
    # parsed into `HydrodynamicConfig` and documented in the TOML but never carried onto
    # `HydrodynamicOptions`, so nothing downstream could read it and the bulk flux was applied
    # unconditionally.
    #
    # It is read from `[atmosphere]`, which is where the TOML actually carries it (next to
    # `atmospheric_source`, `wind_source` and the rest of the atmospheric configuration). The
    # `[hydrodynamics]` section is accepted as a fallback so the key is not silently ignored
    # if it is written there instead.
    bulk_heat_flux = if haskey(config_dict, "atmosphere") &&
                        haskey(config_dict["atmosphere"], "bulk_heat_flux")
        Bool(get_val("atmosphere", "bulk_heat_flux", true))
    else
        Bool(get_val("hydrodynamics", "bulk_heat_flux", true))
    end
    sst_damping = if haskey(config_dict, "atmosphere") &&
                     haskey(config_dict["atmosphere"], "sst_damping_coefficient")
        Float64(get_val("atmosphere", "sst_damping_coefficient", 0.0))
    else
        Float64(get_val("hydrodynamics", "sst_damping_coefficient", 0.0))
    end
    sst_target_val = Float64(get_val("hydrodynamics", "sst_target", 12.0))
    spinup_hours_val = Float64(get_val("hydrodynamics", "spinup_hours", 720.0))
    div_vel_lim = Float64(get_val("hydrodynamics", "divergence_velocity_limit", 20.0))
    # Cap on how many hydrodynamic snapshots are held in memory when building the 4D flow
    # interpolator for particle tracking. A production run stores thousands of snapshots and
    # materialising all of them for u, v, w and T is hundreds of gigabytes; the snapshots are
    # evenly subsampled down to this count. `0` means "no cap" (load every snapshot).
    max_flow_snapshots = Int(get_val("hydrodynamics", "max_flow_snapshots", 400))
    allow_analytical_fallback = Bool(get_val("hydrodynamics", "allow_analytical_fallback", false))
    # Cap on the background current a larva is advected by. A diverged hydrodynamics cell can
    # otherwise teleport a particle across the domain in a single step; 3 m/s is well above any
    # real Scotian Shelf current. Use a negative value to disable the clamp.
    _mcs = Float64(get_val("hydrodynamics", "max_current_speed", 3.0))
    max_current_speed = _mcs > 0 ? _mcs : nothing

    n_parts   = Int(get_val("biology", "n_particles", 100))
    track_days = Float64(get_val("biology", "track_duration_days", 5.0))
    track_dt  = Float64(get_val("biology", "track_dt_seconds", 300.0))
    min_depth = Float64(get_val("biology", "min_seabed_depth", 100.0))
    diff_h    = Float64(get_val("biology", "diffusivity_h", 10.0))
    diff_v    = Float64(get_val("biology", "diffusivity_v", 1e-4))

    rel_mode_str = String(get_val("biology", "release_depth_mode", "bottom"))
    rel_mode = Symbol(lowercase(rel_mode_str))
    raw_offset = get_val("biology", "bottom_release_offset", [0.5, 3.0])
    b_offset = if raw_offset isa Vector && length(raw_offset) >= 2
        (Float64(raw_offset[1]), Float64(raw_offset[2]))
    elseif raw_offset isa Tuple && length(raw_offset) >= 2
        (Float64(raw_offset[1]), Float64(raw_offset[2]))
    else
        (0.5, 3.0)
    end
    init_ascent = Bool(get_val("biology", "enable_initial_ascent", true))
    asc_spd     = Float64(get_val("biology", "ascent_speed", 0.010))
    asc_target  = Float64(get_val("biology", "ascent_target_depth", -10.0))

    enable_dvm = Bool(get_val("dvm", "enable_dvm", true))
    enable_molting = Bool(get_val("molting_and_settlement", "enable_molting", true))

    # Stochastic biology parameters
    settlement_stochastic = Bool(get_val("biology", "settlement_stochastic", true))
    cv_molt = Float64(get_val("biology", "cv_molt", 0.25))
    cv_mortality = Float64(get_val("biology", "cv_mortality", 0.25))
    cv_settlement = Float64(get_val("biology", "cv_settlement", 0.25))

     use_gpu      = Bool(get_val("runtype", "use_gpu", get_val("hardware", "use_gpu", false)))
     fallback_cpu = Bool(get_val("runtype", "fallback_to_cpu", get_val("hardware", "fallback_to_cpu", false)))
    interactive  = Bool(get_val("visualization", "interactive_map", true))
    anim_hydro   = Bool(get_val("visualization", "animate_hydro", false))
    anim_var     = Symbol(lowercase(String(get_val("visualization", "anim_variable", "dashboard"))))
    anim_fps     = Int(get_val("visualization", "anim_fps", 10))
    anim_fmt     = String(get_val("visualization", "anim_format", "mp4"))
    anim_depth   = Float64(get_val("visualization", "anim_depth", -2.5))
    anim_overlay = Bool(get_val("visualization", "anim_overlay_particles", false))
     anim_out_path = String(get_val("visualization", "anim_output_path", ""))

     output_path = String(get_val("storage", "output_path",
                          get_val("storage", "output_dir", joinpath("outputs", "default"))))
     zarr_path   = String(get_val("storage", "zarr_path", joinpath("outputs", "particle_tracking.zarr")))
     storage_backend = String(get_val("storage", "storage_backend", "zarr"))
     
     enable_checkpoint = Bool(get_val("storage", "enable_checkpoint", true))
     checkpoint_prefix = String(get_val("storage", "checkpoint_prefix", "checkpoint_ParticleTracking"))
     checkpoint_schedule_seconds = Float64(get_val("storage", "checkpoint_schedule_seconds", 21600.0))
     checkpoint_dir = String(get_val("storage", "checkpoint_dir", "outputs/checkpoints"))
     checkpoint_cleanup = Bool(get_val("storage", "checkpoint_cleanup", true))
     
     # `storage.output_path` is this run's output DIRECTORY -- under the lakehouse -- and
     # everything a run writes lives beneath it: the GeoData store, this run's `inputs/`
     # ingested-data cache, its provenance and its resolved config. It is deliberately NOT
     # shared between runs: one shared `inputs/` let a run with a changed domain silently
     # reuse another domain's bathymetry, the quiet mismatch `resolved_config.toml` and
     # `data_provenance.json` exist to expose. The cost is disk -- a dataset is fetched once
     # per scenario rather than once per machine -- and that is the intended trade, because it
     # makes each scenario a self-contained, shareable directory under
     # `lakehouse/GeoOceanModel/<name>/`.
     output_dir = output_path
     input_dir  = joinpath(output_dir, "inputs")
     seed       = Int(get_val("runtype", "seed", get_val("paths", "seed", 42)))

     hydro_file = String(get_val("hydrodynamics", "hydro_model_file", ""))
     hydro_only = Bool(get_val("hydrodynamics", "hydro_only", false))
     track_only = Bool(get_val("hydrodynamics", "track_only", false))
    reuse_hydro = Bool(get_val("hydrodynamics", "reuse_hydro", false))
    run_id_val = String(get_val("storage", "run_id", ""))

    enable_cp = Bool(get_val("storage", "enable_checkpoint",
                             get_val("hydrodynamics", "enable_checkpoint", true)))
    cp_pfx_raw = String(get_val("storage", "checkpoint_prefix",
                                get_val("hydrodynamics", "checkpoint_prefix", "")))
    cp_sched  = Float64(get_val("storage", "checkpoint_schedule_seconds",
                                get_val("hydrodynamics", "checkpoint_schedule_seconds", 0.0)))
    # Field-output cadence, distinct from the checkpoint cadence. Read from [storage] rather
    # than hardcoded downstream, so the two can legitimately differ.
    out_sched = Float64(get_val("storage", "output_schedule_seconds", 21600.0))
    max_save_d = Float64(get_val("storage", "max_save_depth", Inf))
     cp_dir    = String(get_val("storage", "checkpoint_dir",
                                get_val("runtype", "checkpoint_dir",
                                        get_val("paths", "checkpoint_dir", ""))))
    cp_clean  = Bool(get_val("storage", "checkpoint_cleanup", true))
    auto_res  = Bool(get_val("hydrodynamics", "auto_restart", true))
    v_mode_str = String(get_val("grid", "vertical_stretching_mode", "oceananigans"))
    # Two-segment shelf grid parameters. `vertical_stretching_mode = "two_segment"` splits `nz`
    # into a surface-refined upper segment above `vertical_break_depth` and a coarse lower segment,
    # which is the only way to resolve the 10-400 m active layer while still reaching the deep basin.
    # `nz_above = 0` means "derive it as 60% of nz".
    nz_above_cfg = Int(get_val("grid", "nz_above", 0))
    vertical_break_depth = abs(Float64(get_val("grid", "vertical_break_depth", 400.0)))
    v_mode = Symbol(lowercase(v_mode_str))
    v_file = strip(String(get_val("grid", "vertical_grid_file", "")))

    # `vertical_grid_file` is the operator's own file: a relative path is resolved against the
    # working directory they ran from, not against a previous run's inputs. If they name a
    # file, it has to exist -- silently falling back to a generated grid would run the whole
    # simulation on a vertical grid nobody chose. The file is then copied into this run's
    # containerised input directory so the run is self-contained and reproducible.
    #
    # `[grid] depths` is the discretised vertical axis written inline in the config, as
    # positive-downward face depths starting at the surface: depths = [0, 10, 20, ...].
    # It is the alternative to naming a CSV, and the two are mutually exclusive so a config
    # can never silently prefer one over the other.
    raw_depths = get_val("grid", "depths", Any[])
    v_depths = Float64[Float64(d) for d in raw_depths]
    if !isempty(v_depths)
        isempty(v_file) || error(
            "[grid] sets both vertical_grid_file and depths. Use one: an explicit depth list " *
            "is easier to review in the config than a side-car CSV.")
        abs(first(v_depths)) <= 1e-6 || error(
            "[grid] depths must start at 0.0 (the surface); got $(first(v_depths)).")
        all(diff(v_depths) .> 0) || error(
            "[grid] depths must increase monotonically (positive downward): $(v_depths).")
        length(v_depths) == nz + 1 || error(
            "[grid] depths has $(length(v_depths)) entries, but nz = $nz requires " *
            "$(nz + 1) faces.")
        # The deepest face has to reach the bottom of the domain, or the grid silently stops
        # short of the seabed and the deepest water column is left unresolvable.
        abs(last(v_depths) - abs(z_min)) <= 1.0 || error(
            "[grid] depths ends at $(last(v_depths)) m but [domain] z_min is $(z_min) m. " *
            "The deepest face must reach the domain bottom.")
    end
    if !isempty(v_file)
        src = abspath(v_file)
        isfile(src) || error(
            "[grid] vertical_grid_file names '$v_file', which does not exist (resolved to " *
            "$src). Supply the file, or remove the key to derive the grid from the depth " *
            "range and nz instead."
        )
        dst = joinpath(input_dir, basename(v_file))
        isfile(dst) || cp(src, dst, force = true)
        v_file = dst
    end
    atmo_src = Symbol(lowercase(String(get_val("atmosphere", "forcing", "era5_climatology"))))
    obc_src_val = get_val("boundaries", "ocean_boundary_source", nothing)
    if isnothing(obc_src_val)
        obc_src_val = get_val("boundaries", "parent_ocean", "glorys12v1")
    end
    obc_src = Symbol(lowercase(String(obc_src_val)))
    obc_tp = Symbol(lowercase(String(get_val("boundaries", "obc_type", "flather_chapman"))))

    # Initial 3-D hydrography (T, S). Read from `[data]`, independent of the
    # bathymetry/wind `bathy_source` and the lateral `ocean_boundary_source`, because the
    # initial water column is a separate choice from the forcing and the boundaries.
    hydro_src = Symbol(lowercase(String(get_val("data", "hydrography_source", "synthetic"))))
    hydro_month = Int(get_val("data", "hydrography_month", 0))

    # Lateral boundary relaxation (Price & Aumont 2011) parameters
    boundary_method = Symbol(lowercase(String(get_val("boundaries", "method", "relaxation"))))
    sponge_width_val = Float64(get_val("boundaries", "sponge_layer_width", 0.35))
    sponge_tau_val   = Float64(get_val("boundaries", "sponge_timescale_seconds", 3600.0))
    u_inflow_val     = Float64(get_val("boundaries", "u_inflow", -0.15))
    v_inflow_val     = Float64(get_val("boundaries", "v_inflow", 0.05))
    act_b_raw        = get_val("boundaries", "active_boundaries", ["east"])
    act_b_tuple      = if act_b_raw isa AbstractVector
        Tuple(Symbol(lowercase(string(b))) for b in act_b_raw)
    elseif act_b_raw isa Tuple
        Tuple(Symbol(lowercase(string(b))) for b in act_b_raw)
    elseif act_b_raw isa Symbol || act_b_raw isa AbstractString
        (Symbol(lowercase(string(act_b_raw))),)
    else
        (:east,)
    end

    # Maximum allowable time step ceiling
    max_dt_val = Float64(get_val("hydrodynamics", "max_dt_seconds", sim_dt))

    # Voronoi tessellation parameters
    enable_voronoi = Bool(get_val("tessellation", "enable_voronoi", false))
    voronoi_n_units = Int(get_val("tessellation", "n_units", 5000))
    voronoi_p_core = Float64(get_val("tessellation", "prob_core", 0.8))
    voronoi_p_shallow = Float64(get_val("tessellation", "prob_shallow", 0.1))
    voronoi_p_deep = Float64(get_val("tessellation", "prob_deep", 0.1))
    voronoi_min_core = Float64(get_val("tessellation", "min_res_core_km", 1.5))
    voronoi_min_shallow = Float64(get_val("tessellation", "min_res_shallow_km", 5.0))
    voronoi_min_deep = Float64(get_val("tessellation", "min_res_deep_km", 10.0))
    # Depth band edges, as positive depths below the surface (the TOML convention). The
    # tessellator works in signed z (negative downwards), so the sign is applied when the
    # bands are passed to it; see the `generate_depth_stratified_voronoi_units` call.
    voronoi_core_min     = Float64(get_val("tessellation", "core_depth_min", 50.0))
    voronoi_core_max     = Float64(get_val("tessellation", "core_depth_max", 350.0))
    voronoi_shallow_min  = Float64(get_val("tessellation", "shallow_depth_min", 0.0))
    voronoi_shallow_max  = Float64(get_val("tessellation", "shallow_depth_max", 50.0))
    voronoi_deep_min     = Float64(get_val("tessellation", "deep_depth_min", 350.0))
    voronoi_deep_max     = Float64(get_val("tessellation", "deep_depth_max", 6000.0))
    voronoi_slope_wt     = Bool(get_val("tessellation", "slope_weighting", true))
    voronoi_slope_fac    = Float64(get_val("tessellation", "slope_factor", 15.0))

    # Construct HydrodynamicOptions with overrides applied
    return HydrodynamicOptions(;
        domain_lon = (lon_min, lon_max),
        domain_lat = (lat_min, lat_max),
        domain_z = (z_min, z_max),
        grid_size = (nx, ny, nz),
        grid_orientation = grid_orientation,
        rotation_azimuth = rotation_azimuth,
        center_lon = center_lon,
        center_lat = center_lat,
        along_shelf_extent_km = along_shelf_extent_km,
        cross_shelf_extent_km = cross_shelf_extent_km,
        bathy_source = bathy_source,
    coastline_source = coastline_source,
    wind_source = wind_source,
    wind_time_iso = wind_time_iso,
        inshore_depth = inshore_depth,
        shelf_slope = shelf_slope,
        enable_tides = enable_tides,
    tides_source = tides_source,
    tidal_constituents = tidal_constituents,
        tidal_u_amp = tidal_u,
        tidal_v_amp = tidal_v,
        s2_u_amp = s2_u,
        s2_v_amp = s2_v,
        scenario = scenario,
        projection_year = proj_year,
        sim_dt = sim_dt,
        sim_duration = sim_hours * 3600.0,
        min_dt_seconds = min_dt_seconds,
        adaptive_cfl = adapt_cfl,
        target_cfl = tgt_cfl,
        surface_heat_flux = heat_flux,
        bulk_heat_flux = Bool(bulk_heat_flux),
        sst_damping_coefficient = sst_damping,
        sst_target = sst_target_val,
        spinup_hours = spinup_hours_val,
        divergence_velocity_limit = div_vel_lim,
        max_flow_snapshots = max_flow_snapshots,
    allow_analytical_fallback = allow_analytical_fallback,
    max_current_speed = (max_current_speed isa Real && max_current_speed > 0) ?
                        Float64(max_current_speed) : nothing,
        n_particles = n_parts,
        track_duration = track_days * 86400.0,
        track_dt = track_dt,
        diffusivity_h = diff_h,
        diffusivity_v = diff_v,
        enable_dvm = enable_dvm,
        enable_molting = enable_molting,
        min_seabed_depth = min_depth,
        buffer_km = buffer_km,
        release_depth_mode = rel_mode,
        bottom_release_offset = b_offset,
        enable_initial_ascent = init_ascent,
        ascent_speed = asc_spd,
        ascent_target_depth = asc_target,
        use_gpu = use_gpu,
        fallback_to_cpu = fallback_cpu,
         interactive_map = interactive,
        zarr_path = zarr_path,
        output_path = output_path,
        storage_backend = storage_backend,
        output_dir = output_dir,
        input_dir = input_dir,
        seed = seed,
        hydro_model_file = hydro_file,
        hydro_only = hydro_only,
        track_only = track_only,
        reuse_hydro = reuse_hydro,
        enable_checkpoint = enable_cp,
        checkpoint_prefix = cp_pfx_raw,
        checkpoint_schedule = cp_sched,
        output_schedule_seconds = out_sched,
        max_save_depth = max_save_d,
        checkpoint_dir = cp_dir,
        checkpoint_cleanup = cp_clean,
        auto_restart = auto_res,
        overwrite = Bool(get_val("storage", "overwrite", false)),
        run_id = run_id_val,
        vertical_stretching_mode = v_mode,
        nz_above = nz_above_cfg,
        vertical_break_depth = vertical_break_depth,
        vertical_grid_file = v_file,
        vertical_depths = v_depths,
        atmospheric_source = atmo_src,
    ocean_boundary_source = obc_src,
    embedding_lon = embedding_domain_ranges(config_dict).lon,
    embedding_lat = embedding_domain_ranges(config_dict).lat,
    hydrography_source = hydro_src,
    hydrography_month = hydro_month,
        obc_type = obc_tp,
        enable_voronoi = enable_voronoi,
        voronoi_n_units = voronoi_n_units,
        voronoi_prob_core = voronoi_p_core,
        voronoi_prob_shallow = voronoi_p_shallow,
        voronoi_prob_deep = voronoi_p_deep,
        voronoi_min_res_core_km = voronoi_min_core,
        voronoi_min_res_shallow_km = voronoi_min_shallow,
        voronoi_min_res_deep_km = voronoi_min_deep,
        voronoi_core_depth_min = voronoi_core_min,
        voronoi_core_depth_max = voronoi_core_max,
        voronoi_shallow_depth_min = voronoi_shallow_min,
        voronoi_shallow_depth_max = voronoi_shallow_max,
        voronoi_deep_depth_min = voronoi_deep_min,
        voronoi_deep_depth_max = voronoi_deep_max,
        voronoi_slope_weighting = voronoi_slope_wt,
        voronoi_slope_factor = voronoi_slope_fac,
        animate_hydro = anim_hydro,
        anim_variable = anim_var,
        anim_fps = anim_fps,
        anim_format = anim_fmt,
        anim_depth = anim_depth,
        anim_overlay_particles = anim_overlay,
        anim_output_path = anim_out_path,
        boundary_method = boundary_method,
        sponge_layer_width = sponge_width_val,
        sponge_timescale = sponge_tau_val,
        tidal_relaxation_timescale = Float64(get_val("tides", "tidal_relaxation_timescale_seconds", 3600.0)),
        u_inflow = u_inflow_val,
        v_inflow = v_inflow_val,
        max_dt = max_dt_val,
        settlement_stochastic = settlement_stochastic,
        cv_molt = cv_molt,
        cv_mortality = cv_mortality,
        cv_settlement = cv_settlement,
        mask_bay_of_fundy = mask_bay_of_fundy,
        active_boundaries = act_b_tuple,
        overrides...
    )
end

"""
    options_to_configuration(opts::HydrodynamicOptions) -> Dict{String, Any}

Extract a complete configuration dictionary from a `HydrodynamicOptions` instance.
"""
function options_to_configuration(opts::HydrodynamicOptions)::Dict{String, Any}
    return Dict{String, Any}(
        "domain" => Dict{String, Any}(
            "lon_min" => opts.domain_lon[1],
            "lon_max" => opts.domain_lon[2],
            "lat_min" => opts.domain_lat[1],
            "lat_max" => opts.domain_lat[2],
            "z_min" => opts.domain_z[1],
            "z_max" => opts.domain_z[2],
            "buffer_km" => opts.buffer_km
        ),
        "grid" => Dict{String, Any}(
            "nx" => opts.grid_size[1],
            "ny" => opts.grid_size[2],
            "nz" => opts.grid_size[3],
            "orientation" => string(opts.grid_orientation),
            "rotation_azimuth" => opts.rotation_azimuth,
            "center_lon" => opts.center_lon,
            "center_lat" => opts.center_lat,
            "along_shelf_extent_km" => opts.along_shelf_extent_km,
            "cross_shelf_extent_km" => opts.cross_shelf_extent_km,
            "vertical_stretching_mode" => string(opts.vertical_stretching_mode),
            "nz_above" => opts.nz_above,
            "vertical_break_depth" => opts.vertical_break_depth,
            "vertical_grid_file" => opts.vertical_grid_file,
            "depths" => opts.vertical_depths
        ),
        "atmosphere" => Dict{String, Any}(
            "source" => string(opts.atmospheric_source)
        ),
        "boundaries" => Dict{String, Any}(
            "ocean_boundary_source" => string(opts.ocean_boundary_source),
            "obc_type" => string(opts.obc_type),
            "method" => string(opts.boundary_method),
            "sponge_layer_width" => opts.sponge_layer_width,
            "sponge_timescale_seconds" => opts.sponge_timescale,
            "u_inflow" => opts.u_inflow,
            "v_inflow" => opts.v_inflow
        ),
        "data" => Dict{String, Any}(
            "bathy_source" => opts.bathy_source,
            "coastline_source" => opts.coastline_source,
            "wind_source" => opts.wind_source,
            "wind_time_iso" => opts.wind_time_iso,
            "hydrography_source" => string(opts.hydrography_source),
            "hydrography_month" => opts.hydrography_month
        ),
        "bathymetry" => Dict{String, Any}(
            "inshore_depth" => opts.inshore_depth,
            "shelf_slope" => opts.shelf_slope,
            "mask_bay_of_fundy" => opts.mask_bay_of_fundy
        ),
        "tides" => Dict{String, Any}(
            "enable_tides" => opts.enable_tides,
            "tides_source" => opts.tides_source,
            "tidal_constituents" => String.(opts.tidal_constituents),
            "tidal_u_amp" => opts.tidal_u_amp,
            "tidal_v_amp" => opts.tidal_v_amp,
            "s2_u_amp" => opts.s2_u_amp,
            "s2_v_amp" => opts.s2_v_amp
        ),
        "climate" => Dict{String, Any}(
            "scenario" => string(opts.scenario),
            "projection_year" => opts.projection_year
        ),
        "hydrodynamics" => Dict{String, Any}(
            "sim_duration_hours" => opts.sim_duration / 3600.0,
            "sim_dt_seconds" => opts.sim_dt,
            "adaptive_cfl" => opts.adaptive_cfl,
            "target_cfl" => opts.target_cfl,
            "max_dt_seconds" => opts.max_dt,
            "min_dt_seconds" => opts.min_dt_seconds,
            "surface_heat_flux" => opts.surface_heat_flux,
            "max_flow_snapshots" => something(opts.max_flow_snapshots, 0),
            "allow_analytical_fallback" => opts.allow_analytical_fallback,
            "max_current_speed" => something(opts.max_current_speed, 0.0),
            "hydro_model_file" => opts.hydro_model_file,
            "hydro_only" => opts.hydro_only,
            "track_only" => opts.track_only,
            "reuse_hydro" => opts.reuse_hydro
        ),
        "biology" => Dict{String, Any}(
            "n_particles" => opts.n_particles,
            "track_duration_days" => opts.track_duration / 86400.0,
            "track_dt_seconds" => opts.track_dt,
            "min_seabed_depth" => opts.min_seabed_depth,
            "buffer_km" => opts.buffer_km,
            "diffusivity_h" => opts.diffusivity_h,
            "diffusivity_v" => opts.diffusivity_v,
            "release_depth_mode" => string(opts.release_depth_mode),
            "bottom_release_offset" => [opts.bottom_release_offset[1], opts.bottom_release_offset[2]],
            "enable_initial_ascent" => opts.enable_initial_ascent,
            "ascent_speed" => opts.ascent_speed,
            "ascent_target_depth" => opts.ascent_target_depth,
            "settlement_stochastic" => opts.settlement_stochastic,
            "cv_molt" => opts.cv_molt,
            "cv_mortality" => opts.cv_mortality,
            "cv_settlement" => opts.cv_settlement
        ),
        "dvm" => Dict{String, Any}(
            "enable_dvm" => opts.enable_dvm
        ),
        "molting_and_settlement" => Dict{String, Any}(
            "enable_molting" => opts.enable_molting,
            "cv_molt" => opts.cv_molt,
            "cv_mortality" => opts.cv_mortality
        ),
        "storage" => Dict{String, Any}(
            "output_dir" => opts.output_dir,
            "enable_zarr" => opts.enable_zarr,
            "zarr_path" => opts.zarr_path,
            "run_id" => opts.run_id,
            "enable_checkpoint" => opts.enable_checkpoint,
            "checkpoint_prefix" => opts.checkpoint_prefix,
            "checkpoint_schedule_seconds" => opts.checkpoint_schedule,
            "output_schedule_seconds" => opts.output_schedule_seconds,
            "checkpoint_dir" => opts.checkpoint_dir,
            "checkpoint_cleanup" => opts.checkpoint_cleanup
        ),
        "tessellation" => Dict{String, Any}(
            "enable_voronoi" => opts.enable_voronoi,
            "n_units" => opts.voronoi_n_units,
            "prob_core" => opts.voronoi_prob_core,
            "prob_shallow" => opts.voronoi_prob_shallow,
            "prob_deep" => opts.voronoi_prob_deep,
            "min_res_core_km" => opts.voronoi_min_res_core_km,
            "min_res_shallow_km" => opts.voronoi_min_res_shallow_km,
            "min_res_deep_km" => opts.voronoi_min_res_deep_km,
            "core_depth_min" => opts.voronoi_core_depth_min,
            "core_depth_max" => opts.voronoi_core_depth_max,
            "shallow_depth_min" => opts.voronoi_shallow_depth_min,
            "shallow_depth_max" => opts.voronoi_shallow_depth_max,
            "deep_depth_min" => opts.voronoi_deep_depth_min,
            "deep_depth_max" => opts.voronoi_deep_depth_max,
            "slope_weighting" => opts.voronoi_slope_weighting,
            "slope_factor" => opts.voronoi_slope_factor
        ),
        "visualization" => Dict{String, Any}(
            "interactive_map" => opts.interactive_map,
            "animate_hydro" => opts.animate_hydro,
            "anim_variable" => string(opts.anim_variable),
            "anim_fps" => opts.anim_fps,
            "anim_format" => opts.anim_format,
            "anim_depth" => opts.anim_depth,
            "anim_overlay_particles" => opts.anim_overlay_particles,
            "anim_output_path" => opts.anim_output_path
        ),
        "runtype" => Dict{String, Any}(
            "use_gpu" => opts.use_gpu,
            "fallback_to_cpu" => opts.fallback_to_cpu,
            "seed" => opts.seed
        )
    )
end

# ─────────────────────────────────────────────────────────────────────────────
# Formally Decoupled Configuration Architecture
# ─────────────────────────────────────────────────────────────────────────────


"""
    to_hydrodynamic_config(opts::HydrodynamicOptions; kwargs...) -> HydrodynamicConfig

Extract and convert a `HydrodynamicOptions` instance into a pure `HydrodynamicConfig`.
"""
function to_hydrodynamic_config(opts::HydrodynamicOptions; kwargs...)::HydrodynamicConfig
    d = Dict{Symbol, Any}(
        :domain_lon               => opts.domain_lon,
        :domain_lat               => opts.domain_lat,
        :domain_z                 => opts.domain_z,
        :grid_size                => opts.grid_size,
        :grid_orientation         => opts.grid_orientation,
        :rotation_azimuth         => opts.rotation_azimuth,
        :center_lon               => opts.center_lon,
        :center_lat               => opts.center_lat,
        :along_shelf_extent_km    => opts.along_shelf_extent_km,
        :cross_shelf_extent_km    => opts.cross_shelf_extent_km,
        :vertical_stretching_mode => opts.vertical_stretching_mode,
        :nz_above                 => opts.nz_above,
        :vertical_break_depth     => opts.vertical_break_depth,
        :vertical_grid_file       => opts.vertical_grid_file,
        :vertical_depths          => opts.vertical_depths,
        :bathymetry_source        => Symbol(opts.bathy_source),
        :bathy_source              => opts.bathy_source,
        :coastline_source          => opts.coastline_source,
        :wind_source               => opts.wind_source,
        :wind_time_iso             => opts.wind_time_iso,
        :inshore_depth            => opts.inshore_depth,
        :shelf_slope              => opts.shelf_slope,
        :mask_bay_of_fundy        => opts.mask_bay_of_fundy,
        :atmospheric_source       => opts.atmospheric_source,
        :drag_formulation         => :garratt_1977,
        :bulk_heat_flux           => true,
        :climatology              => false,
        :mhw_temp_anomaly         => opts.scenario == :mhw ? 3.5 : 0.0,
        :ocean_boundary_source    => opts.ocean_boundary_source,
        :hydrography_source       => opts.hydrography_source,
        :hydrography_month        => opts.hydrography_month,
        :obc_type                 => opts.obc_type,
        :sponge_layer_width       => opts.sponge_layer_width,
        :sponge_timescale         => opts.sponge_timescale,
        :enable_tides             => opts.enable_tides,
        :tides_source             => Symbol(opts.tides_source),
        :tidal_constituents       => opts.tidal_constituents,
        :tidal_u_amp              => opts.tidal_u_amp,
        :tidal_v_amp              => opts.tidal_v_amp,
        :cd_drag                  => 0.0025,
        :bottom_drag              => 0.0001,
        :bbl_mixing_closure       => :tke,
        :sim_duration_seconds     => opts.sim_duration,
        :sim_dt_seconds           => opts.sim_dt,
        :adaptive_cfl             => opts.adaptive_cfl,
        :target_cfl               => opts.target_cfl,
        :target_wave_cfl          => 0.20,
         :max_dt_seconds           => opts.max_dt,
         # Was hardcoded to 2.0, so the second options struct (built from this dict at the call
         # site below) always carried 2.0 regardless of what the TOML asked for. Use the value
         # that was actually parsed.
         :min_dt_seconds           => opts.min_dt_seconds,
         :coriolis_latitude        => 44.5,
         :divergence_limit         => 20.0,
         :output_path              => opts.output_path,
         :storage_backend          => opts.storage_backend,
         :output_schedule_seconds  => opts.output_schedule_seconds,
         :enable_checkpoint        => opts.enable_checkpoint,
         :checkpoint_prefix        => opts.checkpoint_prefix,
         :checkpoint_schedule      => opts.checkpoint_schedule,
         :checkpoint_dir           => opts.checkpoint_dir,
         :checkpoint_cleanup       => opts.checkpoint_cleanup,
         :auto_restart             => opts.auto_restart,
         :use_gpu                  => opts.use_gpu,
         :fallback_to_cpu          => opts.fallback_to_cpu
    )
    for (k, v) in kwargs
        d[k] = v
    end

    return HydrodynamicConfig(
        d[:domain_lon], d[:domain_lat], d[:domain_z], d[:grid_size],
        Symbol(d[:grid_orientation]), Float64(d[:rotation_azimuth]),
        Float64(d[:center_lon]), Float64(d[:center_lat]),
        Float64(d[:along_shelf_extent_km]), Float64(d[:cross_shelf_extent_km]),
        Symbol(d[:vertical_stretching_mode]),
        Int(d[:nz_above]),
        abs(Float64(d[:vertical_break_depth])),
        String(d[:vertical_grid_file]),
        Float64[Float64(x) for x in d[:vertical_depths]],
        Symbol(d[:bathymetry_source]),
        String(d[:bathy_source]),
        String(d[:coastline_source]),
        String(d[:wind_source]),
        String(d[:wind_time_iso]),
        abs(Float64(d[:inshore_depth])),
        abs(Float64(d[:shelf_slope])),
        Symbol(d[:atmospheric_source]),
        Symbol(d[:drag_formulation]), Bool(d[:bulk_heat_flux]),
        Bool(d[:climatology]), Float64(d[:mhw_temp_anomaly]),
        Symbol(d[:ocean_boundary_source]), Symbol(d[:obc_type]),
        Float64(d[:sponge_layer_width]), Float64(d[:sponge_timescale]),
        Bool(d[:enable_tides]), Symbol(d[:tides_source]),
        Vector{Symbol}(d[:tidal_constituents]), Float64(d[:tidal_u_amp]),
        Float64(d[:tidal_v_amp]), Float64(d[:cd_drag]),
        Float64(d[:bottom_drag]), Symbol(d[:bbl_mixing_closure]),
        Float64(d[:sim_duration_seconds]), Float64(d[:sim_dt_seconds]),
        Bool(d[:adaptive_cfl]), Float64(d[:target_cfl]),
        Float64(d[:target_wave_cfl]), Float64(d[:max_dt_seconds]),
        Float64(d[:min_dt_seconds]), Float64(d[:coriolis_latitude]),
         Float64(d[:divergence_limit]), String(d[:output_path]),
         String(d[:storage_backend]), Float64(d[:output_schedule_seconds]),
         Bool(d[:enable_checkpoint]), String(d[:checkpoint_prefix]),
         String(d[:checkpoint_schedule]), String(d[:checkpoint_dir]),
         Bool(d[:checkpoint_cleanup]), Bool(d[:auto_restart]),
         Bool(d[:use_gpu]), Bool(d[:fallback_to_cpu])
    )
end

