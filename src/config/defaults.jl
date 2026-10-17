"""
    defaults.jl

Configuration defaults and study definitions.

Where a configuration file is found, what a bare configuration name resolves to,
the domain each named study covers, and the fully populated default dictionary.
These are all *values*, so they live together; nothing here reads a file.
"""


"""
    find_default_config_path() -> String

Locate the configuration file in the project workspace, or return `""` when there is none.

Only project-root locations are searched. A configuration is an input the operator supplies;
it is never discovered inside a previous run's containerised `inputs/` directory, and it is
never guessed at. In particular there is no fallback to `configs/default.toml`: a run that
silently adopts a different scenario than the one intended produces output that looks
complete and answers a question nobody asked. Callers must fail loudly instead, which is what
`load_configuration` does with an empty path.

To run, pass one explicitly:

    julia --project=. GeoOceanModelRun.jl --config=configs/scotian_shelf.toml

or place `ParticleTracking.toml` at the project root.
"""
function find_default_config_path()::String
    for p in ("ParticleTracking.toml", "ParticleTracking.config")
        isfile(p) && return p
    end
    ""
end

const NO_CONFIG_ADVICE = """
No configuration file was found.

A configuration is required, not optional. It names the datasets to ingest, the domain and
grid, the forcing sources, and the biology parameters, and those choices are what a run is.
Without one the package would have to invent all of it, and the resulting output would carry
no statement of what it actually simulated -- it would be indistinguishable from a run that
was configured deliberately.

Available configurations in this project:
  configs/default.toml       Cape Breton : small box, low resolution, 3 days,
                             keyless data (etopo2022, WOA23, open_meteo). For demos.
  configs/scotian_shelf.toml  The full snow-crab Scotian Shelf domain, nwa_bathymetry
                             and GLORYS12V1 (Copernicus Marine account required).

Run with one of:

    julia --project=. GeoOceanModelRun.jl --config=configs/scotian_shelf.toml

or copy one to ParticleTracking.toml at the project root, which the loader picks up
automatically. For a study of your own, copy the closest config and edit the sections the
run depends on: [data] for datasets, [domain] and [grid] for the box and resolution,
[tides] for tidal forcing, and [biology] for larval parameters."""

"""
    study_domain_ranges(cfg) -> NamedTuple

Geographic bounds of the *study domain* — the computational/analysis grid — read from a
parsed configuration.

Authoritative source is the TOML `[domain]` section (`lon_min`, `lon_max`, `lat_min`,
`lat_max`). No domain is compiled into the package: the configuration file is the single
source of truth, so a run against a different shelf only requires editing TOML.

# Inputs
- `cfg`: Parsed configuration `AbstractDict` (e.g. from [`load_configuration`](@ref)).

# Outputs
- `NamedTuple`: `(lon = (lon_min, lon_max), lat = (lat_min, lat_max))`, in degrees.
"""
function study_domain_ranges(cfg)::NamedTuple
    d = get(cfg, "domain", Dict{String, Any}())
    return (
        lon = (Float64(get(d, "lon_min", -68.0)), Float64(get(d, "lon_max", -57.0))),
        lat = (Float64(get(d, "lat_min", 42.0)), Float64(get(d, "lat_max", 47.5)))
    )
end

"""
    embedding_domain_ranges(cfg) -> NamedTuple

Geographic bounds of the *embedding domain* — the larger region from which open boundary
and atmospheric forcing data are extracted — read from a parsed configuration.

Authoritative source is the TOML `[boundaries]` section (`embedding_lon_min`,
`embedding_lon_max`, `embedding_lat_min`, `embedding_lat_max`). The study domain nests
inside this region.

# Inputs
- `cfg`: Parsed configuration `AbstractDict`.

# Outputs
- `NamedTuple`: `(lon = (lon_min, lon_max), lat = (lat_min, lat_max))`, in degrees.
"""
function embedding_domain_ranges(cfg)::NamedTuple
    b = get(cfg, "boundaries", Dict{String, Any}())
    return (
        lon = (Float64(get(b, "embedding_lon_min", -71.0)), Float64(get(b, "embedding_lon_max", -53.0))),
        lat = (Float64(get(b, "embedding_lat_min", 40.0)), Float64(get(b, "embedding_lat_max", 48.5)))
    )
end

"""
    resolve_config_name(config_file::AbstractString = "") -> String

Extract the base configuration name from a config file path, or return
the default configuration name (`"ParticleTracking"`) if unspecified or empty.

# Inputs
- `config_file::AbstractString`: Path to a `.toml` file or empty string.

# Outputs
- `String`: Clean configuration identifier without directory or extension.
"""
function resolve_config_name(config_file::AbstractString = "")::String
    cfg_path = isempty(strip(config_file)) ? find_default_config_path() : String(config_file)
    base = basename(cfg_path)
    stem = splitext(base)[1]
    return isempty(stem) ? "ParticleTracking" : stem
end


"""
    get_default_configuration() -> Dict{String, Any}

Generate a comprehensive dictionary of all default parameters across all modeling domains.
"""
function get_default_configuration()::Dict{String, Any}
    return Dict{String, Any}(
        "domain" => Dict{String, Any}(
            "lon_min" => -71.0,
            "lon_max" => -53.0,
            "lat_min" => 40.0,
            "lat_max" => 48.5,
            "z_min" => -3500.0,
            "z_max" => 0.0,
            "buffer_km" => 100.0
        ),
        "grid" => Dict{String, Any}(
            "nx" => 50,
            "ny" => 50,
            "nz" => 10,
            "vertical_stretching_mode" => "oceananigans",
            "vertical_grid_file" => ""
        ),
        "data" => Dict{String, Any}(
            "bathy_source" => "etopo2022",
            "coastline_source" => "natural_earth_10m",
            "wind_source" => "open_meteo",
            "hydrography_source" => "woa23",
            "wind_time_iso" => "2023-06-01T00:00:00Z"
        ),
        "bathymetry" => Dict{String, Any}(
            "source" => "numerical_earth",
            "resolution_arcsec" => 15,
            "inshore_depth" => -20.0,
            "shelf_slope" => 500.0
        ),
        "boundaries" => Dict{String, Any}(
            "method" => "relaxation",
            "ocean_boundary_source" => "glorys12v1",
            "obc_type" => "flather_chapman",
            "sponge_layer_width" => 0.35,
            "sponge_timescale" => 3600.0,
            "u_inflow" => -0.15,
            "v_inflow" => 0.05
        ),
        "atmosphere" => Dict{String, Any}(
            "source" => "era5",
            "drag_formulation" => "garratt_1977",
            "bulk_heat_flux" => true
        ),
        "tides" => Dict{String, Any}(
            "enable_tides" => true,
            "constituents" => ["M2"],
            "tidal_u_amp" => 0.25,
            "tidal_v_amp" => 0.12,
            "tidal_period" => 44712.0,
            "tidal_phase" => 0.0
        ),
        "climate" => Dict{String, Any}(
            "scenario" => "ssp245",
            "projection_year" => 2050,
            "baseline_year" => 2015,
            "horizon_year" => 2050
        ),
        "hydrodynamics" => Dict{String, Any}(
            "sim_duration_hours" => 12.0,
            "sim_dt_seconds" => 120.0,
            "adaptive_cfl" => true,
            "target_cfl" => 0.2,
            "divergence_velocity_limit" => 20.0,
            "coriolis_latitude" => 44.5,
            "surface_heat_flux" => 50.0,
            "max_flow_snapshots" => 0,
            "allow_analytical_fallback" => false,
            "turbulence_closure" => "nemotke",
            "enable_sea_ice" => true,
            "enable_oxygen" => true
        ),
        "biology" => Dict{String, Any}(
            "n_particles" => 100,
            "track_duration_days" => 5.0,
            "track_dt_seconds" => 300.0,
            "min_seabed_depth" => 100.0,
            "buffer_km" => 100.0,
            "diffusivity_h" => 10.0,
            "diffusivity_v" => 1e-4,
            "release_depth_mode" => "bottom",
            "bottom_release_offset" => [0.5, 3.0],
            "enable_initial_ascent" => true,
            "ascent_speed" => 0.010,
            "ascent_target_depth" => -10.0
        ),
        "dvm" => Dict{String, Any}(
            "enable_dvm" => true,
            "megalopa_day_depth" => -120.0,
            "megalopa_night_depth" => -60.0,
            "zoea2_depth_factor" => 1.2,
            "megalopa_swim_factor" => 1.5
        ),
        "molting_and_settlement" => Dict{String, Any}(
            "enable_molting" => true,
            "t_base" => 0.0,
            "dd_zoea1_to_zoea2" => 150.0,
            "dd_zoea2_to_megalopa" => 310.0,
            "dd_megalopa_to_settle" => 510.0,
            "mortality_base" => 0.02,
            "mortality_thermal_threshold" => 10.0,
            "mortality_thermal_sensitivity" => 0.015,
            "settlement_min_depth" => -250.0,
            "settlement_max_depth" => -50.0,
            "settlement_max_temp" => 6.0
        ),
         "storage" => Dict{String, Any}(
             "output_path" => joinpath("outputs", "particle_tracking.jld2"),
             "storage_backend" => "zarr",
             "enable_checkpoint" => true,
             "checkpoint_prefix" => "checkpoint_ParticleTracking",
             "checkpoint_schedule_seconds" => 21600.0,
             "checkpoint_dir" => "outputs/checkpoints",
             "checkpoint_cleanup" => true
         ),
        "tessellation" => Dict{String, Any}(
            "enable_voronoi" => false,
            "n_units" => 5000,
            "prob_core" => 0.8,
            "prob_shallow" => 0.1,
            "prob_deep" => 0.1,
            "min_res_core_km" => 1.5,
            "min_res_shallow_km" => 5.0,
            "min_res_deep_km" => 10.0,
            # Depth band edges as positive depths below the surface. Core is the mid-shelf
            # nursery, shallow the nearshore strip, deep everything below the shelf break.
            "core_depth_min" => 50.0,
            "core_depth_max" => 350.0,
            "shallow_depth_min" => 0.0,
            "shallow_depth_max" => 50.0,
            "deep_depth_min" => 350.0,
            "deep_depth_max" => 6000.0,
            "slope_weighting" => true,
            "slope_factor" => 15.0
        ),
        "hardware" => Dict{String, Any}(
            "use_gpu" => false,
            # Off by default, and deliberately. A GPU request that cannot be honoured must
            # fail loudly: silently running the CPU path instead would leave
            # `resolved_config.toml` recording `use_gpu = true` for a run that never touched
            # a GPU, which is the same silent lie the provenance file exists to prevent. Set
            # it to true only on a machine where degrading is genuinely acceptable.
            "fallback_to_cpu" => false
        ),
        "visualization" => Dict{String, Any}(
            "interactive_map" => true,
            "animate_hydro" => false,
            "anim_variable" => "dashboard",
            "anim_fps" => 10,
            "anim_format" => "mp4",
            "anim_depth" => -2.5,
            "anim_overlay_particles" => false,
            "anim_output_path" => "",
            "title" => "Regional Marine Lagrangian Particle Tracking & Dispersion"
        ),
        "paths" => Dict{String, Any}(
            "seed" => 42
        )
    )
end