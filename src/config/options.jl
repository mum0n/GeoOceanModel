"""
    options.jl

Runtime and configuration types.

Every type the configuration system defines: the flat `HydrodynamicOptions` the
rest of the package consumes, with its inner constructors, and the three pure
config structs that are the parsed view of one run. A reader looking for a type
finds all of them here.
"""


"""
    HydrodynamicOptions

Runtime parameter specification for regional hydrodynamic circulation,
tidal harmonics, CMIP6 climate anomalies, and Lagrangian snow crab
(Chionoecetes opilio) larval transport modeling.

# Fields
- `domain_lon::Tuple{Float64, Float64}`: Longitude bounding box (min_lon, max_lon) in °E.
- `domain_lat::Tuple{Float64, Float64}`: Latitude bounding box (min_lat, max_lat) in °N.
- `domain_z::Tuple{Float64, Float64}`: Vertical depth range (z_min, z_max) in meters.
- `grid_size::Tuple{Int, Int, Int}`: Grid cell counts (Nx, Ny, Nz).
- `grid_orientation::Symbol`: Grid orientation (:standard or :rotated).
- `rotation_azimuth::Float64`: Bearing along shelf strike in degrees clockwise from North.
- `center_lon::Float64`: Central longitude in °E for rotated grid coordinates.
- `center_lat::Float64`: Central latitude in °N for rotated grid coordinates.
- `along_shelf_extent_km::Float64`: Domain dimension along shelf strike in kilometers.
- `cross_shelf_extent_km::Float64`: Domain dimension across shelf strike in kilometers.
- `bathy_source::String`: Which bathymetry product to ingest, read verbatim from
  `[data] bathy_source` (for example `"etopo2022"`, `"synthetic"`). There is no separate
  real/synthetic switch: acquisition always reads the named product, and falls back to analytic
  fields with a warning if a download fails.
- `enable_tides::Bool`: Whether to include astronomical tidal body forcing (M2).
- `tidal_u_amp::Float64`: Zonal tidal velocity amplitude in m/s.
- `tidal_v_amp::Float64`: Meridional tidal velocity amplitude in m/s.
- `scenario::Symbol`: Climate scenario (:historical, :ssp126, :ssp245, :ssp585, :mhw).
- `projection_year::Int`: Target climate scenario year (e.g. 2050).
- `sim_dt::Float64`: Initial hydrodynamic integration time step in seconds.
- `sim_duration::Float64`: Total hydrodynamic simulation duration in seconds.
- `adaptive_cfl::Bool`: Whether to adaptively modulate hydrodynamic time stepping.
- `target_cfl::Float64`: Target CFL limit for the numerical wizard.
- `n_particles::Int`: Number of Lagrangian larval particles to track.
- `track_duration::Float64`: Total particle tracking drift duration in seconds.
- `track_dt::Float64`: Particle integration step in seconds.
- `diffusivity_h::Float64`: Horizontal turbulent eddy diffusivity in m^2/s.
- `diffusivity_v::Float64`: Vertical turbulent eddy diffusivity in m^2/s.
- `enable_dvm::Bool`: Whether larvae undergo active Diel Vertical Migration.
- `enable_molting::Bool`: Whether degree-day accumulation triggers stage molting.
- `min_seabed_depth::Float64`: Minimum bathymetric water depth for larval release (meters).
- `buffer_km::Float64`: Spatial buffer distance in km (default 100.0 km) extending beyond CFAs.
- `release_depth_mode::Symbol`: Vertical release mode (:bottom, :range, :surface).
- `bottom_release_offset::Tuple{Float64, Float64}`: Elevation range above seabed (meters).
- `enable_initial_ascent::Bool`: Whether larvae perform active post-hatch vertical ascent.
- `ascent_speed::Float64`: Directed upward swimming speed during ascent (m/s).
- `ascent_target_depth::Float64`: Target epipelagic depth for ascent completion (meters).
- `use_gpu::Bool`: Whether to run hydrodynamic equations on NVIDIA CUDA GPU.
- `fallback_to_cpu::Bool`: Whether a GPU request that cannot be honoured may degrade to CPU.
  Defaults to `false`, so the run fails loudly instead of quietly running somewhere other than
  where `use_gpu` said it would.
- `interactive_map::Bool`: Whether to export interactive HTML5 Leaflet map.
- `output_path::AbstractString`: Base path for simulation outputs. Extension determines format:
  - `.zarr` or directory: Zarr storage via GeoData
  - `.jld2`: Direct JLD2 file for hydrodynamic outputs
  - No extension: treated as directory for JLD2 files (legacy behavior)
- `storage_backend::String`: Storage backend for larval data ("zarr" or "geoparquet").
- `config_file::String`: Source configuration file path.
- `bathy_source::String`: Which bathymetry product to ingest; the run always reads the product
  named here rather than choosing between a real and a synthetic mode.
- `inshore_depth::Float64`: Water depth at the landward edge of the domain in meters
  (positive depth, e.g. 20.0 m).
- `shelf_slope::Float64`: Total seabed relief / rise across the shelf in meters
  (positive depth scale, e.g. 180.0 m).
- `output_schedule_seconds::Float64`: Field-output cadence. Independent of
  `checkpoint_schedule`, which is the state-checkpoint cadence; the two may legitimately differ.
- `input_dir::String`: Directory for raw and processed bathymetry/wind files.
- `seed::Int`: Random number generator seed.
- `hydro_model_file::String`: Target hydrodynamic JLD2 checkpoint path (input/output).
- `hydro_only::Bool`: Run hydrodynamics only and persist to JLD2 checkpoint.
- `track_only::Bool`: Track larvae only using flow fields from hydro_model_file.
- `reuse_hydro::Bool`: Reuse existing hydro_model_file if present on disk; else run.
- `enable_checkpoint::Bool`: Whether to attach Oceananigans Checkpointer for state serialization.
- `checkpoint_schedule::Float64`: State checkpoint frequency in seconds (0.0 = match output schedule).
- `checkpoint_dir::String`: Directory where state checkpoints are archived.
- `checkpoint_cleanup::Bool`: Whether older intermediate checkpoints are automatically deleted.
- `auto_restart::Bool`: Automatically detect and pick up from existing checkpoints if available.
- `run_id::String`: Unique cohort identifier for Zarr persistence and figures.
- `vertical_stretching_mode::Symbol`: Vertical grid stretching mode (:tanh, :uniform,
  :two_segment, or :oceananigans).
- `vertical_break_depth::Float64`: Shelf-slope break depth separating upper refined
  stratum from deep basin in meters (positive depth, e.g. 200.0 or 400.0 m).
- `vertical_grid_file::String`: Path to CSV file specifying vertical layer boundary faces.
- `atmospheric_source::Symbol`: Atmospheric forcing provider (:era5, :synthetic, :climatology).
- `ocean_boundary_source::Symbol`: Open boundary hydrographic data provider (:glorys12v1, :synthetic).
  See the field comments for the full accepted set.
- `obc_type::Symbol`: Lateral open boundary formulation (:flather_chapman, :radiation, :clamped).
- `enable_voronoi::Bool`: Whether to compute multi-resolution Voronoi tessellation analysis.
- `voronoi_n_units::Int`: Number of Voronoi units/centroids to generate across strata.
- `voronoi_prob_core::Float64`: Sampling probability weight for core depth stratum (50 to 350 m).
- `voronoi_prob_shallow::Float64`: Sampling probability weight for shallow stratum (0 to 50 m).
- `voronoi_prob_deep::Float64`: Sampling probability weight for deep stratum (> 350 m).
- `voronoi_min_res_core_km::Float64`: Minimum Poisson-disc separation distance in core zone (km).
- `voronoi_min_res_shallow_km::Float64`: Minimum separation distance in shallow zone (km).
- `voronoi_min_res_deep_km::Float64`: Minimum separation distance in deep zone (km).
- `voronoi_core_depth_min`/`_max`, `voronoi_shallow_depth_min`/`_max`,
  `voronoi_deep_depth_min`/`_max`::Float64: The edges of the three depth strata, as positive
  depths below the surface. These decide which seabed area each stratum samples, so they change
  the connectivity result, not just its resolution. The sign is applied when the bands reach the
  tessellator, which bins on signed `z`.
- `voronoi_slope_weighting::Bool`: Whether minimum spacing is tightened on steep bathymetry.
- `voronoi_slope_factor::Float64`: How aggressively spacing tightens with slope; only used when
  `voronoi_slope_weighting` is true.
"""
struct HydrodynamicOptions
    domain_lon               :: Tuple{Float64, Float64}
    domain_lat               :: Tuple{Float64, Float64}
    domain_z                 :: Tuple{Float64, Float64}
    grid_size                :: Tuple{Int, Int, Int}
    grid_orientation         :: Symbol
    rotation_azimuth         :: Float64
    center_lon               :: Float64
    center_lat               :: Float64
    along_shelf_extent_km    :: Float64
    cross_shelf_extent_km    :: Float64
    bathy_source             :: String
    coastline_source         :: String
    wind_source              :: String
    wind_time_iso            :: String
    inshore_depth            :: Float64
    shelf_slope              :: Float64
    enable_tides             :: Bool
    tides_source             :: String
    tidal_constituents       :: Vector{Symbol}
    tidal_u_amp              :: Float64
    tidal_v_amp              :: Float64
    s2_u_amp                 :: Float64
    s2_v_amp                 :: Float64
    scenario                 :: Symbol
    projection_year          :: Int
    sim_dt                   :: Float64
    sim_duration             :: Float64
    min_dt_seconds           :: Float64
    adaptive_cfl             :: Bool
    target_cfl               :: Float64
    surface_heat_flux        :: Float64
    bulk_heat_flux           :: Bool
    sst_damping_coefficient  :: Float64
    sst_target               :: Float64
    spinup_hours             :: Float64
    divergence_velocity_limit :: Float64
    max_flow_snapshots       :: Int
    allow_analytical_fallback :: Bool
    max_current_speed        :: Union{Nothing, Float64}
    n_particles              :: Int
    track_duration           :: Float64
    track_dt                 :: Float64
    diffusivity_h            :: Float64
    diffusivity_v            :: Float64
    enable_dvm               :: Bool
    enable_molting           :: Bool
    min_seabed_depth         :: Float64
    buffer_km                :: Float64
    release_depth_mode       :: Symbol
    bottom_release_offset    :: Tuple{Float64, Float64}
     enable_initial_ascent    :: Bool
     ascent_speed             :: Float64
     ascent_target_depth      :: Float64
     use_gpu                  :: Bool
     fallback_to_cpu          :: Bool
     interactive_map          :: Bool
     config_file              :: String
     seed                     :: Int
     hydro_model_file         :: String
     hydro_only               :: Bool
     track_only               :: Bool
     reuse_hydro              :: Bool
     enable_checkpoint        :: Bool
     checkpoint_prefix        :: String
     checkpoint_schedule      :: Float64
     output_schedule_seconds  :: Float64
     max_save_depth           :: Float64
     output_path              :: AbstractString
     storage_backend          :: String
     enable_zarr              :: Bool
     zarr_path                :: String
     output_dir               :: String
     input_dir                :: String
    checkpoint_dir           :: String
    checkpoint_cleanup       :: Bool
    auto_restart             :: Bool
    overwrite                :: Bool
    run_id                   :: String
    vertical_stretching_mode :: Symbol
    nz_above :: Int
    vertical_break_depth :: Float64
    vertical_grid_file       :: String
    vertical_depths          :: Vector{Float64}
    atmospheric_source :: Symbol
    ocean_boundary_source :: Symbol
    # The larger region the open-boundary data is cut from. These are separate from
    # `domain_lon`/`domain_lat` because the boundary needs the water just *outside* the study
    # area; a subset cut to the study domain would put the interpolation's edge exactly on the
    # sponge. They are fields rather than re-read from the raw config dict because a value
    # fetched by key from somewhere downstream is a value nothing checks.
    embedding_lon :: Tuple{Float64, Float64}
    embedding_lat :: Tuple{Float64, Float64}
    # Where each piece of the model's outside world comes from.
    #
    # The model is a window cut out of the ocean, so three of its edges are not real coastline
    # but imaginary lines. Along those lines the model has to be told what the water is doing
    # just outside: how warm, how salty, and which way it is moving. That is the
    # "lateral boundary" below. If it is left unset, the model still runs, but water can drift
    # in or out of the open edges unchecked, and results that depend on the flow across an
    # edge are then unreliable.
    #
    # The values are stored as Symbols, so a misspelling stays visible in a comparison rather
    # than quietly becoming an empty field. They are read straight from the TOML with no
    # normalisation, so the accepted spellings are exactly as listed.
    #
    #   ocean_boundary_source  -- what sets the imaginary edges
    #     "synthetic"    Nothing is downloaded. The open edges are simply not constrained, and
    #                    the model behaves as if they are out of sight. This is a real choice
    #                    for a run that only studies the interior; it is not a stand-in for
    #                    data, and it writes no file.
    #     "glorys12v1"   A global real-world reconstruction, GLORYS12V1, at 1/12 degree (about
    #                    9 km). Comes from Copernicus Marine, which is free but does require
    #                    registering an account and putting a username and key in a file
    #                    called ~/.copernicusmarine/credentials. Without that file the download
    #                    is rejected.
    #     "hycom"        HYCOM GOFS at the same 1/12 degree (about 9 km), and unlike the
    #                    options below it also carries how fast the water is moving, which is
    #                    the part that matters most for an open edge. No account, no key. Read
    #                    the note on `fetch_hycom_boundary` before relying on it: its data is
    #                    currently not downloadable even though the catalogue works.
    #     "woa23"        A monthly average of the whole ocean, free and no account. But it
    #                    holds only temperature and salt, not movement, so it fixes what the
    #                    water is like but not how much of it flows through the edge.
    #
    #   atmospheric_source  -- what drives the sea surface from the sky
    #     "open_meteo"        Real weather, served for free by an open website. It also
    #                         reports air temperature, pressure, humidity, cloud cover and
    #                         sunlight, which is what the model needs to work out how much
    #                         heat the sea gains or loses each hour. Currently the only choice
    #                         that makes the heat flux change over the run. It gives one
    #                         representative spot rather than a picture of the sky, so the same
    #                         weather is used across the whole domain.
    #     "era5_climatology"  A fixed typical month, repeating. No download. Use it only to
    #                         debug; the sea will warm monotonically and never vary.
    #     "synthetic"         A single fixed number. No download, no variation. Diagnostics only.
    #
    #   tides_source  -- what sets the rise and fall of the tides
    #     "tpxo9_atlas"   Measured tide heights and currents, free and no account. Supplies the
    #                     real size of the M2 and S2 tides, which is what makes the spring-neap
    #                     cycle (big tides every two weeks) come out right.
    #     "tpxo9"         The same product, named more briefly.
    #     "analytic"      Tide shapes made up from a single number you supply. No download, and
    #                     no detail across space, so tides are the same everywhere in the
    #                     domain. Fine for checking the machinery, not for a real tide.
    #
    #   bathy_source  -- where the sea floor shape comes from
    #     "etopo2022"  A global map of the sea floor, free and no account, detailed to about
    #                  450 m. "etopo" and "open" are other spellings of the same thing.
    #
    #   hydrography_source  -- what the water starts out like at t = 0
    #     "woa23"      Average temperature and salt for a chosen month, free and no account.
    #     "synthetic"  A made-up smooth profile, no download. For testing.
    #     "glorys12v1" A real reconstruction; needs the Copernicus account described above.
    #
    #   coastline_source  -- where the outline of the land comes from
    #     "natural_earth_10m"  Very detailed, public domain, no account. "50m" and "110m" are
    #                         coarser outlines for when the detail is not needed.
    #
    # None of these falls back quietly. If a value is not recognised the run stops and says
    # what it could have used instead. That is deliberate: a run that asked for one product
    # and quietly received a different one would have different temperature or flow at its
    # edges, and nothing in the output would reveal it.
    # Source for the 3-D hydrographic (T, S) initial state. This is deliberately
    # separate from `bathy_source`/`ocean_boundary_source`, which govern bathymetry,
    # wind and lateral boundaries: the initial water column is its own choice.
    #   :synthetic — analytic profile from `set_initial_stratification!`
    #   :woa23     — World Ocean Atlas 2023 climatology (keyless, NOAA NCEI)
    #   :glorys12v1— GLORYS12V1 reanalysis (requires copernicusmarine credentials)
    hydrography_source       :: Symbol
    hydrography_month        :: Int      # WOA23 month selector (0 = annual climatology)
    obc_type                 :: Symbol
    enable_voronoi           :: Bool
    voronoi_n_units          :: Int
    voronoi_prob_core        :: Float64
    voronoi_prob_shallow     :: Float64
    voronoi_prob_deep        :: Float64
    voronoi_min_res_core_km  :: Float64
    voronoi_min_res_shallow_km :: Float64
    voronoi_min_res_deep_km  :: Float64
    voronoi_core_depth_min   :: Float64
    voronoi_core_depth_max   :: Float64
    voronoi_shallow_depth_min :: Float64
    voronoi_shallow_depth_max :: Float64
    voronoi_deep_depth_min   :: Float64
    voronoi_deep_depth_max   :: Float64
    voronoi_slope_weighting  :: Bool
    voronoi_slope_factor     :: Float64
    animate_hydro            :: Bool
    anim_variable            :: Symbol
    anim_fps                 :: Int
    anim_format              :: String
    anim_depth               :: Float64
    anim_overlay_particles   :: Bool
    anim_output_path         :: String
    # Lateral boundary relaxation parameters (Price & Aumont 2011)
    boundary_method          :: Symbol   # :relaxation | :none
    sponge_layer_width       :: Float64  # sponge buffer width (degrees)
    sponge_timescale         :: Float64  # relaxation timescale (s)
    tidal_relaxation_timescale :: Float64  # relaxation toward the prescribed tide (s)
    u_inflow                 :: Float64  # reference zonal inflow velocity (m/s)
    v_inflow                 :: Float64  # reference meridional inflow velocity (m/s)
    # Time integration ceiling
    max_dt                   :: Float64  # maximum allowable time step (s)
    # Stochastic biology parameters
    settlement_stochastic    :: Bool     # whether settlement uses probabilistic HSI draw
    cv_molt                  :: Float64  # coefficient of variation for molt thresholds (default 0.25)
    cv_mortality             :: Float64  # coefficient of variation for mortality (default 0.25)
    cv_settlement            :: Float64  # coefficient of variation for settlement HSI (default 0.25)
    mask_bay_of_fundy        :: Bool     # whether Bay of Fundy shallows are masked to land
    active_boundaries        :: Tuple{Vararg{Symbol}} # active boundaries for relaxation
end

function HydrodynamicOptions(;
    domain_lon            :: Tuple{Real, Real} = (-71.0, -53.0),
    domain_lat            :: Tuple{Real, Real} = (40.0, 48.5),
    domain_z              :: Tuple{Real, Real} = (-3500.0, 0.0),
    grid_size             :: Tuple{Int, Int, Int} = (50, 50, 10),
    grid_orientation      :: Symbol = :standard,
    rotation_azimuth      :: Real = 50.0,
    center_lon            :: Real = -63.0,
    center_lat            :: Real = 44.0,
    along_shelf_extent_km :: Real = 850.0,
    cross_shelf_extent_km :: Real = 220.0,
    bathy_source           :: AbstractString = "etopo2022",
    coastline_source       :: AbstractString = "natural_earth_10m",
    wind_source            :: AbstractString = "open_meteo",
    wind_time_iso          :: AbstractString = "2020-06-01T00:00:00Z",
    inshore_depth          :: Real = 20.0,
    shelf_slope            :: Real = 180.0,
    enable_tides          :: Bool = true,
    tides_source          :: AbstractString = "tpxo9",
    tidal_constituents    :: AbstractVector{Symbol} = [:M2, :S2],
    tidal_u_amp           :: Real = 0.25,
    tidal_v_amp           :: Real = 0.12,
    s2_u_amp              :: Real = 0.11,
    s2_v_amp              :: Real = 0.05,
    scenario              :: Symbol = :ssp245,
    projection_year       :: Int = 2050,
    sim_dt                :: Real = 120.0,
    sim_duration          :: Real = 43200.0,
    min_dt_seconds        :: Real = 2.0,
    adaptive_cfl          :: Bool = true,
    target_cfl            :: Real = 0.2,
    surface_heat_flux     :: Real = 50.0,
    bulk_heat_flux        :: Bool = true,
    sst_damping_coefficient :: Real = 0.0,
    sst_target            :: Real = 12.0,
    spinup_hours          :: Real = 720.0,
    divergence_velocity_limit :: Real = 20.0,
    max_flow_snapshots    :: Int = 400,
    allow_analytical_fallback :: Bool = false,
    max_current_speed     :: Union{Nothing, Real} = 3.0,
    n_particles           :: Int = 100,
    track_duration        :: Real = 86400.0 * 5,
    track_dt              :: Real = 300.0,
    diffusivity_h         :: Real = 10.0,
    diffusivity_v         :: Real = 1e-4,
    enable_dvm            :: Bool = true,
    enable_molting        :: Bool = true,
    min_seabed_depth      :: Real = 100.0,
    buffer_km             :: Real = 100.0,
    release_depth_mode    :: Symbol = :bottom,
    bottom_release_offset :: Tuple{Real, Real} = (0.5, 3.0),
    enable_initial_ascent :: Bool = true,
    ascent_speed          :: Real = 0.010,
    ascent_target_depth   :: Real = -10.0,
    use_gpu               :: Bool = false,
    fallback_to_cpu       :: Bool = false,
    interactive_map       :: Bool = true,
    enable_zarr         :: Bool = true,
    zarr_path           :: AbstractString = joinpath("outputs", "particle_tracking.zarr"),
    output_path         :: AbstractString = joinpath("outputs", "particle_tracking.jld2"),
    storage_backend     :: AbstractString = "zarr",
    config_file           :: AbstractString = find_default_config_path(),
    output_dir            :: AbstractString = "outputs",
    input_dir             :: AbstractString = "inputs",
    seed                  :: Int = 42,
    hydro_model_file      :: AbstractString = "",
    hydro_only            :: Bool = false,
    track_only            :: Bool = false,
    reuse_hydro           :: Bool = false,
    enable_checkpoint     :: Bool = true,
    checkpoint_prefix     :: AbstractString = "",
    checkpoint_schedule   :: Real = 0.0,
    output_schedule_seconds :: Real = 21600.0,
    max_save_depth        :: Real = Inf,
    checkpoint_dir        :: AbstractString = "",
    checkpoint_cleanup    :: Bool = true,
    auto_restart             :: Bool = true,
    overwrite                :: Bool = false,
    run_id                   :: AbstractString = "",
    vertical_stretching_mode :: Symbol = :oceananigans,
    nz_above :: Int = 0,
    vertical_break_depth :: Real = 400.0,
    vertical_grid_file       :: AbstractString = "",
    vertical_depths          :: AbstractVector{<:Real} = Float64[],
    atmospheric_source       :: Symbol = :era5,
    ocean_boundary_source    :: Symbol = :glorys12v1,
    embedding_lon            :: Tuple{Float64, Float64} = (-71.0, -53.0),
    embedding_lat            :: Tuple{Float64, Float64} = (40.0, 48.5),
    hydrography_source       :: Symbol = :synthetic,
    hydrography_month        :: Int     = 0,
    obc_type                 :: Symbol = :flather_chapman,
    enable_voronoi           :: Bool = false,
    voronoi_n_units          :: Int = 5000,
    voronoi_prob_core        :: Real = 0.8,
    voronoi_prob_shallow     :: Real = 0.1,
    voronoi_prob_deep        :: Real = 0.1,
    voronoi_min_res_core_km  :: Real = 1.5,
    voronoi_min_res_shallow_km :: Real = 5.0,
    voronoi_min_res_deep_km  :: Real = 10.0,
    voronoi_core_depth_min   :: Real = 50.0,
    voronoi_core_depth_max   :: Real = 350.0,
    voronoi_shallow_depth_min :: Real = 0.0,
    voronoi_shallow_depth_max :: Real = 50.0,
    voronoi_deep_depth_min   :: Real = 350.0,
    voronoi_deep_depth_max   :: Real = 6000.0,
    voronoi_slope_weighting  :: Bool = true,
    voronoi_slope_factor     :: Real = 15.0,
    animate_hydro            :: Bool = false,
    anim_variable            :: Symbol = :dashboard,
    anim_fps                 :: Int = 10,
    anim_format              :: AbstractString = "mp4",
    anim_depth               :: Real = -2.5,
    anim_overlay_particles   :: Bool = false,
    anim_output_path         :: AbstractString = "",
    boundary_method          :: Symbol = :relaxation,
    sponge_layer_width       :: Real = 0.35,
    sponge_timescale         :: Real = 3600.0,
    tidal_relaxation_timescale :: Real = 3600.0,
    u_inflow                 :: Real = -0.15,
    v_inflow                 :: Real = 0.05,
    max_dt                   :: Real = 600.0,
    settlement_stochastic    :: Bool = true,
    cv_molt                  :: Real = 0.25,
    cv_mortality             :: Real = 0.25,
    cv_settlement            :: Real = 0.25,
    mask_bay_of_fundy        :: Bool = false,
    active_boundaries                = (:east,)
)
    resolved_cp_prefix = if !isempty(strip(checkpoint_prefix)) && checkpoint_prefix != "checkpoint"
        String(checkpoint_prefix)
    else
        cfg_name = resolve_config_name(config_file)
        "checkpoint_$(cfg_name)"
    end

    # The open-boundary embedding region arrives as the `embedding_lon`/`embedding_lat`
    # keywords rather than being re-derived from a config dict here: this constructor takes
    # individual keywords and has no parsed `cfg` to read. `configuration_to_options`, which
    # does have the parsed TOML, is what supplies them -- so the value has exactly one path
    # from the file into the struct.

    return HydrodynamicOptions(
        (Float64(domain_lon[1]), Float64(domain_lon[2])),
        (Float64(domain_lat[1]), Float64(domain_lat[2])),
        (Float64(domain_z[1]), Float64(domain_z[2])),
        grid_size,
        Symbol(lowercase(string(grid_orientation))),
        Float64(rotation_azimuth),
        Float64(center_lon),
        Float64(center_lat),
        Float64(along_shelf_extent_km),
        Float64(cross_shelf_extent_km),
        bathy_source,
        coastline_source,
        wind_source,
        wind_time_iso,
        abs(Float64(inshore_depth)),
        abs(Float64(shelf_slope)),
        enable_tides,
        tides_source,
        tidal_constituents,
        Float64(tidal_u_amp),
        Float64(tidal_v_amp),
        Float64(s2_u_amp),
        Float64(s2_v_amp),
        scenario,
        projection_year,
        Float64(sim_dt),
        Float64(sim_duration),
        Float64(min_dt_seconds),
        adaptive_cfl,
        Float64(target_cfl),
        Float64(surface_heat_flux),
        bulk_heat_flux,
        Float64(sst_damping_coefficient),
        Float64(sst_target),
        Float64(spinup_hours),
        Float64(divergence_velocity_limit),
        Int(max_flow_snapshots),
        allow_analytical_fallback,
        (max_current_speed isa Real && max_current_speed > 0) ? Float64(max_current_speed) : nothing,
        n_particles,
        Float64(track_duration),
        Float64(track_dt),
        Float64(diffusivity_h),
        Float64(diffusivity_v),
        enable_dvm,
        enable_molting,
        Float64(min_seabed_depth),
        Float64(buffer_km),
        release_depth_mode,
        (Float64(bottom_release_offset[1]), Float64(bottom_release_offset[2])),
        enable_initial_ascent,
        Float64(ascent_speed),
        Float64(ascent_target_depth),
        use_gpu,
        fallback_to_cpu,
        interactive_map,
        String(config_file),
        seed,
        String(hydro_model_file),
        hydro_only,
        track_only,
        reuse_hydro,
        enable_checkpoint,
        resolved_cp_prefix,
        Float64(checkpoint_schedule),
        Float64(output_schedule_seconds),
        Float64(max_save_depth),
        String(output_path),
        String(storage_backend),
        enable_zarr,
        String(zarr_path),
        String(output_dir),
        String(input_dir),
        String(checkpoint_dir),
        checkpoint_cleanup,
        auto_restart,
        overwrite,
        String(run_id),
        vertical_stretching_mode,
        Int(nz_above),
        abs(Float64(vertical_break_depth)),
        String(vertical_grid_file),
        Float64[Float64(d) for d in vertical_depths],
        atmospheric_source,
        ocean_boundary_source,
        embedding_lon,
        embedding_lat,
        hydrography_source,
        hydrography_month,
        obc_type,
        enable_voronoi,
        voronoi_n_units,
        Float64(voronoi_prob_core),
        Float64(voronoi_prob_shallow),
        Float64(voronoi_prob_deep),
        Float64(voronoi_min_res_core_km),
        Float64(voronoi_min_res_shallow_km),
        Float64(voronoi_min_res_deep_km),
        Float64(voronoi_core_depth_min),
        Float64(voronoi_core_depth_max),
        Float64(voronoi_shallow_depth_min),
        Float64(voronoi_shallow_depth_max),
        Float64(voronoi_deep_depth_min),
        Float64(voronoi_deep_depth_max),
        voronoi_slope_weighting,
        Float64(voronoi_slope_factor),
        animate_hydro,
        anim_variable,
        anim_fps,
        String(anim_format),
        Float64(anim_depth),
        anim_overlay_particles,
        String(anim_output_path),
        boundary_method,
        Float64(sponge_layer_width),
        Float64(sponge_timescale),
        Float64(tidal_relaxation_timescale),
        Float64(u_inflow),
        Float64(v_inflow),
        Float64(max_dt),
        settlement_stochastic,
        Float64(cv_molt),
        Float64(cv_mortality),
        Float64(cv_settlement),
        mask_bay_of_fundy,
        Tuple(Symbol(lowercase(string(b))) for b in active_boundaries)
    )
end


"""
    HydrodynamicConfig

Physical parameters specifying Eulerian ocean circulation, lateral boundary
conditions, atmospheric fluxes, and numerical time integration.
"""
struct HydrodynamicConfig
    domain_lon               :: Tuple{Float64, Float64}
    domain_lat               :: Tuple{Float64, Float64}
    domain_z                 :: Tuple{Float64, Float64}
    grid_size                :: Tuple{Int, Int, Int}
    grid_orientation         :: Symbol
    rotation_azimuth         :: Float64
    center_lon               :: Float64
    center_lat               :: Float64
    along_shelf_extent_km    :: Float64
    cross_shelf_extent_km    :: Float64
    vertical_stretching_mode :: Symbol
    nz_above :: Int
    vertical_break_depth :: Float64
    vertical_grid_file       :: String
    vertical_depths          :: Vector{Float64}
    bathymetry_source        :: Symbol
    bathy_source             :: String
    coastline_source         :: String
    wind_source              :: String
    wind_time_iso            :: String
    inshore_depth            :: Float64
    shelf_slope              :: Float64
    atmospheric_source       :: Symbol
    drag_formulation         :: Symbol
    bulk_heat_flux           :: Bool
    climatology              :: Bool
    mhw_temp_anomaly         :: Float64
    ocean_boundary_source    :: Symbol
    obc_type                 :: Symbol
    sponge_layer_width       :: Float64
    sponge_timescale         :: Float64
    tidal_relaxation_timescale :: Float64
    enable_tides             :: Bool
    tides_source             :: Symbol
    tidal_constituents       :: Vector{Symbol}
    tidal_u_amp              :: Float64
    tidal_v_amp              :: Float64
    cd_drag                  :: Float64
    bottom_drag              :: Float64
    bbl_mixing_closure       :: Symbol
    sim_duration_seconds     :: Float64
    sim_dt_seconds           :: Float64
    adaptive_cfl             :: Bool
     target_cfl               :: Float64
     target_wave_cfl          :: Float64
     max_dt_seconds           :: Float64
     min_dt_seconds           :: Float64
      coriolis_latitude        :: Float64
       divergence_limit         :: Float64
       output_path              :: AbstractString
       output_schedule_seconds  :: Float64
       enable_checkpoint        :: Bool
       checkpoint_prefix        :: AbstractString
       checkpoint_schedule      :: Real
       checkpoint_dir           :: AbstractString
       checkpoint_cleanup       :: Bool
       auto_restart             :: Bool
       use_gpu                  :: Bool
       fallback_to_cpu          :: Bool
       storage_backend          :: String
end
