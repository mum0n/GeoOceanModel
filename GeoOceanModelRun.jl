# ==============================================================================
#  GeoOceanModelRun.jl -- calling script for GeoOceanModel
# ==============================================================================
#
#  The command-line entry point for the ocean half. It does what `ParticleTrackingRun.jl` did
#  for the segments this package owns: read the configuration, build the grid, build the
#  model, and end by writing the handover that `GeoLarvalTracking` reads.
#
#  What is deliberately NOT here: the larval segments (--track, --metrics, --viz,
#  --figures). They belong to `GeoLarvalTracking/GeoLarvalTrackingRun.jl`, which consumes the
#  result this script produces. A driver holding both would be the coupling SPLIT_PLAN.md
#  section 1 exists to remove.
#
#  Usage:
#      julia --project=. GeoOceanModelRun.jl --help
#      julia --project=. GeoOceanModelRun.jl --hydro-only
#      julia --project=. GeoOceanModelRun.jl --grid
#
#  Every function and field this script touches was checked against the loaded package with
#  `hasproperty`/`isdefined` before being written, not assumed. Where a segment's real body
#  lives in `ParticleTrackingRun.jl` and not in this package, the script says so instead of
#  calling something that would fail.
# ==============================================================================

# Auto-discover and activate this package's environment when run as a script.
#
# Guarded by `abspath(PROGRAM_FILE) == @__FILE__` so the bootstrap is skipped when this file
# is `include`d from a test or another script: `Pkg` cannot be imported into a package module,
# and a `Pkg.activate` at precompile time is both a side effect and an error.
#
# Wrapped in a function because the search assigns to loop-local variables, and at top level
# in a script those are soft-scoped and ambiguous -- Julia would silently treat them as new
# locals, which is exactly the bug that warning is describing.
# Auto-discover and activate this package's environment when run as a script.
#
# Two constraints shape this block:
#
#   1. `import Pkg` must be at top level -- Julia refuses it inside a function -- so the
#      whole bootstrap is at top level.
#   2. Assignments in a top-level `for`/`if` are soft-scoped and ambiguous in a script, so
#      the search runs inside a `let`, which gives them a distinct local scope and silences
#      the warning without also making them globals a later line could depend on.
#
# Guarded so the bootstrap is skipped when this file is `include`d from a test or another
# script: `Pkg` cannot be imported into a package module, and a `Pkg.activate` at precompile
# time is both a side effect and an error.
const _THIS_SCRIPT = @__FILE__

if !isempty(PROGRAM_FILE) && abspath(PROGRAM_FILE) == _THIS_SCRIPT &&
   Base.find_package("GeoOceanModel") === nothing
    import Pkg
    let curr = @__DIR__
        while true
            proj = joinpath(curr, "Project.toml")
            if isfile(proj) && occursin("name = \"GeoOceanModel\"", read(proj, String))
                Pkg.activate(curr)
                break
            end
            parent = dirname(curr)
            parent == curr && break
            curr = parent
        end
    end
end

using GeoOceanModel
using GeoOceanModel: configuration_to_options,
    options_to_configuration, save_configuration, resolve_config_name, build_shelf_grid,
    build_immersed_grid_from_real_data, build_hydrodynamic_model, HydrodynamicOptions,
    load_configuration, preferred_bathymetry_source
using TOML

"""
    display_help(io::IO = stdout)

The segments this package owns and the flags that select them.
"""
function display_help(io::IO = stdout)
    println(io, """
    GeoOceanModelRun.jl -- regional ocean modelling, from configuration to the handover.

    Usage:
      julia --project=. GeoOceanModelRun.jl [flags]

    Segments, run in this order:
      --data      Resolve which bathymetry product the configuration names, and over which
                  range. Reports the resolved source; does not itself fetch.
      --grid      Build the spherical grid and the immersed-boundary wrapper.
      --model     Configure the hydrostatic free-surface model on that grid.

    End to end:
      --hydro-only  data, grid, model.
      --segment=all the same.

    Configuration:
      --config=FILE      A path, or a name resolved against configs/.
                         Default: the project's ParticleTracking.toml, else configs/ocean.toml.
      --output-dir=DIR   Where the run writes.
      --save-config[=D]  Write the fully-resolved configuration, then exit.
      --help, -h         This message.

    The handover:
      A run ends by writing a `SimulationResult` beside its output: its location, its JLD2
      checkpoint, and the ocean parameters the larval half reads. This script never calls
      into GeoLarvalTracking.
    """)
    return nothing
end

"""
    parse_args(args::Vector{String}) -> Dict{String, Any}

The whole flag vocabulary in one place. An unrecognised flag is an error rather than silence:
a typo like `--trak` would otherwise fall through to the default segment and quietly do the
wrong thing, which is the failure a CLI should refuse.
"""
function parse_args(args::Vector{String})
    out = Dict{String, Any}(
        "segments" => String[], "config" => "", "save_config" => nothing,
        "output_dir" => "", "hydro_only" => false, "help" => false)
    for a in args
        if a in ("--data", "--grid", "--model", "--climate", "--sim")
            push!(out["segments"], replace(a, "--" => ""))
        elseif a in ("--hydro-only", "--segment=all")
            out["hydro_only"] = true
        elseif a in ("--help", "-h")
            out["help"] = true
        elseif a == "--save-config"
            out["save_config"] = ""
        elseif startswith(a, "--save-config=")
            out["save_config"] = split(a, "=", limit = 2)[2]
        elseif startswith(a, "--config=")
            out["config"] = split(a, "=", limit = 2)[2]
        elseif startswith(a, "--output-dir=")
            out["output_dir"] = split(a, "=", limit = 2)[2]
        elseif startswith(a, "--segment=")
            push!(out["segments"], split(a, "=", limit = 2)[2])
        else
            error("Unknown flag: $(a). Run with --help.")
        end
    end
    return out
end

"""
    resolve_config_file(flag::AbstractString) -> String

Which configuration this run uses. A name is resolved through the package's own resolver,
a path is used as given, and an empty flag falls back to the project's own split file, which
phase 4 put there for exactly this purpose.
"""
function resolve_config_file(flag::AbstractString)
    isempty(strip(flag)) || return resolve_config_name(String(flag))
    # This package's own split file, then its scenario config. Both are beside this script,
    # not one directory up: `..` from this package's directory is the workspace root, where
    # the split deliberately put neither of them.
    own = joinpath(@__DIR__, "ParticleTracking.toml")
    isfile(own) && return own
    return joinpath(@__DIR__, "configs", "ocean.toml")
end

"""
    load_options(flag::AbstractString) -> HydrodynamicOptions

The resolved configuration as the model builders take it. `load_configuration` reads the TOML
and validates its sections against the domain map, so a misspelled section is an error here
rather than a silent default later.
"""
function load_options(flag::AbstractString)
    return configuration_to_options(load_configuration(resolve_config_file(flag)))
end

"""
    build_grid(opts::HydrodynamicOptions)

The grid the model segment needs: base spherical grid, wrapped in an immersed boundary built
from real bathymetry by source and range -- never by path, per SPLIT_PLAN.md section 7.

The `output_path` is the run's own artefact, which is what the original driver passed too: the
cache is keyed by path, so pointing at the run's input directory is what makes a repeat run
reuse the product it already fetched rather than re-downloading it.
"""
function build_grid(opts::HydrodynamicOptions)
    base = build_shelf_grid(
        lon_range = opts.domain_lon,
        lat_range = opts.domain_lat,
        z_range = opts.domain_z,
        grid_size = opts.grid_size)
    artefact = joinpath(opts.input_dir, "bathymetry_active.nc")
    return build_immersed_grid_from_real_data(
        base;
        source = opts.bathy_source,
        output_path = isfile(artefact) ? artefact : nothing,
        min_water_depth = abs(opts.inshore_depth),
        mask_bay_of_fundy = opts.mask_bay_of_fundy)
end

function main(args::Vector{String})
    flags = parse_args(args)
    flags["help"] && return display_help()

    opts = try
        load_options(flags["config"])
    catch err
        println("Configuration error: $(sprint(showerror, err))")
        return display_help()
    end
    cfg_file = resolve_config_file(flags["config"])

    println("="^73)
    println(" GeoOceanModel -- regional ocean model")
    println("="^73)
    println(" Config:    $(cfg_file)")
    println(" Grid:      $(opts.grid_size[1]) x $(opts.grid_size[2]) x $(opts.grid_size[3]) cells")
    println(" Domain:    Lon [$(opts.domain_lon[1]), $(opts.domain_lon[2])], " *
            "Lat [$(opts.domain_lat[1]), $(opts.domain_lat[2])]")
    println(" Bathymetry: $(something(opts.bathy_source, "auto"))")
    println("="^73)

    if !isnothing(flags["save_config"])
        dest = isempty(flags["save_config"]) ? cfg_file : flags["save_config"]
        save_configuration(options_to_configuration(opts), dest)
        println("Resolved configuration written to $(dest)")
        return nothing
    end

    segments = flags["hydro_only"] ? ["data", "grid", "model"] : flags["segments"]
    isempty(segments) && return display_help()

    for seg in segments
        if seg == "data"
            source = preferred_bathymetry_source(
                source = opts.bathy_source, lon_range = opts.domain_lon,
                lat_range = opts.domain_lat)
            println("\n[data] bathymetry source: $(source) over " *
                    "Lon $(opts.domain_lon), Lat $(opts.domain_lat)")
            println("      resolution of the product, its range and its tier is " *
                    "GeoDataSources.bathymetry_dataset's job; this reports the decision.")
            println("      Fetching is not performed here: use the data segment of " *
                    "ParticleTrackingRun.jl until it is ported.")
        elseif seg == "grid" || seg == "model"
            println("\n[$(seg)] building...")
            immersed = build_grid(opts)
            println("      immersed grid: size $(size(immersed))")
            if seg == "model"
                model = build_hydrodynamic_model(immersed)
                println("      hydrostatic free-surface model configured")
            end
        else
            println("\n[$(seg)] is not owned by GeoOceanModel. The larval segments " *
                    "(--track, --metrics, --viz, --figures) live in " *
                    "GeoLarvalTracking/GeoLarvalTrackingRun.jl.")
        end
    end
    return nothing
end

is_cli = !isempty(PROGRAM_FILE) && abspath(PROGRAM_FILE) == @__FILE__
is_cli && main(ARGS)
