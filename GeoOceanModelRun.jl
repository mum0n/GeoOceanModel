# ==============================================================================
#  GeoOceanModelRun.jl -- CLI for the ocean half
# ==============================================================================
#
#  This file holds CLI interaction only: environment bootstrap, flag parsing, and dispatch.
#  The segments -- data, grid, model, climate, simulation -- are `src/driver.jl`, imported
#  below, because a script that defines the behaviour is a script nobody can test and
#  `Pkg.test` cannot reach a function a script defines. `run_segments` composes them in the
#  dependency order `run_production_pipeline` uses.
#
#  What is deliberately NOT here: the larval segments (--track, --metrics, --viz,
#  --figures). They belong to `GeoLarvalTracking/GeoLarvalTrackingRun.jl`, which tracks
#  through the flow solution this script produces. A driver holding both would be the
#  coupling SPLIT_PLAN.md section 1 exists to remove.
#
#  Usage:
#      julia --project=. GeoOceanModelRun.jl --help
#      julia --project=. GeoOceanModelRun.jl --all
#      julia --project=. GeoOceanModelRun.jl --grid
#
#  Every name this script imports was checked with `isdefined` against the loaded package,
#  not assumed: it dispatches to functions `src/driver.jl` actually defines.
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
    options_to_configuration, save_configuration, resolve_config_name, load_configuration,
    run_segment_data, run_segment_grid, run_segment_model, run_segment_climate,
    run_segment_simulation, HydrodynamicOptions
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

    Segments, run in this order (each one runs the ones before it):
      --data      Fetch the coastline, the bathymetry and the surface winds the config names.
      --grid      Build the spherical grid and the immersed-boundary wrapper.
      --model     Configure the hydrostatic free-surface model on that grid.
      --climate   Apply the scenario's warming and freshening to the stratification.
                  A perturbation, not a prerequisite: it runs only when named.
      --sim       Integrate the primitive equations and write the flow solution.

    End to end:
      --all        data through --sim, without the climate perturbation: produce the flow
                   solution, then stop. Use --climate to run the scenario as well.

    Configuration:
      --config=FILE      A path, or a name resolved against configs/.
                         Default: configs/default.toml.
      --output-dir=DIR   Where the run writes.
      --save-config[=D]  Write the fully-resolved configuration, then exit.
      --help, -h         This message.

    Two configurations:
      default         Cape Breton : small box, low resolution, 3 days,
                      keyless data (etopo2022 bathymetry, WOA23, open_meteo
                      winds, synthetic edges). For demos and error testing.
       scotian_shelf   The full snow-crab Scotian Shelf domain, nwa_bathymetry
                       and GLORYS12V1 (Copernicus Marine account required).
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
        "output_dir" => "", "all" => false, "help" => false)
    for a in args
        if a in ("--data", "--grid", "--model", "--climate", "--sim")
            push!(out["segments"], replace(a, "--" => ""))
        elseif a == "--all"
            out["all"] = true
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

Which configuration this run uses, as a *path*. An empty flag is this half's own
`configs/default.toml`; anything else is resolved through `resolve_config_path`. This returns a
path -- `load_configuration` opens the file -- so unlike the package's `resolve_config_name`
(which returns a stem for checkpoint prefixes) it never reduces a config to just its name.
"""
function resolve_config_file(flag::AbstractString)
    isempty(strip(flag)) && return joinpath(@__DIR__, "configs", "default.toml")
    return resolve_config_path(String(flag))
end

"""
    resolve_config_path(flag::AbstractString) -> String

A file path is used directly; a bare name is looked up under this package's `configs/`
directory. A name that resolves to nothing is an error naming what was tried, rather than a
fall-back to a default the operator did not choose.
"""
function resolve_config_path(flag::AbstractString)::String
    f = String(flag)
    isfile(f) && return abspath(f)
    name = endswith(f, ".toml") ? f : f * ".toml"
    named = joinpath(@__DIR__, "configs", name)
    isfile(named) && return named
    error("Configuration file not found: $(f). Pass a path, or a name under configs/ " *
          "(for example: default, scotian_shelf).")
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
    SEG_CHAIN

The ocean segments in dependency order, as `run_production_pipeline` composes them. A later
segment needs the ones before it, so requesting one runs its prerequisites rather than
erroring on a missing grid or model.
"""
const SEG_CHAIN = ["data", "grid", "model", "climate", "sim"]

"""
    run_segments(opts::HydrodynamicOptions, requested::Vector{String}) -> NamedTuple

Run the ocean segments, in `SEG_CHAIN` order, up to the furthest one requested. Each step
takes the previous one's result exactly as `run_production_pipeline` does -- the model gets
the grid's immersed wrapper and Segment 1's wind stress; the climate and simulation steps get
the model -- so a partial run and a full run reach the same code path.

Returns `(data, grid, model, climate, sim)`, with `nothing` in the slots not run.
"""
function run_segments(opts::HydrodynamicOptions, requested::Vector{String})
    order = String.(requested)
    unknown = filter(s -> s ∉ SEG_CHAIN, order)
    isempty(unknown) || error("Unknown segment(s): $(join(unknown, ", ")). Known: " *
                              "$(join(SEG_CHAIN, ", ")).")

    # Furthest requested segment defines how far down the chain to run, and every segment
    # before it runs too because a later one builds on an earlier one's state -- the model
    # needs the grid's immersed wrapper, the simulation needs the model. An empty request
    # stops after the model (data, grid, model); `--all` runs every segment.
    #
    # climate is the exception. It perturbs the stratification instead of building anything,
    # so it runs only when named: `--all` and `--sim` alone both produce the flow solution
    # from the unperturbed model, which is what the larval half tracks through.
    target_idx = isempty(order) ? 3 : maximum(i -> findfirst(==(i), SEG_CHAIN), order)
    plan = filter(1:target_idx) do i
        SEG_CHAIN[i] != "climate" || "climate" in order
    end

    data_res = grid_res = model_res = climate_res = sim_res = nothing
    for i in plan
        seg = SEG_CHAIN[i]
        if seg == "data"
            data_res = run_segment_data(; opts = opts)
        elseif seg == "grid"
            grid_res = run_segment_grid(; opts = opts)
        elseif seg == "model"
            tau_x, tau_y = isnothing(data_res) ? (1e-4, 0.0) : (data_res.tau_x, data_res.tau_y)
            model_res = run_segment_model(; opts = opts,
                                          immersed_grid = grid_res.immersed_grid,
                                          tau_x = tau_x, tau_y = tau_y)
        elseif seg == "climate"
            climate_res = run_segment_climate(; opts = opts, model = model_res.model)
        elseif seg == "sim"
            sim_res = run_segment_simulation(; opts = opts, model = model_res.model)
        end
    end
    return (data = data_res, grid = grid_res, model = model_res,
            climate = climate_res, sim = sim_res)
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

    # --all runs the whole chain, data through the simulation, without the climate
    # perturbation (see `run_segments`: `climate` is named or it does not run). The segment
    # functions are the package's own `src/driver.jl`, imported above; this script only
    # decides how far down the chain to go and then dispatches.
    requested = flags["all"] ? SEG_CHAIN : flags["segments"]
    isempty(requested) && return display_help()

    results = run_segments(opts, requested)
    return nothing
end

is_cli = !isempty(PROGRAM_FILE) && abspath(PROGRAM_FILE) == @__FILE__
is_cli && main(ARGS)
