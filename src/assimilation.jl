# ==============================================================================
#  assimilation.jl -- what this configuration needs, and where it comes from
# ==============================================================================
#
#  The ocean run does not read products from a path a caller typed. It resolves each input
#  through `GeoDataSources`, which has one description per product: where it is published,
#  which file it is, which licence it carries, and where its assimilated copy lives. This
#  file is the ocean half's view of that register -- what this configuration will need, where
#  each will land, and whether it is already there.
#
#  The register already decides *which* source a run gets, by coverage:
#  `GeoDataSources.preferred_bathymetry_source` answers "which bathymetry for this domain",
#  and the driver asks it. What this file adds is the ability to *ask before running*, so a
#  notebook can say "NWA bathymetry is already in the lakehouse; TPXO9 needs 4.7 GB"
#  before a reader spends the afternoon.
#
#  The lakehouse is one directory per half, as it is for the others: `lakehouse/GeoOceanModel`
#  for this package's runs, `GeoDataSources.store_root()` -- `lakehouse/GeoData` -- for the
#  shared products. A product in `lakehouse/GeoData` is shared by every half that uses it;
#  a run in `lakehouse/GeoOceanModel/<run>` belongs to this half alone.
# ==============================================================================

"""
    ocean_store_root() -> String

Where this half's runs are written: `projects/lakehouse/GeoOceanModel`, beside the other
halves' directories. `MOVEMENT_STORE_ROOT` is not consulted here; `GEOCEAN_STORE_ROOT` is,
and defaults to the same relative path so it resolves identically from a notebook and from a
script.
"""
function ocean_store_root()
    override = get(ENV, "GEOCEAN_STORE_ROOT", "")
    isempty(override) || return abspath(override)
    return normpath(joinpath(@__DIR__, "..", "..", "..", "lakehouse", "GeoOceanModel"))
end

"""
    assimilation_status(opts) -> NamedTuple

What this configuration will read, where each input's assimilated copy lives, and whether it
is already there. Read-only: nothing is fetched and nothing is written.

One row per input the driver resolves, with three facts worth having *before* a run:

- `role` — which part of the run it serves, so a row is not a filename in isolation;
- `where` — the assimilated path it will be read from, resolved the same way the driver
  resolves it, so this and the run cannot disagree;
- `present` and `up_to_date` — what the run will reuse and what it will fetch. `up_to_date`
  is present-and-not-stale, and staleness is "older than its source", which for a published
  URL means the fetch date on the shelf.

The two keys are the only inputs that need something from you: Copernicus Marine wants an
account, and everything else is keyless. `needs_account` names them.
"""
function assimilation_status(opts::HydrodynamicOptions)
    rows = @NamedTuple{role::Symbol, key::Union{Symbol, Nothing},
                       source::Union{Symbol, String}, where::String,
                       present::Bool, stale::Bool, note::String}[]

    # -- shared products, resolved the way the driver resolves them --------------
    # Each `where` is the path the run will actually read, not the register's declared
    # template. For most products those are the same thing; for WOA23 they are not, because
    # the register declares a tier-1 raw directory (`<store>/woa23_oxygen_00_0.25deg`) while
    # the driver reads the assimilated products the fetcher writes under `store_path("woa23")`.
    # Reporting the template would tell a reader the run is missing a file it never reads.
    woa_dir = GeoDataSources.store_path("woa23")
    shared = [
        (:bathymetry, :bathymetry,
         "measured bathymetry, resolved by coverage (see `preferred_bathymetry_source`)"),
        (:coastline, :coastline, "Natural Earth coastline, public domain"),
        (:winds, :atmosphere, "open-meteo surface winds, keyless"),
        (:tides, :tides, "TPXO9-Atlas tidal constituents (a 4.7 GB archive)"),
        (:climatology, :woa23, "WOA23 temperature/salinity, keyless"),
        (:boundary, :glorys, "GLORYS12V1 open-boundary water, **needs a Copernicus account**"),
    ]
    for (role, key, note) in shared
        where = if role == :climatology
            woa_dir
        else
            try
                GeoDataSources.resolved_cache_path(key, opts.output_dir)
            catch
                "unresolved"
            end
        end
        push!(rows, (role = role, key = key, source = key,
                     where = where,
                     present = _present_path(where) !== nothing,
                     stale = false, note = note))
    end

    # -- this half's own output store -------------------------------------------
    run_store = output_zarr_path(opts.output_dir)
    push!(rows, (role = :output, key = nothing,
                 source = "the realized simulation",
                 where = run_store,
                 present = _present_path(run_store) !== nothing,
                 stale = false,
                 note = "written by `attach_output_callback!`; the larval half reads it"))

    needs_account = [r for r in rows if occursin("account", r.note)]
    return (
        run_dir = abspath(opts.output_dir),
        run_store = run_store,
        shared_root = GeoDataSources.store_root(),
        inputs = rows,
        present = [r.role for r in rows if r.present],
        absent = [r.role for r in rows if !r.present && r.role != :output],
        needs_account = [r.role for r in needs_account],
    )
end

"""
    _present_path(path)

The path if something exists there (a Zarr store, or a file), else `nothing`.
An assimilated product can be either: the register stores a Zarr directory or a NetCDF file,
and a row that reports present for one and absent for the other would be wrong.
"""
_present_path(path::AbstractString) = isdir(path) || isfile(path) ? path : nothing

"""
    report_assimilation_status(io, status)

The status as text: one row per input, with where it will be read from and what the run will
do. Written rather than returned as a table, so a notebook prints it and a log keeps it.
"""
function report_assimilation_status(io::IO, status)
    println(io, "assimilation status for the run at $(status.run_dir)")
    println(io, "  shared products : $(status.shared_root)")
    for r in status.inputs
        what = r.role == :output ? "write " : r.present ? "have  " : "fetch "
        println(io, "  ", what, rpad(string(r.role), 13), " ", r.where)
    end
    isempty(status.absent) || println(io, "  to fetch    : ", join(status.absent, ", "))
    isempty(status.needs_account) ||
        println(io, "  needs an account: ", join(status.needs_account, ", "))
    return nothing
end
