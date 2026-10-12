# ==============================================================================
#  simulation_result.jl -- the handover between the two halves
# ==============================================================================
#
#  THE ONE COUPLING BETWEEN GeoOceanModel AND GeoLarvalTracking.
#
#  SPLIT_PLAN.md section 1: "The one coupling is therefore the output file. There is no import
#  edge from GeoLarvalTracking to GeoOceanModel, and no shared TOML." This file is that
#  coupling, written down.
#
#  What crosses the boundary is a LOCATION and the METADATA BESIDE IT, and nothing else. A
#  location points at a JLD2 checkpoint, or at an assimilated store -- YAXArray over Zarr,
#  because that is what carries named variables and their attributes.
#
#  Two stores, two jobs, and the split is deliberate:
#
#    JLD2     the checkpoint. Restart capacity. Oceananigans' own Checkpointer and JLD2Writer
#             write this, so restart rides the writer that exists rather than one we would
#             have to write and verify.
#    YAXArray the output. Model slices, with metadata attached to the variables so a reader
#             can name what it reads and find its units without guessing.
#
#  Why a location rather than an object: the two halves are separate packages and the ocean
#  half finishes before the larval half starts. Handing over a live object would make them
#  agree on Oceananigans' version, the architecture, and to run in one process -- which is
#  the coupling this split exists to remove. A path is a path.
#
#  ---------------------------------------------------------------------------
#  A measured constraint on the YAXArray store, recorded because it costs a workaround
#  ---------------------------------------------------------------------------
#
#  Verified against YAXArrays 0.7.8 on Zarr 0.10.2: VALUES round-trip exactly, and VARIABLE
#  properties survive (units, long_name, anything else a variable carries). Two things do
#  not, and are worked around here rather than left to surprise the next reader:
#
#    1. AXIS properties do not survive the round trip. A `Lon` with units "degrees_east"
#       comes back as a bare range with no properties. So the store below carries axis units
#       in each variable's attributes (`lon_units`, `lat_units`, `time_units`), which do
#       survive, and a reader takes them from there.
#    2. `YAXArrays.getAxis` / `lookup` on a reopened dataset raise StackOverflowError in
#       0.7.8. So nothing here calls them; the store records the axis VALUES in attributes
#       too, so a reader never needs to look an axis up by name.

# The contract: what the ocean half promises the larval half it recorded. The larval half
# reads these, and if any is missing it errors rather than defaulting -- a track through a
# flow that does not state its own domain would interpolate through the wrong ocean and
# nothing downstream could tell.
const OCEAN_PARAMETER_KEYS = (
    :domain_lon,
    :domain_lat,
    :z_range,
    :bathy_source,
    :bathymetry_tier,
    :realized_domain_lon,
    :realized_domain_lat,
    :realized_grid_spacing_lon,
    :realized_grid_spacing_lat,
    :ocean_parameters_resolved,
    :simulation_start_time,
    :simulation_end_time,
    :schema_version,
)

"""
    SimulationResult

Where a realized ocean simulation lives, and what metadata must travel beside it.

`checkpoint` is the JLD2 restart artefact, and `location` is the read path -- which is the
YAXArray output store when there is one, and the checkpoint when there is not. Both are paths;
neither carries the other by assumption.

This type holds no flow fields and no arrays. It is the handover, not the payload.
"""
struct SimulationResult
    location::String
    checkpoint::Union{Nothing, String}
    format::Symbol
    metadata::Dict{Symbol, Any}
end

"""
    simulation_result_location(result) -> String

The location a consumer is handed. A path, and nothing else.
"""
simulation_result_location(result::SimulationResult) = result.location

"""
    open_simulation_result(location; required = nothing) -> SimulationResult

Open a realized ocean simulation by location.

The location is a path: whether it is a JLD2 archive or a YAXArray store is resolved from
what is on disk, never from a suffix a caller supplies -- a store named with one extension
and written with another is the reader's fault, not its guess. Absence is an error, never a
synthetic field: a run told which ocean to read must read that ocean.

`required` names the metadata keys that must be present, and defaults to
`OCEAN_PARAMETER_KEYS`, so a caller that omits it gets the whole contract.
"""
function open_simulation_result(location::AbstractString; required::Union{Nothing, Tuple} = nothing)
    path = abspath(location)
    !ispath(path) && error(
        "No realized simulation at $(path). This is the location the run was handed, and a " *
        "missing simulation cannot be substituted: run the ocean half first, or correct the " *
        "path. Nothing here writes a synthetic stand-in.")

    format = _infer_result_format(path)
    metadata = format === :jld2 ? _read_ocean_metadata_jld2(path, format) :
                                  _read_ocean_metadata_yaxarray(path)
    checkpoint = _checkpoint_for(path, format)

    wanted = something(required, OCEAN_PARAMETER_KEYS)
    missing = [k for k in wanted if !haskey(metadata, k)]
    isempty(missing) || error(
        "The simulation at $(path) is missing $(length(missing)) required ocean parameter(s): " *
        "$(missing). A track through a flow that does not state its own domain would silently " *
        "interpolate through the wrong ocean, so this is an error rather than a default.")

    return SimulationResult(path, checkpoint, format, metadata)
end

"""
    ocean_parameters(result) -> Dict{Symbol, Any}

The ocean half's realized parameters, read from the handover rather than recomputed.

Every key was checked against `OCEAN_PARAMETER_KEYS` at open time, so a value a reader gets
here was written by the ocean half -- including values resolved during the run, which is the
point. A consumer cannot know which decision a run made, so it must not re-derive one.
"""
ocean_parameters(result::SimulationResult) = result.metadata

"""
    write_simulation_result(location, checkpoint, metadata) -> SimulationResult

Write the handover: a location, its checkpoint, and the ocean parameters recorded beside it.

`metadata` is validated against `OCEAN_PARAMETER_KEYS` here rather than at read time, so a
missing parameter is caught by the half that owns the value instead of by whichever consumer
happens to run next.
"""
function write_simulation_result(location::AbstractString, checkpoint::AbstractString,
                                 metadata::Dict{Symbol, Any})
    path = abspath(location)
    missing = [k for k in OCEAN_PARAMETER_KEYS if !haskey(metadata, k)]
    isempty(missing) || error(
        "Cannot write a simulation result without its ocean parameters. Missing: $(missing). " *
        "The ocean half owns these values and must record them, because the consumer that " *
        "reads the result cannot reconstruct them.")

    format = _infer_result_format(path)
    if format === :yaxarray
        _write_ocean_metadata_yaxarray(path, metadata)
        # The checkpoint travels inside the store, so a result and its restart file are moved
        # together rather than the store being moved and the checkpoint orphaned. Skipped only
        # when it is already there, in the exact place the reader will look -- comparing parent
        # directories instead would skip it whenever the ocean half happened to write beside
        # the store, which is the common case.
        ck = abspath(checkpoint)
        inside = joinpath(abspath(path), "checkpoint.jld2")
        if isfile(ck) && ck != inside
            cp(ck, inside; force = true)
        end
    else
        _write_ocean_metadata_jld2(path, metadata)
    end
    return SimulationResult(path, _checkpoint_for(path, format), format, copy(metadata))
end

"""
    _checkpoint_for(path, format) -> Union{Nothing, String}

Where the JLD2 restart artefact sits for a result at `path`, or `nothing` when there is none.

When the result IS the JLD2 archive, the location is the checkpoint -- there is no second file
to derive a path to, and inventing one (`archive.jld2.checkpoint.jld2`) would send a restart to
a path that was never written. When the result is a YAXArray store, the checkpoint sits inside
the store, so it travels with it.
"""
function _checkpoint_for(path::AbstractString, format::Symbol)
    p = abspath(path)
    if format === :jld2
        return isfile(p) ? p : nothing
    end
    inside = joinpath(p, "checkpoint.jld2")
    return isfile(inside) ? inside : nothing
end

"""
    _infer_result_format(path) -> Symbol

`:yaxarray` for a YAXArray/Zarr store, `:jld2` for a JLD2 archive, from what is on disk.

Resolved from the store itself and never from the suffix. Two checks run, because a Zarr store
carries its own `.zgroup` marker while a JLD2 file does not, and telling them apart from a
suffix is how a reader opens the wrong thing and reports success.
"""
function _infer_result_format(path::AbstractString)
    path = abspath(path)
    if isdir(path) && isfile(joinpath(path, ".zgroup"))
        return :yaxarray
    end
    return isfile(path) ? :jld2 : :yaxarray
end

# ---- encoding, because JSON has no tuples ----------------------------------
#
# The contract's `domain_lon`/`domain_lat`/`z_range` are ordered pairs. JSON represents an
# ordered pair as an array, so without encoding they read back as a 2-element vector and the
# caller cannot tell a range from a list of two other numbers. Each range is therefore written
# as `{"__range": [lo, hi]}` and restored to a tuple on read, which keeps the meaning of every
# contract key the same on both sides of the boundary.

_range_tag(k::AbstractString) = "type::$(k)"

_is_range_encoding(v) = v isa AbstractDict && haskey(v, "__range")

"""
    _encode_metadata_value(v) -> Any

Encode a contract value for the sidecar format. A `Tuple` of two numbers, which is how this
package's own config spells a range, becomes a tagged mapping; anything else is stored as-is.
"""
function _encode_metadata_value(v)
    (v isa Tuple && length(v) == 2 && all(x -> x isa Number, v)) || return v
    return Dict{String, Any}("__range" => [Float64(v[1]), Float64(v[2])])
end

"""
    _decode_metadata_value(v) -> Any

Undo `_encode_metadata_value`: a tagged mapping becomes a range tuple again.
"""
function _decode_metadata_value(v)
    _is_range_encoding(v) || return v
    pair = v["__range"]
    length(pair) == 2 || error(
        "A stored range must hold exactly two values, but this one holds $(length(pair)).")
    return (Float64(pair[1]), Float64(pair[2]))
end

# ---- metadata sidecars -----------------------------------------------------
#
# The metadata is written in ONE of two places, chosen by the result's own format, and in
# neither case inside the arrays: JLD2 keeps it in the archive, YAXArray keeps it in the
# store's own `.zattrs`. Both survive a move, because both travel with the store.

function _metadata_sidecar(path::AbstractString)
    p = abspath(path)
    return isdir(p) ? p : p * ".ocean.json"
end

function _read_ocean_metadata_jld2(path::AbstractString, format::Symbol)
    sidecar = _metadata_sidecar(path)
    if isfile(sidecar)
        return _parse_metadata_sidecar(sidecar)
    end
    # A JLD2 archive carries its metadata inside itself when the sidecar is absent, which is
    # the case for a checkpoint written by Oceananigans' own writer.
    return _read_metadata_from_jld2_archive(path)
end

function _read_metadata_from_jld2_archive(path::AbstractString)
    recorded = Dict{Symbol, Any}()
    # JLD2 is a dependency of this package (ParticleTracking.jl `using JLD2`), so it is in
    # scope here. The try/catch is for a caller that loaded this file into a context where
    # it is not, and reports that rather than failing at an undefined symbol.
    can_read = try
        isdefined(@__MODULE__, :JLD2)
    catch
        false
    end
    can_read || error(
        "The result at $(path) is a JLD2 archive, so its ocean parameters are read with " *
        "JLD2, which is not available in this module. It is a dependency of this package; " *
        "load it before calling.")

    JLD2.jldopen(path, "r") do file
        haskey(file, "ocean_parameters") || return nothing
        stored = file["ocean_parameters"]
        stored isa AbstractDict || return nothing
        for (k, v) in stored
            recorded[Symbol(k)] = v
        end
    end
    isempty(recorded) && error(
        "The JLD2 archive at $(path) carries no `ocean_parameters` group, so the ocean half " *
        "never recorded its parameters. A consumer that fell back to defaults here would " *
        "track larvae through an ocean it never read.")
    return recorded
end

function _write_ocean_metadata_jld2(path::AbstractString, metadata::Dict{Symbol, Any})
    p = abspath(path)
    mkpath(dirname(p))
    open(_metadata_sidecar(p), "w") do io
        JSON3.write(io, Dict{String, Any}(k => _encode_metadata_value(v)
                                          for (k, v) in metadata))
    end
    return nothing
end

# ---- YAXArray store --------------------------------------------------------

"""
    _write_ocean_metadata_yaxarray(store_path, metadata)

Record the ocean parameters in a YAXArray store's own `.zattrs`.

Each parameter is written under ONE key, `ocean_<name>`, rather than both bare and prefixed:
writing both means a second write accumulates prefixes (`ocean_ocean_schema_version`) and the
reader then has to decide which copy to trust. One key, prefixed, because it cannot collide
with a variable's name -- and the prefix is stripped on read, which is the only place the
contract's names are restored.
"""
function _write_ocean_metadata_yaxarray(store_path::AbstractString,
                                        metadata::Dict{Symbol, Any})
    p = abspath(store_path)
    mkpath(p)

    attrs = Dict{String, Any}()
        contract = string.(OCEAN_PARAMETER_KEYS)
        for (k, v) in metadata
            name = String(k)
            name in contract || error(
                "Ocean parameter `$(name)` is not part of the contract. The parameters the larval " *
                "half may rely on are $(contract). Add it to `OCEAN_PARAMETER_KEYS` rather than " *
                "recording it beside them, or it becomes a second contract.")
            attrs["ocean_$(name)"] = _encode_metadata_value(v)
        end
    open(joinpath(p, ".zattrs"), "w") do io
        JSON3.write(io, attrs)
    end
    return nothing
end

function _read_ocean_metadata_yaxarray(path::AbstractString)
    p = abspath(path)
    attrs_path = joinpath(p, ".zattrs")
    if isfile(attrs_path)
        parsed = JSON3.read(read(attrs_path, String))
        metadata = Dict{Symbol, Any}()
        contract = string.(OCEAN_PARAMETER_KEYS)
        for (k, v) in pairs(parsed)
            name = String(k)
            startswith(name, "ocean_") || continue
            bare = name[length("ocean_")+1:end]
            haskey(metadata, Symbol(bare)) && continue
            metadata[Symbol(bare)] = _decode_metadata_value(v)
        end
        isempty(metadata) || return metadata
    end

    sidecar = _metadata_sidecar(path)
    isfile(sidecar) && return _parse_metadata_sidecar(sidecar)

    error(
        "The store at $(path) carries no ocean parameters: neither `.zattrs` nor a sidecar. " *
        "The ocean half must record them, because the consumer cannot reconstruct them.")
end

function _parse_metadata_sidecar(sidecar::AbstractString)
    parsed = JSON3.read(read(sidecar, String))
    metadata = Dict{Symbol, Any}()
    for (k, v) in pairs(parsed)
        metadata[Symbol(k)] = _decode_metadata_value(v)
    end
    return metadata
end

"""
    ocean_variable_attributes(lon, lat, time; var_units = nothing) -> Dict{String, Any}

The attributes to attach to an output variable, carrying what the store cannot:

- `lon_units`, `lat_units`, `time_units` -- axis units, which do NOT survive a YAXArrays
  round trip (verified on 0.7.8: an axis with properties reads back as a bare range).
- `lon`, `lat`, `time` -- the axis values themselves, so a reader never has to look an axis
  up, which would raise StackOverflowError on this version.

This is the workaround the axis bug forces, and it lives here rather than at each write site
so the shape of the attributes is decided once.
"""
function ocean_variable_attributes(lon::AbstractVector, lat::AbstractVector,
                                   time::AbstractVector;
                                   lon_units::AbstractString = "degrees_east",
                                   lat_units::AbstractString = "degrees_north",
                                   time_units::AbstractString = "seconds")
    return Dict{String, Any}(
        "lon" => Float64.(collect(lon)),
        "lat" => Float64.(collect(lat)),
        "time" => Float64.(collect(time)),
        "lon_units" => lon_units,
        "lat_units" => lat_units,
        "time_units" => time_units,
    )
end
