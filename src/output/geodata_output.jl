# ==============================================================================
#  geodata_output.jl -- the simulated output as a GeoData store
# ==============================================================================
#
#  Why this exists. A run's output used to reach the larval half as a JLD2 archive plus a
#  sidecar file, and the reader had to know Oceananigans' storage layout to interpret it
#  (halos, C-grids, which variable lives on which face). That coupling is removed here: the
#  ocean half writes its output straight into a GeoData-native store, and the larval half
#  opens it through the data layer rather than through Oceananigans.
#
#  What is stored, and how:
#
#    JLD2     the checkpoint only. Oceananigans can checkpoint-restart in its own format and
#             no other, so restart stays there.
#    Zarr     the simulated slices -- u, v, w, T at every output interval -- written by a
#             callback, with coordinates and attributes attached via YAXArrays.
#
#  Why a callback rather than Oceananigans' own `ZarrWriter`. Verified against the installed
#  Oceananigans: `ZarrWriter` constructs, but `Simulation` accepts neither `output_writers` nor
#  `output_writers =`, so there is no supported way to attach it to a run. `Callback` and
#  `add_callback!` are supported, and writing through YAXArrays is code this project already
#  has and has tested. A callback is therefore the route that works with the Oceananigans
#  available rather than the one needing a newer version.
#
#  What makes the sidecar unnecessary: the grid's interior cell centres are written into the
#  store itself, as axis values. A reader no longer needs a second file stating halos and face
#  extents, because the same information reaches it as coordinates it can interpolate
#  against. The store is self-describing.
# ==============================================================================

using Oceananigans
using Oceananigans.Units
using YAXArrays
using Zarr
using Dates

const YAX_ATTRS_NS = nothing

"""
    output_zarr_path(output_dir::AbstractString, name::AbstractString = "hydrodynamics")

Where the simulated output store lives, as a path. A path, not a format: how it is written is
this file's business, and what it contains is the reader's.
"""
output_zarr_path(output_dir::AbstractString, name::AbstractString = "hydrodynamics") =
    joinpath(output_dir, name * ".zarr")

"""
    GeodataOutputCallback

An Oceananigans `Callback` that writes the ocean state into a Zarr store at each firing.

One store per run, appended to at each interval, so a long simulation writes incrementally
rather than holding its history in memory and losing it on a crash. That is also why the
writing is through `Zarr`'s directory store directly: a `YAXArrays` dataset built once and
saved at the end would need the whole history resident.
"""
struct GeodataOutputCallback
    store::String
    field_names::Vector{String}
    written::Base.RefValue{Int}
end

"""
    GeodataOutputCallback(output_dir, model; fields = ("u","v","w","T"), resume = false)

Construct the output writer for one run. A fresh run clears the store and starts at slice 0.
`resume = true` keeps the store and continues from however many slices it already holds, so an
extended/interrupted run appends to its own output instead of re-downloading nothing and
overwriting the interval it already wrote.
"""
function GeodataOutputCallback(output_dir::AbstractString, model;
                               fields::Tuple = ("u", "v", "w", "T"), resume::Bool = false)
    isdir(output_dir) || mkpath(output_dir)
    store = output_zarr_path(output_dir)
    if !resume && ispath(store)
        rm(store; recursive = true, force = true)   # a fresh run overwrites its own output
    end
    mkpath(store)

    # On resume, keep the interval this store already holds: the next slice continues at the
    # existing count. On a fresh run there is no time.json, so the count is 0.
    start = 0
    if resume
        tf = joinpath(store, "time.json")
        start = isfile(tf) ? length(Vector{Float64}(JSON3.read(read(tf, String)))) : 0
    end

    # The frame's metadata: which fields, and from when. Written as `.zattrs` so a reader can
    # tell what it is holding without opening every array.
    frame = Dict{String, Any}(
        "fields" => collect(fields),
        "title" => "Realized hydrodynamic simulation",
        "created_utc" => string(Dates.now(Dates.UTC)))
    open(joinpath(store, ".zgroup"), "w") do io
        write(io, "{\"zarr_format\": 2}")
    end
    open(joinpath(store, ".zattrs"), "w") do io
        JSON3.write(io, frame)
    end

    return GeodataOutputCallback(store, collect(String.(fields)), Ref(start))
end

"""
    _field_arrays(model) -> (; x, y, z)

The grid's **interior** cell centres as geographic `(lon, lat, depth)`.

The halo must be stripped: writing the halo-inclusive coordinate array would put store index
`i` at halo position `i` rather than interior position `i` — a silent displacement of the whole
field with a shape that still looks fine. Rather than slice the halo range here, this delegates
to `extract_grid_coordinates`, the project's tested halo-stripped extractor in degrees and
metres, so the store's coordinates are the model's real interior centres by construction.
"""
function _field_arrays(model)
    g = model.grid
    base = hasproperty(g, :underlying_grid) ? g.underlying_grid : g
    # Geographic (spherical) grid -- the real model. Delegate to the project's tested
    # degrees/metres extractor so the store's coordinates are the same interior centres the
    # rest of the code reports; hand-slicing the halo here shifted them by the halo width.
    if base isa Oceananigans.Grids.LatitudeLongitudeGrid
        c = extract_grid_coordinates(base)
        return (x = Float64.(c.lons), y = Float64.(c.lats), z = Float64.(c.depths))
    end
    # Metric/rectilinear grid (the unit tests): Cartesian horizontal centres in metres, halo
    # stripped, with the vertical centre from `znode` as above.
    ix = (base.Hx + 1):(base.Hx + base.Nx)
    iy = (base.Hy + 1):(base.Hy + base.Ny)
    x = Float64[v for v in base.xᶜᵃᵃ[ix]]
    y = Float64[v for v in base.yᵃᶜᵃ[iy]]
    z = [Float64(Oceananigans.Grids.znode(k, base, Oceananigans.Grids.Center()))
         for k in 1:base.Nz]
    return (x = x, y = y, z = z)
end

function (cb::GeodataOutputCallback)(sim::Oceananigans.Simulation)
    model = sim.model
    m = cb.written[]

    # The interior cell centres, resolved from the model's own grid, so the store states where
    # its cells are on Earth rather than leaving a reader to guess from an index.
    ax = _field_arrays(model)

    fields = Dict{String, Any}()
    for name in cb.field_names
        f = name == "u" ? model.velocities.u :
            name == "v" ? model.velocities.v :
            name == "w" ? model.velocities.w :
            name == "T" ? model.tracers.T : error("unknown field $(name)")
        fields[name] = Array{Float64}(interior(f))
    end

    # `time` lives on the model's clock, not on the Simulation: `Simulation` has no `clock`
    # field, and reaching for one is a FieldError at the first write.
    t = Float64(sim.model.clock.time)
    time_file = joinpath(cb.store, "time.json")
    stored = isfile(time_file) ? Vector{Float64}(JSON3.read(read(time_file, String))) :
                                 Float64[]
    push!(stored, t)
    open(time_file, "w") do io
        JSON3.write(io, stored)
    end

    for (name, arr) in fields
        idx_dir = joinpath(cb.store, name, string(m))
        mkpath(idx_dir)
        # One Zarr array per interval, created with `zcreate` and opened for write. The API is
        # `zopen`/`zcreate`, not `Zarr.Zarray`: this version has no such type, and the name
        # exists only in newer releases.
        d = Zarr.zcreate(Float64, Zarr.DirectoryStore(idx_dir), size(arr)...;
                         fill_value = 0.0)
        # Cartesian indexing, so the assignment is one broadcast rather than a loop over the
        # axes' offsets: a Zarr array is indexed from 1 like any Julia array, but writing it
        # slice-by-slice is how a transposed field stays invisible.
        d .= arr
    end
    cb.written[] = m + 1
    return nothing
end

"""
    attach_output_callback!(sim, output_dir; fields = ("u","v","w","T"))

Build the store and attach the writer to `sim`. Returns the store path, so the caller records
it in the handover and the larval half is given a location rather than an object.

The callback is `Callback(cb)` with no schedule: Oceananigans supplies one from the
simulation's own output interval, which is the run's actual cadence. Passing an explicit
`Schedule.Every(...)` would need `Oceananigans.Schedule`, which this version does not export,
and would risk two different cadences in one run.
"""
function attach_output_callback!(sim, output_dir::AbstractString;
                                  fields::Tuple = ("u", "v", "w", "T"), resume::Bool = false)
    cb = GeodataOutputCallback(output_dir, sim.model; fields = fields, resume = resume)
    Oceananigans.add_callback!(sim, Oceananigans.Callback(cb))
    store = output_zarr_path(output_dir)
    write_grid_metadata!(store, sim.model)
    return store
end

"""
    write_grid_metadata!(store, model)

Record the grid's interior cell centres in the store's own attributes.

Written, not derived, because this is the moment the information exists: a reader has only a
path, and a store that does not say where its cells are on Earth cannot be interpolated
against. This is what replaced the sidecar -- the same coordinates now travel *inside* the
store instead of in a second file next to it.
"""
function write_grid_metadata!(store::AbstractString, model)
    ax = _field_arrays(model)
    zone = joinpath(store, ".zattrs")
    existing = isfile(zone) ? JSON3.read(read(zone, String)) : Dict{String, Any}()
    out = Dict{String, Any}(string(k) => v for (k, v) in existing)
    out["lon"] = ax.x
    out["lat"] = ax.y
    out["depth"] = ax.z
    d = length(ax.x) > 1 ? ax.x[2] - ax.x[1] : 0.0
    e = length(ax.y) > 1 ? ax.y[2] - ax.y[1] : 0.0
    out["dlon"] = d
    out["dlat"] = e
    open(zone, "w") do io
        JSON3.write(io, out)
    end
    return store
end
