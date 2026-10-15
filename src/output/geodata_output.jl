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

An Oceananigans `Callback` that writes the ocean state into a store at each firing.

One store per run, appended to at each interval, so a long simulation writes incrementally
rather than holding its history in memory and losing it on a crash.

That store is a **GeoData** store -- the variables are a `GeoDataset`'s, with their axes and
dimension records, opened once at the first output and written into slice by slice. This is
what makes a reader use `geoload` and the dataset's own `.variables`/`.dims` rather than
walking Zarr: an earlier version wrote each interval to `T/<m>/`, and GeoData's Zarr reader
discovers variables as the root group's *child arrays*, so that store opened as an empty
dataset and every consumer -- the larval half and the figures -- hand-rolled its own reads.

`store_ds` and `time_array` are held state, which is why this is a `mutable struct`: the store
is built at the first output and the same dataset is written into at every later one, so the
callback must remember the handle it opened.
"""
mutable struct GeodataOutputCallback
    store::String
    output_dir::String
    field_names::Vector{String}
    written::Base.RefValue{Int}
    store_ds::Union{GeoDataset, Nothing}
    # The store's own time array, held from the group it was opened with. `nothing` until the
    # first output, because the store does not exist until then.
    time_array::Union{Zarr.ZArray, Nothing}
    # The output interval in simulated seconds, from the schedule the caller attached. A field
    # rather than a scan of the simulation's callbacks: at the first firing the callback being
    # fired is not necessarily in `sim.callbacks` yet, and scanning finds nothing -- which sizes
    # the time axis to one interval and silently truncates the run's output.
    interval::Float64
    # Spin-up interval in seconds. Time-slices prior to this are excluded from storage.
    spinup_seconds::Float64
end

"""
    GeodataOutputCallback(output_dir, model; fields = ("u","v","w","T"), resume = false)

Construct the output writer for one run. A fresh run clears the store and starts at slice 0.
`resume = true` keeps the store and continues from however many slices it already holds, so an
extended/interrupted run appends to its own output instead of re-downloading nothing and
overwriting the interval it already wrote.
"""
function GeodataOutputCallback(output_dir::AbstractString, interval::Real,
                               model; fields::Tuple = ("u", "v", "w", "T"),
                               resume::Bool = false, spinup_seconds::Real = 0.0)
    isdir(output_dir) || mkpath(output_dir)
    store = output_zarr_path(output_dir)
    if !resume && ispath(store)
        rm(store; recursive = true, force = true)   # a fresh run overwrites its own output
    end
    mkpath(store)

    start = 1
    if resume
        start = GeoOceanModel.store_interval_count(store) + 1
    end
    return GeodataOutputCallback(store, output_dir, collect(String.(fields)), Ref(start),
                                 nothing, nothing, Float64(interval), Float64(spinup_seconds))
end

"""
    _interval_count(sim, cb) -> Int

How many output intervals this run will write, so the store's time axis can be sized once
instead of growing an array that cannot be appended to. Intervals falling strictly within
the spin-up period [0, spinup_seconds) are excluded from the output store.
"""
function _interval_count(sim::Oceananigans.Simulation, cb)
    stop = Float64(sim.stop_time)
    effective_duration = max(0.0, stop - cb.spinup_seconds)
    cb.interval > 0 || return 1
    effective_duration > 0 || return 1
    return max(1, ceil(Int, effective_duration / cb.interval) + 1)
end

"""
    _build_store_dataset(store, ax, names, nt, model) -> GeoDataset

The run's output store, created as a GeoData `GeoDataset` and written once so each interval is
a slice of an already-open array.

Axes, not offsets: `ax` is the interior cell centres in `(lon, lat, depth)` from the model's
own grid, so the dataset records where its cells are rather than leaving a reader to infer
positions from indices. `nt` sizes the time axis to the run.

The variables are the named fields, each `(lon, lat, depth, time)`, created empty and filled
by the callback. `ds` is returned open, and writing `v.data[:, :, :, m] = field` goes
straight to the Zarr array behind it, so nothing is held in memory.
"""
function _build_store_dataset(store::AbstractString, ax, names::Vector{String}, nt::Int,
                              model)
    crs = GeoData.CoordinateSystem(crs = "EPSG:4326")
    nx, ny, nz = length(ax.x), length(ax.y), length(ax.z)

    axe(name, coords, units, std, dim_type) = Dimension(
        name = name, size = length(coords), coords = coords,
        units = units, standard_name = std, dim_type = dim_type)

    dl = axe(:lon, ax.x, "degrees_east", "longitude", GeoData.DIM_SPATIAL)
    dt = axe(:lat, ax.y, "degrees_north", "latitude", GeoData.DIM_SPATIAL)
    dz = axe(:depth, ax.z, "m", "depth", GeoData.DIM_PARAMETRIC)
    tim = axe(:time, Float64.(1:nt), "s", "time", GeoData.DIM_TEMPORAL)

    vars = Dict{String, GeoArray}()
    for name in names
        d = dims = (dl, dt, dz, tim)
        vars[name] = GeoArray(zeros(Float64, nx, ny, nz, nt), d, crs, Dict{String, Any}())
    end
    coords = Dict{Symbol, GeoArray}(
        :lon => GeoArray(ax.x, (dl,), crs, Dict{String, Any}()),
        :lat => GeoArray(ax.y, (dt,), crs, Dict{String, Any}()),
        :depth => GeoArray(ax.z, (dz,), crs, Dict{String, Any}()),
        :time => GeoArray(tim.coords, (tim,), crs, Dict{String, Any}()))

    ds = GeoDataset(vars, coords, Dict(:lon => dl, :lat => dt, :depth => dz, :time => tim), crs,
                    Dict{String, Any}(), GeoData.ZarrBackend(), String(store))
    GeoData.geosave(String(store), ds; backend = :zarr, overwrite = true)
    # Reopened for write, because the callback writes into this dataset's arrays: `geoload`
    # defaults to read-only, and of the Zarr modes only `"w"` hands out a writeable array
    # here. It does not truncate -- a second `"w"` open appends, which is checked explicitly,
    # because a writer that silently cleared its earlier intervals would produce a run whose
    # first slice is missing.
    return GeoData.geoload(String(store); backend = :zarr, mode = "w")
end

"""
    _to_centres(arr, grid_shape)

A field written on its own staggered grid, moved onto cell centres by averaging the two faces
that bracket each centre. A field already on the centre grid is returned unchanged.

This is where staggering is resolved, once, on the write. `u` arrives as 346×245×13 for a
345×245×13 grid (one more x-face), `v` as 345×246×13, and `w` as 345×245×14, while `T` is
already centred -- Oceananigans' own staggering. A dimension that is one longer than the grid
is a face axis and is averaged; anything else must agree, or the field is not of this grid.

Which axes move is measured rather than looked up, and averaging rather than dropping the last
face matters because dropping shifts the field by half a cell in the grid's own coordinates, and
a shifted current is a different current. Doing it here means a reader -- the larval half, the
figures -- holds one grid and needs no knowledge of staggering at all.
"""
function _to_centres(arr::AbstractArray{T, 3}, grid_shape::Tuple{Int, Int, Int}) where {T}
    out = arr
    for d in 1:3
        if size(out, d) == grid_shape[d] + 1
            out = 0.5 .* (selectdim(out, d, 1:(size(out, d) - 1)) .+
                          selectdim(out, d, 2:size(out, d)))
        end
    end
    size(out) == grid_shape || error(
        "Field of size $(size(arr)) on a $(grid_shape) grid: not this grid's field, and not " *
        "its faces either.")
    return out
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
    # stripped, with the vertical centre from `znode` as above. `_host_copy` rather than a
    # comprehension over the range, for the reason in its docstring.
    ix = (base.Hx + 1):(base.Hx + base.Nx)
    iy = (base.Hy + 1):(base.Hy + base.Ny)
    x = Float64.(_host_copy(base.xᶜᵃᵃ)[ix])
    y = Float64.(_host_copy(base.yᵃᶜᵃ)[iy])
    z = [Float64(Oceananigans.Grids.znode(k, base, Oceananigans.Grids.Center()))
         for k in 1:base.Nz]
    return (x = x, y = y, z = z)
end

function (cb::GeodataOutputCallback)(sim::Oceananigans.Simulation)
    model = sim.model
    t = Float64(model.clock.time)
    if t < cb.spinup_seconds
        return nothing
    end
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
        # `u` is on x-faces, `v` on y-faces, `w` on z-faces: Oceananigans' own staggering. They
        # are averaged onto cell centres here, in one `_to_centres` call, so the store holds
        # every field on the one grid its dimension records describe -- and a reader does not
        # have to know about staggering to read the store.
        fields[name] = _to_centres(Array{Float64}(interior(f)),
                                   (length(ax.x), length(ax.y), length(ax.z)))
    end

    ds = cb.store_ds
    if isnothing(ds)
        # First interval: the store is created as a GeoData store, with the time axis sized to
        # the run. Written through GeoData because a store written by hand -- one directory
        # per interval -- is one GeoData's `geoload` cannot open: its reader discovers
        # variables as the root group's *child arrays*, so that store opens as an empty dataset
        # and every consumer -- the larval half and the figures -- then hand-rolls its own
        # Zarr walk. That is exactly what they were doing.
        nt = _interval_count(sim, cb)
        ds = _build_store_dataset(output_zarr_path(cb.output_dir), ax, cb.field_names, nt,
                                  sim.model)
        cb.store_ds = ds
        record_run_metadata!(ds, model)
    end
    if isnothing(cb.time_array)
        # The store's own time array, opened once. See the note where it is written: this is
        # the coordinate GeoData's reader materialises, so it is written at the backend.
        #
        # The array is taken from the opened group itself. Re-wrapping it -- `Zarr.ZArray(a)`
        # -- gives an array that reports `writeable = true` and silently writes to nothing:
        # the blob the group holds is re-wrapped into a second handle whose writes go to a
        # copy, so a run's time axis stays the placeholder it was created with.
        g = Zarr.zopen(output_zarr_path(cb.output_dir), "w")
        cb.time_array = g.arrays["time"]
    end

    # `time` lives on the model's clock, not on the Simulation: `Simulation` has no `clock`
    # field, and reaching for one is a FieldError at the first write.
    t = Float64(sim.model.clock.time)
    for (name, arr) in fields
        ds.variables[name].data[:, :, :, m] = arr
    end
    # The time coordinate goes through the store's own array, not through `ds.coords[:time]`:
    # GeoData's Zarr reader materialises a coordinate into a plain host `Vector` when it opens
    # a store, so writing there would update memory the next read never sees. This is the one
    # array GeoData's read path cannot write back, and it is a coordinate rather than science,
    # so it is the one place a backend call belongs.
    cb.time_array[m] = t
    cb.written[] = m + 1
    return nothing
end

"""
    attach_output_callback!(sim, output_dir; fields = ("u","v","w","T"), schedule = nothing,
                            resume = false)

Build the store and attach the writer to `sim`. Returns the store path, so the caller records
it in the handover and the larval half is given a location rather than an object.

`schedule` is a `TimeInterval` in simulated seconds, and it must be given: an `Oceananigans`
`Callback` with no schedule fires on *every* timestep, not on the simulation's output
interval, so a 30-day adaptive run wrote 4200-odd intervals instead of the ~240 the
`output_schedule_seconds` in the TOML asked for. `Oceananigans.Schedule` is not exported in
this version, so the caller passes the interval it already built for the JLD2 writer and both
writers share one cadence by construction rather than by convention.
"""
function attach_output_callback!(sim, output_dir::AbstractString;
                                 fields::Tuple = ("u", "v", "w", "T"),
                                 schedule = nothing, resume::Bool = false,
                                 spinup_seconds::Real = 0.0)
    if isnothing(schedule)
        schedule = Oceananigans.TimeInterval(6 * 3600.0)
    end
    cb = GeodataOutputCallback(output_dir, schedule.interval, sim.model; fields = fields,
                               resume = resume, spinup_seconds = spinup_seconds)
    Oceananigans.add_callback!(sim, Oceananigans.Callback(cb, schedule))
    # The store itself does not exist yet: it is built at the first output, when the grid axes
    # and the field shapes are known. What is returned is where it *will* be, so the caller can
    # record a location and hand the larval half a path rather than an object.
    return output_zarr_path(output_dir)
end

"""
    _host_copy(a)

`a` copied to the host, **with its index space unchanged**.

Not `Array(a)`, which for an offset array drops the offsets and so changes what a physical
index means. And not element-by-element access either: when `a` lives on the GPU — which every
grid coordinate and the immersed seabed do on a GPU machine — that is scalar indexing, and CUDA
refuses it. The whole seabed extraction was written that way, so the run died there on a GPU
and nowhere else, with an error that never mentions the seabed. Copying the storage and
re-wrapping it in the original axes is one bulk device-to-host transfer, and the caller's
indices then mean exactly what they meant before.
"""
_host_copy(a) = OffsetArray(Array{eltype(a)}(parent(a)), axes(a)...)

"""
    seabed_from_model(model) -> Union{Matrix{Float64}, Nothing}

The immersed seabed the simulation actually ran on, as `[i_lon, j_lat]` with the halo
stripped. `nothing` when the model is not on an immersed grid.

This is the same value the JLD2 `serialized/grid` branch of `read_immersed_bathymetry`
returns, extracted from a live model so the store can carry it: a figure that masks cells
against the seabed must see the one the run used, not a re-derived stand-in.
"""
function seabed_from_model(model)
    g = model.grid
    g isa ImmersedBoundaryGrid || return nothing
    ib = g.immersed_boundary
    (hasproperty(ib, :bottom_height) && !isnothing(ib.bottom_height)) || return nothing
    Hx, Hy, Nx, Ny = g.Hx, g.Hy, g.Nx, g.Ny
    return _host_copy(ib.bottom_height)[(Hx + 1):(Hx + Nx), (Hy + 1):(Hy + Ny), 1]
end

"""
    record_run_metadata!(ds, model)

Record what a reader cannot recover from a path, into the dataset's own attributes.

The grid no longer belongs here: the store's dimension records carry the interior cell centres,
in GeoData's own structure, because the store is a GeoData dataset. What is left is the seabed
the run used -- the same value the JLD2 `serialized/grid` branch of
`read_immersed_bathymetry` returns, extracted from the live model so a figure that masks below
the seafloor sees the one the run used and not a re-derived stand-in. It does not vary with
time, and it is not a dimension of any field, so an attribute is where it belongs.
"""
function record_run_metadata!(ds::GeoDataset, model)
    seabed = seabed_from_model(model)
    if !isnothing(seabed)
        ds.attrs["bottom_height"] = seabed
    end
    ds.attrs["title"] = "Realized hydrodynamic simulation"
    ds.attrs["created_utc"] = string(Dates.now(Dates.UTC))
    return nothing
end
