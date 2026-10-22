using Pkg
Pkg.activate("C:/Users/Nathalie/projects/GeoOceanModel")

using GeoOceanModel
using GeoData
using GeoDataSources
using CairoMakie
using Statistics
import CairoMakie: record

# =============================================================================
#  Hydrodynamic Animation Generator
#
#  Renders one variable, at one depth, over a run's stored time steps.
#
#    julia --project=. animate_hydrodynamic_variable.jl [variable] [depth] [store]
#
#    variable : temperature | salinity | u | v | w | speed   (default: temperature)
#    depth    : metres below the surface (50, -50) or "surface" (default: surface)
#    store    : path to the GeoData store                     (default: the Scotian Shelf store)
#
#  The depth is a depth in metres, not a level number. The store's depth axis runs
#  deepest first -- level 1 is the deepest layer -- so a level number would mean
#  something different in every store. The surface layer is the one nearest z = 0,
#  whichever end of the axis it sits on.
# =============================================================================

const DEFAULT_STORE = "C:/Users/Nathalie/projects/lakehouse/GeoOceanModel/scotian_shelf/hydrodynamics.zarr"

const SUPPORTED_VARIABLES = (:temperature, :salinity, :u, :v, :w, :speed, :flow)

const FPS = 2                    # frames per second
const N_FRAMES = nothing         # nothing = use all available time steps (after time-interval selection)
const TIME_INTERVAL_DAYS = 1.0   # default days between animation frames
# The figure's size, in pixels. The flow arrows are drawn in pixels and converted back
# to data units, so the geometry needs it.
const FIG_SIZE = (1200, 600)
# The length of the longest flow arrow, in pixels. An arrow of unstated length is a
# decoration: the same picture would be drawn by a 1 cm/s current and a 50 cm/s one.
# The scale is printed on the figure, so the reader can convert it.
const ARROW_PIXELS = 60.0
# The arrows are drawn on a stride: one arrow per ARROW_STRIDE cells.
const ARROW_STRIDE = 20

# The variables the reader can draw, with the field they come from and the factor
# that puts them in display units.
const VARIABLE_SPEC = Dict(
    :temperature => (:temperature, 1.0, "Temperature (°C)"),
    :salinity    => (:salinity, 1.0, "Salinity (PSU)"),
    :u           => (:u, 100.0, "Zonal velocity (cm/s)"),
    :v           => (:v, 100.0, "Meridional velocity (cm/s)"),
    :w           => (:w, 100.0, "Vertical velocity (cm/s)"),
    :speed       => (:speed, 100.0, "Speed (cm/s)"),
    :flow        => (:flow, 100.0, "Flow"),
)

function usage()
    println("""
    Usage: julia --project=. animate_hydrodynamic_variable.jl [variable] [depth] [store] [days_per_frame]

      variable      : $(join(SUPPORTED_VARIABLES, " | "))   (default: temperature)
      depth         : metres below the surface (50, -50), "surface", or "bottom"   (default: surface)
      store         : path to the GeoData store   (default: $(DEFAULT_STORE))
      days_per_frame: days between animation frames   (default: $TIME_INTERVAL_DAYS)

      Flags: --bottom, --interval-days=N, --variable=..., --store=..., --depth=...
      "flow" draws the horizontal current as arrows over the speed.
    """)
end

"""
    parse_variable(name) -> Symbol

The variable to draw. An unknown name is an error with the list of the ones that
exist: an animation of a field that is not there is not a thing to produce.
"""
function parse_variable(name::AbstractString)
    v = Symbol(lowercase(strip(name)))
    v in SUPPORTED_VARIABLES || error(
        "Unknown variable '$(name)'. Available: $(join(SUPPORTED_VARIABLES, ", ")).")
    return v
end

"""
    parse_depth(text) -> Union{Nothing, Float64}

The depth to draw, in metres below the surface. `nothing` is the surface layer --
the one nearest z = 0 in the store's own depth axis.
"""
function parse_depth(text::AbstractString)
    t = lowercase(strip(text))
    (t == "surface" || t == "surf" || isempty(t)) && return nothing
    z = tryparse(Float64, t)
    isnothing(z) && error("Could not read '$(text)' as a depth. Give metres (50, -50) or 'surface'.")
    return z
end

"""
    bottom_level(depths, bathy, i, j) -> Union{Int, Nothing}

The index of the deepest wet vertical level at horizontal cell (i, j), using the
seabed height when available. The store's depth axis runs deepest first, so this
is the first level whose depth is at or above the seabed.
"""
function bottom_level(depths, bathy, i::Int, j::Int)
    bval = if bathy isa AbstractMatrix && size(bathy, 1) >= i && size(bathy, 2) >= j
        bathy[i, j]
    else
        nothing
    end
    for (kk, z) in enumerate(depths)
        if !isfinite(z)
            continue
        end
        if bval === nothing || !isfinite(bval) || z >= bval
            return kk
        end
    end
    return nothing
end

"""
    bottom_matrix(ds, arr3d::AbstractArray{<:Real,3}) -> Matrix{Float64}

Extract the bottom-most value of a 3D field, cell by cell, rather than at one
uniform depth level.

When the store carries bathymetry, the deepest wet level is the first level whose
depth is at or above the seabed. When bathymetry is absent (as in some GeoData
stores), the deepest level holding a finite (non-NaN) value is used instead --
levels below the seabed hold no real water and must not be mistaken for the bottom.
"""
function bottom_matrix(ds, arr3d::AbstractArray{<:Real,3})
    depths = ds.depths
    bathy = hasproperty(ds, :bathymetry) ? ds.bathymetry : nothing
    nx, ny = size(arr3d, 1), size(arr3d, 2)
    nz = size(arr3d, 3)
    out = fill(NaN, nx, ny)
    for i in 1:nx, j in 1:ny
        kb = if bathy isa AbstractMatrix && size(bathy, 1) >= i && size(bathy, 2) >= j
            bval = bathy[i, j]
            _bottom_level_by_bathy(depths, bval)
        else
            # No bathymetry: find the deepest level holding a finite (wet) value.
            _bottom_level_by_field(arr3d, i, j, nz)
        end
        isnothing(kb) && continue
        out[i, j] = Float64(arr3d[i, j, kb])
    end
    return out
end

"""
    _bottom_level_by_bathy(depths, bval) -> Union{Int, Nothing}

Deepest wet level from seabed height `bval`. The depth axis runs deepest first,
so this is the first level whose depth is at or above the seabed (z >= bval).
"""
function _bottom_level_by_bathy(depths::AbstractVector{<:Real}, bval::Real)
    for (kk, z) in enumerate(depths)
        if isfinite(z) && z >= bval
            return kk
        end
    end
    return nothing
end

"""
    _bottom_level_by_field(arr3d, i, j, nz) -> Union{Int, Nothing}

Deepest level holding a finite value at cell (i, j), scanning deepest first.
Used when bathymetry is unavailable; levels below the seabed hold no real water.
"""
function _bottom_level_by_field(arr3d::AbstractArray{<:Real,3}, i::Int, j::Int, nz::Int)
    for k in 1:nz
        v = Float64(arr3d[i, j, k])
        if isfinite(v)
            return k
        end
    end
    return nothing
end
    return out
end

"""
    _load_coastline(store) -> Vector{NamedTuple}

The coastline rings to draw over the field: the run's own fetched coastline when the run
has one -- the same land the model ran on -- and the region's bundled rings otherwise.

Each ring is `(name, code, lons, lats)`.
"""
function _load_coastline(store::AbstractString)
    candidates = AbstractString[
        joinpath(dirname(store), "inputs", "coastline.parquet"),
        joinpath(dirname(store), "inputs", "coastline.dat"),
    ]
    for c in candidates
        isfile(c) || continue
        endswith(c, ".parquet") || continue
        rings = GeoData.load_coastline_polygons(c)
        isempty(rings) || return rings
    end
    return GeoDataSources.SCOTIAN_SHELF_COASTLINE
end

function main()
    if any(a -> lowercase(a) in ("-h", "--help"), ARGS)
        usage()
        return nothing
    end

    positional = String[]
    time_interval_days = TIME_INTERVAL_DAYS
    bottom = false
    variable_override = nothing
    store_override = nothing
    depth_override_text = nothing
    i = 1
    while i <= length(ARGS)
        tok = String(ARGS[i])
        if startswith(tok, "--") || contains(tok, "=")
            if contains(tok, "=")
                parts = split(tok, "=", limit = 2)
                key = replace(String(parts[1]), r"^[-]+" => "")
                val = String(parts[2])
                advance = 1
            else
                key = replace(tok, r"^[-]+" => "")
                if i < length(ARGS) && !startswith(String(ARGS[i + 1]), "-") && !contains(String(ARGS[i + 1]), "=")
                    val = String(ARGS[i + 1])
                    advance = 2
                else
                    val = "true"
                    advance = 1
                end
            end
            lk = lowercase(replace(key, "_" => "-"))
            if lk in ("interval", "time-interval", "interval-days", "days-per-frame", "frame-interval", "time-step", "timestep")
                time_interval_days = parse(Float64, val)
            elseif lk in ("bottom", "bottom-level", "benthic", "seabed")
                bottom = lowercase(strip(val)) in ("true", "1", "yes", "") ? true : parse(Bool, val)
            elseif lk == "depth"
                if lowercase(strip(val)) in ("bottom", "bottom-level", "seabed", "benthic")
                    bottom = true
                else
                    depth_override_text = val
                end
            elseif lk == "variable"
                variable_override = parse_variable(val)
            elseif lk == "store"
                store_override = val
            elseif lk in ("interval-days",)
                time_interval_days = parse(Float64, val)
            end
            i += advance
        else
            push!(positional, tok)
            i += 1
        end
    end

    variable = if !isnothing(variable_override)
        variable_override
    elseif isempty(positional)
        :temperature
    else
        parse_variable(positional[1])
    end

    depth_text = if !isnothing(depth_override_text)
        depth_override_text
    elseif length(positional) < 2
        "surface"
    else
        positional[2]
    end
    if lowercase(strip(depth_text)) in ("bottom", "bottom-level", "seabed", "benthic")
        bottom = true
        depth = nothing
    else
        depth = parse_depth(depth_text)
    end

    store = if !isnothing(store_override)
        store_override
    elseif length(positional) < 3
        DEFAULT_STORE
    else
        positional[3]
    end
    if length(positional) >= 4
        time_interval_days = parse(Float64, positional[4])
    end
    isdir(store) || error("No store at $(store).")

    # The output is named for what it holds, so a run of one variable at one depth does
    # not overwrite the run of another.
    output_dir = joinpath(dirname(store), "outputs")
    mkpath(output_dir)
    depth_tag = bottom ? "bottom" : (isnothing(depth) ? "surface" : replace(string(abs(depth)), "." => "p"))
    output_path = joinpath(output_dir, "$(variable)_$(depth_tag)_animation.mp4")

    field, factor, var_label = VARIABLE_SPEC[variable]

    println("="^60)
    println("GeoOceanModel — Hydrodynamic Animation Generator")
    println("="^60)
    println("Store:      $store")
    println("Variable:   $variable")
    println("Depth:      ", bottom ? "ocean bottom" : (isnothing(depth) ? "the surface layer" : "$(depth) m"))
    println("Time step:  $time_interval_days day(s) per frame")
    println("FPS:        $FPS")
    println("Output:     $output_path")
    println("="^60)

    # --- Discover number of time steps in store ---
    println("\nDiscovering time steps in store...")
    # The store's own time axis is the count. Probing for it -- reading one interval at
    # a time until the read fails -- re-opens the whole store per interval, so a run
    # with hundreds of intervals pays hundreds of full reads to learn what the store
    # already states.
    n_times = 0
    times_all = Float64[]
    try
        probe = GeoData.geoload(store; backend = :zarr)
        if haskey(probe.dims, :time)
            n_times = probe.dims[:time].size
        end
        if hasproperty(probe, :coords) && haskey(probe.coords, :time)
            times_all = collect(Float64, probe.coords[:time].data)
        end
    catch err
        println("Could not read the store's time axis: $(sprint(showerror, err))")
    end

    if n_times == 0
        error("No time steps found in store at $store")
    end

    println("Found $n_times time steps in store")

    # --- Get grid dimensions from first time step ---
    ds_first = GeoOceanModel.extract_hydrodynamic_dataset(store; time_index = 1, depth = depth)
    lons = ds_first.lons
    lats = ds_first.lats
    depths = ds_first.depths
    # The level the reader resolved: the surface layer unless a depth was named. Taken
    # from the reader rather than computed here, so the figure and the data cannot
    # disagree about which layer this is.
    k = if bottom
        argmin(ds_first.depths)
    else
        ds_first.depth_index
    end
    depth_m = ds_first.depths[k]

    println("Grid: $(length(lons)) × $(length(lats)) × $(length(depths))")
    println("Depth: level $k of $(length(depths)) = $(round(depth_m, digits=1)) m " *
            "(the store's depth axis runs deepest first)")

    # Determine frame indices
    selected = if isempty(times_all) || length(times_all) != n_times
        collect(1:n_times)
    else
        interval_sec = time_interval_days * 86400.0
        out = Int[1]
        last_t = times_all[1]
        for i in 2:n_times
            if isfinite(times_all[i]) && times_all[i] >= last_t + interval_sec - 1e-6
                push!(out, i)
                last_t = times_all[i]
            elseif times_all[i] < last_t
                last_t = times_all[i]
            end
        end
        length(out) < 2 ? collect(1:n_times) : out
    end
    if N_FRAMES === nothing
        frame_indices = selected
    else
        n_frames = min(N_FRAMES, length(selected))
        frame_indices = selected[round.(Int, range(1, length(selected), length = n_frames))]
    end
    total_frames = length(frame_indices)
    println("Frames to render: $total_frames")

    # --- Helper: the store's slice at the resolved level ---
    #
    # One read per frame: the store is read whole, and the field, the flow and the time
    # all come out of that one read. Reading it twice per frame -- once for the field
    # and once for the arrows -- doubled the cost of every animation for nothing.
    function get_slice(i::Int)
        return GeoOceanModel.extract_hydrodynamic_dataset(store; time_index = i, depth_level = k)
    end

    # The field at the resolved level, in display units.
    function field_of(ds)
        if bottom
            if field === :speed || field === :flow
                u = bottom_matrix(ds, Float64.(ds.u)) .* factor
                v = bottom_matrix(ds, Float64.(ds.v)) .* factor
                return sqrt.(u .^ 2 .+ v .^ 2)
            end
            return bottom_matrix(ds, Float64.(ds[Symbol(field)])) .* factor
        end
        if field === :speed || field === :flow
            u = Float64.(ds.u[:, :, k]) * factor
            v = Float64.(ds.v[:, :, k]) * factor
            return sqrt.(u .^ 2 .+ v .^ 2)
        end
        return Float64.(ds[Symbol(field)][:, :, k]) * factor
    end

    # --- Get first frame data for color range ---
    first_var = field_of(get_slice(first(frame_indices)))
    finite_first = filter(isfinite, first_var)
    cmin, cmax = if !isempty(finite_first)
        cmin0, cmax0 = minimum(finite_first), maximum(finite_first)
        cspan = cmax0 - cmin0
        if cspan > 0
            (cmin0 - 0.05 * cspan, cmax0 + 0.05 * cspan)
        else
            (cmin0 - 1.0, cmax0 + 1.0)
        end
    else
        (-1.0, 1.0)
    end

    # --- The coastline ---
    #
    # A field without the land beside it is a field with no geography: a reader cannot
    # tell whether the pale area is a bay, a bank, or the edge of the data. The run's own
    # coastline is used when it has one -- the same land the model ran on -- and the
    # region's bundled rings otherwise.
    coast = _load_coastline(store)
    if isempty(coast)
        println("No coastline found for this store; the land is not drawn.")
    else
        println("Coastline: $(length(coast)) rings")
    end

    # --- Create figure ---
    fig = Figure(size = FIG_SIZE, fontsize = 12)

    ax = Axis(
        fig[1, 1],
        title = "",
        xlabel = "Longitude (°E)",
        ylabel = "Latitude (°N)",
        limits = (minimum(lons), maximum(lons), minimum(lats), maximum(lats)),
        titlesize = 14,
        xlabelsize = 11,
        ylabelsize = 11
    )

    # Initialize heatmap with Observable
    init_data = fill(NaN, length(lons), length(lats))
    hm_data = Observable(init_data)
    hm = heatmap!(ax, lons, lats, hm_data;
                  colormap = :thermal,
                  colorrange = (cmin, cmax),
                  interpolate = false)

    # The colour bar is attached to the heatmap. Left unattached it shows its own
    # default range -- 0 to 1 -- while the field is in centimetres per second, so the
    # legend read as a scale the picture did not have.
    #
    # No width or height is given: in this Makie those are RELATIVE sizes, and the
    # `width = 20` this used to carry asked for a bar twenty times the figure's width.
    # The layout then squeezed the axis to a line, and every frame of the animation was
    # a horizontal stripe with no latitude axis at all.
    field_label = field === :flow ? "Speed (cm/s)" : var_label
    cb = Colorbar(fig[1, 2], hm; label = field_label)
    fig[1, 1] = ax
    fig[1, 2] = cb

    # The coastline, drawn over the field.
    for ring in coast
        lines!(ax, ring.lons, ring.lats; color = :white, linewidth = 1.4)
    end

    title_lbl = Label(fig[1, 1, Top()], "", fontsize = 14, font = :bold,
                      halign = :center, valign = :bottom)

    # --- The flow arrows, for the vector mode ---
    #
    # A vector field is only a vector field if it is drawn in the plane. The store holds
    # (u, v) as separate fields; only the pair is a vector, and only the pair is drawn.
    #
    # The arrow geometry is computed in PIXELS and converted back to data units, because
    # the plot's geography is not square: a degree of longitude is about 80 km at this
    # latitude and a degree of latitude about 111 km, and the figure's own aspect makes
    # a degree of latitude taller than a degree of longitude. Working in pixels is what
    # keeps an arrow's direction true on the screen; working in degrees alone would tilt
    # every one of them.
    arrow_pts = Point2f[]
    arrow_dirs = Observable(Vec2f[])
    arrow_scale_note = ""
    if field === :flow
        lon_idx = collect(1:ARROW_STRIDE:length(lons))
        lat_idx = collect(1:ARROW_STRIDE:length(lats))
        for i in lon_idx, j in lat_idx
            push!(arrow_pts, Point2f(lons[i], lats[j]))
        end
        m_per_deg_lat = 111320.0
        m_per_deg_lon = 111320.0 * cosd(mean(lats))
        px_per_deg_lon = FIG_SIZE[1] / (maximum(lons) - minimum(lons))
        px_per_deg_lat = FIG_SIZE[2] / (maximum(lats) - minimum(lats))
        # The scale: the fastest arrow spans ARROW_PIXELS, and every other arrow is
        # proportional to it. Stated on the figure, because an arrow of unstated length
        # is a decoration.
        ds0 = get_slice(first(frame_indices))
        if bottom
            u0 = bottom_matrix(ds0, Float64.(ds0.u)) .* factor
            v0 = bottom_matrix(ds0, Float64.(ds0.v)) .* factor
        else
            u0 = Float64.(ds0.u[:, :, k]) * factor
            v0 = Float64.(ds0.v[:, :, k]) * factor
        end
        sp0 = sqrt.(u0 .^ 2 .+ v0 .^ 2)
        sp_max = maximum(filter(isfinite, sp0))
        # The same field in pixels: the reference for every arrow's length.
        sp_px_max = 0.0
        for i in 1:size(u0, 1), j in 1:size(u0, 2)
            sx = u0[i, j] / m_per_deg_lon * px_per_deg_lon
            sy = v0[i, j] / m_per_deg_lat * px_per_deg_lat
            (isfinite(sx) && isfinite(sy)) || continue
            sp_px_max = max(sp_px_max, hypot(sx, sy))
        end
        arrow_scale_note = "arrows: longest = $(round(sp_max, digits=1)) cm/s " *
                           "($(ARROW_PIXELS) px)"
        if isdefined(Makie, :arrows2d!)
            arrows2d!(ax, arrow_pts, arrow_dirs;
                      tipwidth = 6, tiplength = 10, lengthscale = 1.0,
                      tipcolor = :white, shaftcolor = :white)
        else
            arrows!(ax, arrow_pts, arrow_dirs;
                    tipwidth = 6, tiplength = 10, lengthscale = 1.0,
                    tipcolor = :white, shaftcolor = :white)
        end
    end

    # --- Animation frame function ---
    function make_frame(i::Int)
        ds = get_slice(i)
        t_hr = Float64(ds.time_seconds) / 3600.0
        hm_data[] = field_of(ds)

        if field === :flow
        if bottom
            uu = bottom_matrix(ds, Float64.(ds.u)) .* 100.0
            vv = bottom_matrix(ds, Float64.(ds.v)) .* 100.0
        else
            uu = Float64.(ds.u[:, :, k]) * 100.0
            vv = Float64.(ds.v[:, :, k]) * 100.0
        end
            dirs = Vec2f[]
            for (n, _) in enumerate(arrow_pts)
                i_idx = (n - 1) ÷ length(lat_idx) + 1
                j_idx = (n - 1) % length(lat_idx) + 1
                u_val = uu[lon_idx[i_idx], lat_idx[j_idx]]
                v_val = vv[lon_idx[i_idx], lat_idx[j_idx]]
                if !isfinite(u_val) || !isfinite(v_val)
                    push!(dirs, Vec2f(0.0, 0.0))
                    continue
                end
                # On screen: east and north in pixels, then scaled by the speed.
                sx = u_val / m_per_deg_lon * px_per_deg_lon
                sy = v_val / m_per_deg_lat * px_per_deg_lat
                sp = hypot(sx, sy)
                length_px = sp > 0 ? (sp / sp_px_max) * ARROW_PIXELS : 0.0
                push!(dirs, Vec2f(sx / max(sp, 1e-12) * length_px / px_per_deg_lon,
                                  sy / max(sp, 1e-12) * length_px / px_per_deg_lat))
            end
            arrow_dirs[] = dirs
        end

        depth_desc = bottom ? "ocean bottom" : "$(round(depth_m, digits=1)) m depth"
        title_lbl.text[] = "$var_label at $depth_desc — " *
                           "Time: $(round(t_hr, digits=1)) h  (Step $i / $n_times)" *
                           (isempty(arrow_scale_note) ? "" : "\n$arrow_scale_note")

        return fig
    end

    # --- Record animation ---
    println("\nRecording animation...")
    t_start = time()

    record(fig, output_path, frame_indices; framerate = FPS) do i
        make_frame(i)
        if i % 10 == 0 || i == total_frames
            println("  Frame $i / $total_frames  ($(round(time() - t_start, digits=1)) s elapsed)")
        end
    end

    t_elapsed = time() - t_start
    println("\n" * "="^60)
    println("Animation complete!")
    println("Output:   $output_path")
    println("Frames:   $total_frames")
    println("Time:     $(round(t_elapsed, digits=1)) s")
    println("="^60)
end

main()
