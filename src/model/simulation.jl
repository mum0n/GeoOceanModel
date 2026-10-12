"""
    simulation.jl

Simulation setup, adaptive time integration, numerical stability diagnostics,
and output writing for regional hydrodynamic modeling.
"""

using Oceananigans
using Oceananigans.AbstractOperations: ∂z
using Oceananigans.Fields: indices
using SeawaterPolynomials
using SeawaterPolynomials: TEOS10EquationOfState
import SeawaterPolynomials.ρ
using SpecialFunctions: erf
using Oceanostics: KineticEnergyDissipationRate
using Oceananigans.Units
using Oceananigans.Utils: prettytime
using Oceananigans.OutputWriters: JLD2Writer, Checkpointer, checkpoint
import Oceananigans.OutputWriters: cleanup_checkpoints, write_output!
using JLD2
import OffsetArrays

"""
    to_cpu_offset_array(x)

Safely transfer an array (including GPU `OffsetArray` / `CuArray`) to host memory as a
standard CPU array while preserving index axes and avoiding disallowed GPU scalar indexing.
"""
to_cpu_offset_array(x::OffsetArrays.OffsetArray) =
    OffsetArrays.OffsetArray(Array(parent(x)), axes(x))
to_cpu_offset_array(x::AbstractArray) = Array(x)
to_cpu_offset_array(x) = x

# Import domain constants for fallback coordinate generation

"""
    compute_advective_cfl(
        model,
        Δt::Real;
        bottom_drag::Real = 1e-4,
        cd_drag::Real = 2.5e-3
    ) -> Float64

Estimate the composite Courant-Friedrichs-Lewy (CFL) stability parameter
across the computational domain, evaluating advective velocity transport
in interior cells and the explicit bottom friction / drag Courant limit.

# Mathematical Formulation
```math
\\text{CFL}_{\\text{adv}} = \\max \\left(
    \\frac{|u| \\Delta t}{\\Delta x},
    \\frac{|v| \\Delta t}{\\Delta y},
    \\frac{|w| \\Delta t}{\\Delta z}
\\right)
```
```math
\\text{CFL}_{\\text{drag}} = (r_{\\text{drag}} + C_d |\\boldsymbol{u}_h|) \\Delta t
```
The composite stability index returned is:
```math
\\text{CFL} = \\max(\\text{CFL}_{\\text{adv}}, 0.5 \\, \\text{CFL}_{\\text{drag}})
```
ensuring that adaptive time-step controllers (e.g. `TimeStepWizard`) automatically
throttle \$\\Delta t\$ during intense spring-neap tidal acceleration episodes.

# References
- Canuto, C., et al. (2007). *Spectral Methods*. Springer-Verlag.
"""
# Smallest vertical cell thickness (m).
#
# The CFL must use the SMALLEST cell, never the mean. `Lz / Nz` is the mean, and on a stretched
# grid it badly underestimates the constraint: the two-segment shelf grid reports `Lz/Nz = 125 m`
# while its surface cell is about 5.4 m, a factor of ~23. Using the mean let the adaptive
# time-stepper pick a dt stable for 125 m cells but violently unstable for the surface layer -- the
# run reached max(|u|) = 7.6 m/s at CFL 0.428 against a 0.20 target and diverged. On the previous
# uniform grid the same formula was already ~5.6x optimistic (250 m mean against a 44.7 m cell).
#
# Falls back to `Lz / Nz` if the grid does not expose a cell-centre vector.
function min_vertical_spacing(base_g)
    if hasproperty(base_g, :z) && hasproperty(base_g.z, :cᵃᵃᶠ)
        c = collect(Float64, parent(base_g.z.cᵃᵃᶠ))
        if length(c) > 1
            d = minimum(abs.(diff(c)))
            isfinite(d) && d > 0 && return Float64(d)
        end
    end
    return Float64(base_g.Lz) / base_g.Nz
end

"""
    velocity_peak_location(sim_or_model, loc) -> Tuple{Float64, Float64, Float64}

Extract geographical (longitude, latitude, depth in meters) coordinates of an interior field
index `loc` (a `CartesianIndex` or index tuple) from the simulation or model grid.
Uses `extract_grid_coordinates` to retrieve halo-stripped coordinates matching interior fields.
"""
function velocity_peak_location(sim_or_model, loc)
    grid = hasproperty(sim_or_model, :model) ? sim_or_model.model.grid :
           hasproperty(sim_or_model, :grid) ? sim_or_model.grid : sim_or_model
    lons, lats, depths = extract_grid_coordinates(grid)
    t = Tuple(loc)
    i = max(1, min(t[1], length(lons)))
    j = max(1, min(t[2], length(lats)))
    k = max(1, min(t[3], length(depths)))
    return (Float64(lons[i]), Float64(lats[j]), Float64(depths[k]))
end

"""
    format_geographic_location(lon_deg::Real, lat_deg::Real, depth_m::Real) -> String

Format longitude, latitude, and depth into a clear, unambiguous geographical string
with cardinal direction suffixes (`°W`/`°E`, `°S`/`°N`).
"""
function format_geographic_location(lon_deg::Real, lat_deg::Real, depth_m::Real)
    lon_abs = abs(round(Float64(lon_deg), digits = 2))
    lat_abs = abs(round(Float64(lat_deg), digits = 2))
    dep_val = round(Float64(depth_m), digits = 1)
    lon_dir = lon_deg < 0 ? "°W" : "°E"
    lat_dir = lat_deg < 0 ? "°S" : "°N"
    return "$(lon_abs)$(lon_dir), $(lat_abs)$(lat_dir), $(dep_val)m"
end

function compute_advective_cfl(
    model,
    Δt::Real;
    bottom_drag::Real = 1e-4,
    cd_drag::Real = 2.5e-3
)
    u_max = maximum(abs, interior(model.velocities.u))
    v_max = maximum(abs, interior(model.velocities.v))
    w_int = interior(model.velocities.w)
    w_max = maximum(abs, w_int)

    # Estimate horizontal grid spacing (in meters)
    base_g = model.grid isa ImmersedBoundaryGrid ? model.grid.underlying_grid : model.grid
    dx_approx = if hasproperty(base_g, :radius)
        # Spherical grid: compute minimum zonal spacing at highest absolute latitude
        r_earth = Float64(base_g.radius)
        dlon_deg = minimum(diff(collect(base_g.λᶠᵃᵃ[1:base_g.Nx + 1])))
        lat_max_abs = max(abs(base_g.φᵃᶠᵃ[1]), abs(base_g.φᵃᶠᵃ[base_g.Ny + 1]))
        r_earth * cosd(lat_max_abs) * deg2rad(dlon_deg)
    elseif hasproperty(base_g, :Lx)
        Float64(base_g.Lx) / base_g.Nx
    else
        1000.0
    end
    dy_approx = if hasproperty(base_g, :radius)
        r_earth = Float64(base_g.radius)
        dlat_deg = minimum(diff(collect(base_g.φᵃᶠᵃ[1:base_g.Ny + 1])))
        r_earth * deg2rad(dlat_deg)
    elseif hasproperty(base_g, :Ly)
        Float64(base_g.Ly) / base_g.Ny
    else
        1000.0
    end
    # Smallest vertical cell, NOT Lz/Nz: see `min_vertical_spacing`. This single line is the
    # difference between a stable surface layer and a diverging one on a stretched grid.
    dz_approx = min_vertical_spacing(base_g)

    cfl_x = (u_max * Δt) / max(1.0, dx_approx)
    cfl_y = (v_max * Δt) / max(1.0, dy_approx)

    # Layer-resolved vertical Courant number:
    # On a vertically stretched grid, Δz varies by orders of magnitude (e.g. 10 m at the
    # surface to 3000 m in the abyss). Pairing a deep-ocean vertical velocity peak with the
    # surface layer thickness artificially inflates CFL_z by up to 300x. The physical
    # stability criterion requires evaluating |w_k| against the local layer thickness Δz_k:
    # CFL_z = max_k (max_{i,j}(|w_{i,j,k}|, |w_{i,j,k+1}|) * Δt / Δz_k).
    nz = base_g.Nz
    cfl_z = 0.0
    if hasproperty(base_g, :z) && hasproperty(base_g.z, :Δᵃᵃᶜ)
        dz_raw = base_g.z.Δᵃᵃᶜ
        if dz_raw isa Real
            dz_k = Float64(dz_raw)
            cfl_z = (w_max * Δt) / max(1.0, dz_k)
        elseif dz_raw isa AbstractArray
            dz_vec = to_cpu_offset_array(dz_raw)
            w_int_cpu = Array(w_int)
            for k in 1:nz
                dz_k = Float64(dz_vec[k])
                w_bot = maximum(abs, @view(w_int_cpu[:, :, k]))
                w_top = maximum(abs, @view(w_int_cpu[:, :, k + 1]))
                w_k = max(w_bot, w_top)
                cfl_z = max(cfl_z, (w_k * Δt) / max(1.0, dz_k))
            end
        else
            dz_approx = min_vertical_spacing(base_g)
            cfl_z = (w_max * Δt) / max(1.0, dz_approx)
        end
    else
        dz_approx = min_vertical_spacing(base_g)
        cfl_z = (w_max * Δt) / max(1.0, dz_approx)
    end

    cfl_adv = max(cfl_x, cfl_y, cfl_z)

    # Frictional / quadratic drag Courant number
    speed_h = sqrt(u_max^2 + v_max^2)
    cfl_drag = (bottom_drag + cd_drag * speed_h) * Δt

    return Float64(max(cfl_adv, 0.5 * cfl_drag))
end

"""
    find_latest_checkpoint(
        checkpoint_dir::AbstractString;
        prefix::AbstractString = "checkpoint_\$(resolve_config_name())"
    ) -> Union{Nothing, String}

Search `checkpoint_dir` for serialized Oceananigans model checkpoints matching
`\$(prefix)_iteration*.jld2` and return the path with the highest iteration index.
Falls back to the most recently modified matching checkpoint file if iterations
cannot be parsed from filenames. Returns `nothing` if no checkpoints exist.

# Inputs
- `checkpoint_dir::AbstractString`: Directory containing checkpoint archives.
- `prefix::AbstractString`: Checkpoint filename prefix (default `"checkpoint_\$(resolve_config_name())"`).

# Outputs
- `Union{Nothing, String}`: Path to the latest checkpoint, or `nothing` if none found.
"""
function find_latest_checkpoint(
    checkpoint_dir::AbstractString;
    prefix::AbstractString = "checkpoint_$(resolve_config_name())"
)::Union{Nothing, String}
    if !isdir(checkpoint_dir)
        return nothing
    end
    resolved_prefix = if isempty(strip(prefix)) || prefix == "checkpoint"
        "checkpoint_$(resolve_config_name())"
    else
        String(prefix)
    end
    files = readdir(checkpoint_dir)
    esc_pfx = replace(resolved_prefix, "." => "\\.")
    pattern = Regex("^" * esc_pfx * raw"_iteration(\d+)\.jld2$")
    matches = Tuple{Int, String}[]
    for f in files
        m = match(pattern, f)
        if !isnothing(m)
            iter = tryparse(Int, m.captures[1])
            if !isnothing(iter)
                push!(matches, (iter, joinpath(checkpoint_dir, f)))
            end
        end
    end
    if isempty(matches)
        fallback = filter(f -> startswith(f, resolved_prefix) && endswith(f, ".jld2"), files)
        if isempty(fallback)
            return nothing
        end
        full_paths = joinpath.(checkpoint_dir, fallback)
        return full_paths[argmax(mtime.(full_paths))]
    end
    sort!(matches, by = first)
    return last(matches)[2]
end

"""
    list_candidate_checkpoints(
        checkpoint_dir::AbstractString;
        prefix::AbstractString = "checkpoint_$(resolve_config_name())"
    ) -> Vector{String}

Return all candidate checkpoint file paths in `checkpoint_dir` matching `prefix`,
ordered from highest iteration index (most recent) to lowest.
"""
function list_candidate_checkpoints(
    checkpoint_dir::AbstractString;
    prefix::AbstractString = "checkpoint_$(resolve_config_name())"
)::Vector{String}
    if !isdir(checkpoint_dir)
        return String[]
    end
    resolved_prefix = if isempty(strip(prefix)) || prefix == "checkpoint"
        "checkpoint_$(resolve_config_name())"
    else
        String(prefix)
    end
    files = readdir(checkpoint_dir)
    esc_pfx = replace(resolved_prefix, "." => "\\.")
    pattern = Regex("^" * esc_pfx * raw"_iteration(\d+)\.jld2$")
    matches = Tuple{Int, String}[]
    for f in files
        m = match(pattern, f)
        if !isnothing(m)
            iter = tryparse(Int, m.captures[1])
            if !isnothing(iter)
                push!(matches, (iter, joinpath(checkpoint_dir, f)))
            end
        end
    end
    if isempty(matches)
        fallback = filter(f -> startswith(f, resolved_prefix) && endswith(f, ".jld2"), files)
        isempty(fallback) && return String[]
        full_paths = joinpath.(checkpoint_dir, fallback)
        return sort(full_paths, by = mtime, rev = true)
    end
    sort!(matches, by = first, rev = true)
    return [fp for (_, fp) in matches]
end

"""
    inspect_hydrodynamic_checkpoint(filepath::AbstractString) -> NamedTuple

Inspect an Oceananigans model checkpoint archive on disk, extracting its discrete
iteration number, simulation clock time, serialized prognostic fields, and validity.

# Inputs
- `filepath::AbstractString`: Path to the `.jld2` checkpoint file.

# Outputs
- `NamedTuple`:
  - `exists::Bool`: Whether the target file exists on disk.
  - `valid::Bool`: Whether the file is a readable JLD2 archive with valid metadata.
  - `iteration::Int`: Discrete numerical iteration number recorded at checkpoint.
  - `time::Float64`: Elapsed simulation time in seconds at checkpoint.
  - `fields::Vector{Symbol}`: Prognostic variable symbols found in the archive.
  - `filepath::String`: Normalized file path.
"""
function inspect_hydrodynamic_checkpoint(filepath::AbstractString)
    if !isfile(filepath)
        return (
            exists = false,
            valid = false,
            iteration = 0,
            time = 0.0,
            fields = Symbol[],
            filepath = String(filepath)
        )
    end
    try
        jldopen(filepath, "r") do f
            # Locate the model root group (nested under simulation/model or model or root)
            model_root = if haskey(f, "simulation") && haskey(f["simulation"], "model")
                f["simulation"]["model"]
            elseif haskey(f, "model")
                f["model"]
            else
                f
            end

            iter = 0
            t_val = 0.0

            if haskey(model_root, "clock")
                clk = model_root["clock"]
                if hasproperty(clk, :iteration)
                    iter = Int(clk.iteration)
                end
                if hasproperty(clk, :time)
                    t_val = Float64(clk.time)
                end
            elseif haskey(f, "iteration")
                iter = Int(f["iteration"])
            elseif haskey(f, "clock/iteration")
                iter = Int(f["clock/iteration"])
            end

            if haskey(f, "time") && t_val == 0.0
                t_val = Float64(f["time"])
            elseif haskey(f, "clock/time") && t_val == 0.0
                t_val = Float64(f["clock/time"])
            end

            # Filename fallback for iteration if unparsed
            if iter == 0
                m = match(r"_iteration(\d+)\.jld2$", basename(filepath))
                if !isnothing(m)
                    parsed_iter = tryparse(Int, m.captures[1])
                    if !isnothing(parsed_iter)
                        iter = parsed_iter
                    end
                end
            end

            f_keys = Symbol[]
            for group_name in ("velocities", "tracers")
                if haskey(model_root, group_name)
                    for k in keys(model_root[group_name])
                        push!(f_keys, Symbol(k))
                    end
                elseif haskey(f, group_name)
                    for k in keys(f[group_name])
                        push!(f_keys, Symbol(k))
                    end
                end
            end
            unique!(f_keys)

            return (
                exists = true,
                valid = true,
                iteration = iter,
                time = t_val,
                fields = f_keys,
                filepath = String(filepath)
            )
        end
    catch err
        @warn "Corrupted or unreadable checkpoint archive at $(filepath): $(err)"
        return (
            exists = true,
            valid = false,
            iteration = 0,
            time = 0.0,
            fields = Symbol[],
            filepath = String(filepath)
        )
    end
end


"""
    checkpoint_halos(f, data_size) -> NTuple{3,Int}

Halo widths `(Hx, Hy, Hz)` to exclude when reading a checkpoint's velocity data.

The widths are read from the checkpoint itself, where `save_hydrodynamic_checkpoint` records
them as `file["halo"]`. These grids are built with a halo of `(7, 7, 5)`, not the 3 that a
hardcoded `4:end-3` assumed, so guessing would leave four columns of halo in the "interior" and
four rows of never-evolved cells in the divergence check.

Checkpoints written before the halo was recorded fall back to 3, which is the conservative
choice: it inspects slightly more of the domain rather than silently ignoring a live region.
"""
function checkpoint_halos(f, data_size::Tuple)
    h = get(f, "halo", nothing)
    if !isnothing(h) && length(h) == 3
        try
            widths = Int.(collect(h))
            all(>=(0), widths) && return (widths[1], widths[2], widths[3])
        catch
        end
    end
    return (3, 3, 3)
end

"""
    verify_checkpoint_compatibility(
        filepath::AbstractString,
        expected_u_dim::Tuple;
        divergence_velocity_limit::Real = 20.0
    ) -> Bool

Inspect a JLD2 checkpoint archive for grid compatibility and numerical stability
prior to resuming simulation execution. Validates spatial array dimensions and
bounds checks the physical interior domain velocities (excluding boundary halos).

# Inputs
- `filepath::AbstractString`: Path to the JLD2 checkpoint archive on disk.
- `expected_u_dim::Tuple`: Expected tuple dimensions of the `u` velocity field.
- `divergence_velocity_limit::Real`: Velocity threshold (m/s) indicating numerical instability.

# Outputs
- `Bool`: `true` if checkpoint exists, dimensions match, and interior velocity is stable.
"""
function verify_checkpoint_compatibility(
    filepath::AbstractString,
    expected_u_dim::Tuple;
    divergence_velocity_limit::Real = 20.0
)
    !isfile(filepath) && return false
    is_compat = true
    try
        jldopen(filepath, "r") do f
            if haskey(f, "simulation") && haskey(f["simulation"], "model") &&
               haskey(f["simulation"]["model"], "velocities") &&
               haskey(f["simulation"]["model"]["velocities"], "u") &&
               haskey(f["simulation"]["model"]["velocities"]["u"], "data")

                u_data = f["simulation"]["model"]["velocities"]["u"]["data"]
                cp_dim = size(u_data)
                if cp_dim != expected_u_dim
                    @warn "Checkpoint u dimensions $(cp_dim) differ from model $(expected_u_dim). " *
                          "Ignoring incompatible checkpoint: $(filepath)"
                    is_compat = false
                    return
                end

                # Exclude the boundary halos. The halo width is not fixed at 3: these grids
                # are built with a wider halo (7 in x and y, 5 in z), so a hardcoded
                # `4:end-3` left halo cells in the "interior" and made this check sensitive to
                # a region of the domain that is never physically evolved. Derive the trim from
                # the checkpoint's own grid metadata when it is present, and fall back to
                # trimming a conservative 3 only when it is not.
                halos = checkpoint_halos(f, size(u_data))
                lo = halos .+ 1
                hi = size(u_data) .- halos
                interior_view(d) = (ndims(d) == 3 && all(hi .>= lo)) ?
                    @view(d[lo[1]:hi[1], lo[2]:hi[2], lo[3]:hi[3]]) : d

                max_u = maximum(abs, interior_view(u_data))

                max_v = 0.0
                if haskey(f["simulation"]["model"]["velocities"], "v") &&
                   haskey(f["simulation"]["model"]["velocities"]["v"], "data")
                    max_v = maximum(abs, interior_view(f["simulation"]["model"]["velocities"]["v"]["data"]))
                end

                spd_max = max(max_u, max_v)
                if isnan(spd_max) || isinf(spd_max) || spd_max > divergence_velocity_limit
                    @warn "Checkpoint $(filepath) contains divergent interior velocities " *
                          "(max|u,v| = $(round(spd_max, digits=2)) m/s > limit $(divergence_velocity_limit) m/s). " *
                          "Ignoring corrupt checkpoint."
                    is_compat = false
                end
            else
                is_compat = false
            end
        end
    catch err
        @warn "Failed to inspect checkpoint $(filepath): $(err)"
        is_compat = false
    end
    return is_compat
end

"""
    is_model_velocity_stable(model; divergence_velocity_limit::Real = 20.0) -> (Bool, Float64)

Inspect current physical velocity fields of the model to ensure velocities remain
finite and within physical bounds prior to serializing checkpoints.
"""
function is_model_velocity_stable(model; divergence_velocity_limit::Real = 20.0)
    try
        u_int = interior(model.velocities.u)
        v_int = interior(model.velocities.v)
        nx_c = min(size(u_int, 1), size(v_int, 1))
        ny_c = min(size(u_int, 2), size(v_int, 2))
        nz_c = min(size(u_int, 3), size(v_int, 3))
        
        u_sub = @view u_int[1:nx_c, 1:ny_c, 1:nz_c]
        v_sub = @view v_int[1:nx_c, 1:ny_c, 1:nz_c]
        
        max_u = maximum(abs, u_sub)
        max_v = maximum(abs, v_sub)
        spd_max = max(max_u, max_v)
        
        is_stable = !(isnan(spd_max) || isinf(spd_max) || spd_max > divergence_velocity_limit)
        return (is_stable, Float64(spd_max))
    catch err
        @warn "Error evaluating model velocity stability before checkpoint: $(err)"
        return (false, NaN)
    end
end

"""
    guarded_write_checkpoint!(c::Checkpointer, simulation)

Guarded checkpoint serializer for Oceananigans simulations. Intercepts the write step
to evaluate physical interior velocity stability before serializing prognostic state to
disk. If interior velocities exceed `divergence_velocity_limit`, checkpoint creation is
aborted, preventing numerical divergence from overwriting known valid restart states.
"""
function guarded_write_checkpoint!(c::Checkpointer, simulation)
    iter = iteration(simulation)
    
    # Pre-checkpoint verification gate: check velocity magnitude before writing
    stable, max_spd = is_model_velocity_stable(simulation.model)
    if !stable
        @warn "Rejecting checkpoint at iteration $(iter) (time $(prettytime(simulation))): " *
              "interior velocity magnitude ($(round(max_spd, digits=2)) m/s) diverges."
        return nothing
    end

    filepath = Oceananigans.OutputWriters.checkpoint_path(iter, c)
    t1 = time_ns()
    state = Oceananigans.OutputWriters.prognostic_state(simulation)

    # Write state to temporary file first to ensure atomic commit
    tmp_filepath = filepath * ".tmp"
    try
        jldopen(tmp_filepath, "w") do file
            Oceananigans.OutputWriters.serializeproperty!(file, "simulation", state)
        end
        mv(tmp_filepath, filepath; force = true)
    catch err
        isfile(tmp_filepath) && rm(tmp_filepath; force = true)
        @error "Failed to write checkpoint at iteration $(iter): $(err)"
        return nothing
    end

    t2 = time_ns()
    sz = isfile(filepath) ? filesize(filepath) : 0
    if c.verbose
        @info "Checkpointing done: time=$(prettytime((t2 - t1) * 1e-9)), size=$(Oceananigans.Utils.pretty_filesize(sz))"
    end

    c.cleanup && Oceananigans.OutputWriters.cleanup_checkpoints(c)

    return nothing
end

"""
    guarded_cleanup_checkpoints(checkpointer::Checkpointer)

Prune older intermediate checkpoint files generated during time integration,
retaining the 3 most recently written valid snapshots.
"""
function guarded_cleanup_checkpoints(checkpointer::Checkpointer)
    prefix = Oceananigans.OutputWriters.checkpoint_superprefix(checkpointer.prefix)
    filepaths = Oceananigans.OutputWriters.glob(prefix * "*.jld2", checkpointer.dir)
    
    # Retain the most recent 3 valid checkpoints rather than strictly 1
    if length(filepaths) <= 3
        return nothing
    end

    # Extract iteration numbers from filepaths if available, or sort by mtime
    sort!(filepaths, by = fp -> try
        m = match(r"iteration(\d+)", fp)
        m !== nothing ? parse(Int, m.captures[1]) : Int(mtime(fp))
    catch
        Int(mtime(fp))
    end)

    # Retain the three latest checkpoints for robust rollback
    retained = Set(filepaths[max(1, end - 2):end])

    for filepath in filepaths
        if !(filepath in retained)
            try
                rm(filepath; force = true)
            catch err
                if err isa IOError
                    GC.gc()
                    try
                        rm(filepath; force = true)
                    catch retry_err
                        @warn "Unable to prune intermediate checkpoint $(filepath) (resource locked): $(retry_err)"
                    end
                else
                    rethrow(err)
                end
            end
        end
    end
    return nothing
end

"""
    install_checkpoint_hooks!()

Dynamically install guarded checkpoint writing and retention hooks into Oceananigans
at runtime. This avoids precompilation method-overwrite errors in Julia while ensuring
all checkpoint operations respect the velocity stability gate and 3-checkpoint retention.
"""
function install_checkpoint_hooks!()
    try
        m_write = first(methods(Oceananigans.write_output!, (Checkpointer, Any)))
        Base.delete_method(m_write)
        @eval Oceananigans function write_output!(
            c::Oceananigans.OutputWriters.Checkpointer,
            simulation
        )
            ParticleTracking.guarded_write_checkpoint!(c, simulation)
        end
    catch err
        @warn "Unable to hook Oceananigans.write_output!: $(err)"
    end

    try
        m_clean = first(methods(Oceananigans.OutputWriters.cleanup_checkpoints, (Checkpointer,)))
        Base.delete_method(m_clean)
        @eval Oceananigans.OutputWriters function cleanup_checkpoints(
            checkpointer::Oceananigans.OutputWriters.Checkpointer
        )
            ParticleTracking.guarded_cleanup_checkpoints(checkpointer)
        end
    catch err
        @warn "Unable to hook Oceananigans.OutputWriters.cleanup_checkpoints: $(err)"
    end

    return nothing
end


"""
    inspect_hydrodynamic_file(
        filepath::AbstractString;
        expected_stop_time::Union{Nothing, Real} = nothing
    ) -> NamedTuple

Inspect an Oceananigans JLD2 diagnostic timeseries archive. Verifies structural
integrity, extracts recorded timestamps, and evaluates whether the simulation
reached the planned integration stop time.

# Mathematical Formulation
A simulation archive is marked complete when the latest recorded timestamp
`t_last` satisfies:
```math
t_{\\text{last}} \\ge t_{\\text{stop}} - \\epsilon
```
where `\\epsilon = 10^{-3}` seconds accounts for discrete floating point
time-stepping rounding.

# Inputs
- `filepath::AbstractString`: Path to the `.jld2` timeseries output file.
- `expected_stop_time::Union{Nothing, Real}`: Optional target stop duration in seconds.

# Outputs
- `NamedTuple`:
  - `exists::Bool`: Whether the output file exists on disk.
  - `valid::Bool`: Whether the archive contains readable timeseries groups.
  - `is_complete::Bool`: Whether the archive reached `expected_stop_time`.
  - `last_time::Float64`: Most advanced simulated timestamp in seconds.
  - `n_timesteps::Int`: Total count of recorded diagnostic time levels.
  - `times::Vector{Float64}`: Chronologically ordered timestamps.
  - `filepath::String`: Normalized file path.
"""
function inspect_hydrodynamic_file(
    filepath::AbstractString;
    expected_stop_time::Union{Nothing, Real} = nothing,
    stop_time::Union{Nothing, Real} = nothing
)
    tgt_stop = isnothing(expected_stop_time) ? stop_time : expected_stop_time

    if !isfile(filepath)
        return (
            exists = false,
            valid = false,
            is_complete = false,
            last_time = 0.0,
            n_timesteps = 0,
            n_records = 0,
            times = Float64[],
            filepath = String(filepath)
        )
    end

    try
        jldopen(filepath, "r") do f
            if !haskey(f, "timeseries/t") || !haskey(f, "timeseries/u")
                return (
                    exists = true,
                    valid = false,
                    is_complete = false,
                    last_time = 0.0,
                    n_timesteps = 0,
                    n_records = 0,
                    times = Float64[],
                    filepath = String(filepath)
                )
            end

            t_obj = f["timeseries/t"]
            t_vals = if t_obj isa JLD2.Group
                num_keys = filter(k -> !isnothing(tryparse(Float64, k)), collect(keys(t_obj)))
                sorted_k = sort(num_keys, by = k -> parse(Float64, k))
                [Float64(t_obj[k]) for k in sorted_k]
            else
                collect(Float64, t_obj)
            end

            last_t = isempty(t_vals) ? 0.0 : last(t_vals)
            complete = if isnothing(tgt_stop)
                !isempty(t_vals)
            else
                last_t >= (Float64(tgt_stop) - 1e-3)
            end

            return (
                exists = true,
                valid = true,
                is_complete = complete,
                last_time = last_t,
                n_timesteps = length(t_vals),
                n_records = length(t_vals),
                times = t_vals,
                filepath = String(filepath)
            )
        end
    catch err
        @warn "Failed to inspect hydrodynamic timeseries file at $(filepath): $(err)"
        return (
            exists = true,
            valid = false,
            is_complete = false,
            last_time = 0.0,
            n_timesteps = 0,
            n_records = 0,
            times = Float64[],
            filepath = String(filepath)
        )
    end
end

"""
    fill_field_const(grid, value) -> Field

A centre-point `Field` on `grid` filled with the constant `value` (m² s⁻¹), halo filled.

Used to publish constant closure diffusivity/viscosity as an ordinary gridded output so
consumers can treat it exactly like any other diagnostic field.
"""
function fill_field_const(grid, value::Real)
    f = Field{Center, Center, Center}(grid)
    f .= value
    return f
end

"""
    closure_diffusivity_constants(model) -> Tuple{Union{Nothing, Real}, Union{Nothing, Real}}

Extract the scalar vertical diffusivity `κ` and viscosity `ν` actually used by the model's
closure, if they are constants.

Handles `ScalarDiffusivity` (scalar `κ`/`ν`); returns `(nothing, nothing)` for closures whose
mixing is a prognostic `Field`, which is handled by [`native_mixing_diagnostics`](@ref).
"""
function closure_diffusivity_constants(model)
    hasproperty(model, :closure) || return (nothing, nothing)

    κ, ν = nothing, nothing
    # `model.closure` is a Tuple when several closures are supplied, but a single closure
    # object when the model normalises one (which is what `ocean_simulation` produces).
    closures = model.closure isa Tuple ? model.closure : (model.closure,)
    for c in closures
        (c isa Tuple || c isa NamedTuple) && continue
        hasproperty(c, :κ) || continue
        hasproperty(c, :ν) || continue
        k_raw = getproperty(c, :κ)
        n_raw = getproperty(c, :ν)
        κ_cand = scalar_or_constant(k_raw)
        ν_cand = scalar_or_constant(n_raw)
        κ_cand === nothing && continue
        ν_cand === nothing && continue
        κ = κ_cand
        ν = ν_cand
    end
    return (κ, ν)
end

"""
    scalar_or_constant(x) -> Union{Nothing, Real}

Return `x` as a `Real` when it is a constant diffusivity/viscosity, or when it is a
`NamedTuple` (as `ClimateDataWrangler`/`ScalarDiffusivity` closures present mixing
coefficients) whose entries are all constants. Returns `nothing` when the coefficient is a
grid `Field`, i.e. genuinely spatially varying, since that case is published as a field
rather than a constant.
"""
function scalar_or_constant(x)
    x isa Real && return Float64(x)
    if x isa NamedTuple
        vals = collect(values(x))
        isempty(vals) && return nothing
        all(v -> v isa Real, vals) || return nothing
        return Float64(first(vals))
    end
    return nothing
end

"""
    closure_field_smagarinsky_ν(model) -> Union{Nothing, Any}

Return the Smagorinsky grid eddy-viscosity field `νₑ` from `model.closure_fields` if the
closure carries one, otherwise `nothing`.
"""
function closure_field_smagarinsky_ν(model)
    hasproperty(model, :closure_fields) || return nothing
    cfs = model.closure_fields isa Tuple ? model.closure_fields : (model.closure_fields,)
    for cf in cfs
        isnothing(cf) && continue
        cf isa NamedTuple || continue
        haskey(cf, :νₑ) || continue
        return cf.νₑ
    end
    return nothing
end

"""
    native_mixing_diagnostics(model) -> Dict{Symbol, Any}

Model-native vertical mixing fields (`κ`, `ν`) promoted to `Field`s on the model's grid.

The hydrodynamics uses `ScalarDiffusivity` closures, so the vertical eddy diffusivity `κ`
and eddy viscosity `ν` that actually enter the tracer/velocity equations are *constants*.
Reporting those constants is the only way for downstream figures to reflect the mixing the
model really applied; re-deriving a spatially varying Richardson-number proxy in the
visualization layer produces values that contradict the integrated physics.

If the closure exposes a prognostic grid `Field` (e.g. `NEMOTKE`, or the Smagorinsky `νₑ`
in `closure_fields`), that `Field` is preferred, since it is then the genuinely spatially
varying quantity the model used.

# Inputs
- `model`: Oceananigans `AbstractModel` (e.g. `HydrostaticFreeSurfaceModel`).

# Outputs
- `Dict{Symbol, Any}`: `:κ` and/or `:ν` mapped to grid `Field`s, restricted to whatever the
  closure can supply. Empty when the closure exposes nothing usable.
"""
function native_mixing_diagnostics(model)
    out = Dict{Symbol, Any}()
    grid = model.grid

    κ_val, ν_val = closure_diffusivity_constants(model)
    isnothing(κ_val) || (out[:κ] = fill_field_const(grid, κ_val))
    isnothing(ν_val) || (out[:ν] = fill_field_const(grid, ν_val))

    # A Smagorinsky closure carries a prognostic eddy viscosity `νₑ` in `closure_fields`.
    # It is published under its own key rather than overwriting the constant background `ν`,
    # so downstream consumers can see both the prescribed and the turbulent contribution
    # instead of having one silently replace the other.
    νₑ = closure_field_smagarinsky_ν(model)
    if !isnothing(νₑ) && occursin("Field", string(typeof(νₑ).name.wrapper))
        out[:νₑ] = νₑ
    end

    return out
end

# Reference seawater density and gravitational acceleration used to convert the
# potential-density gradient into buoyancy frequency: N² = ∂z b.
# (Buoyancy is defined as b = g * (ρ₀ - ρ) / ρ₀, so N² = ∂z b = -(g/ρ₀) ∂ρ/∂z.)
const g_ref = 9.80665  # m s⁻²

"""
    model_equation_of_state(model) -> Union{Nothing, BoussinesqEquationOfState}

Recover the Boussinesq equation of state the model is actually integrating with.

`HydrostaticFreeSurfaceModel` stores it at `model.buoyancy.formulation.equation_of_state`, where
the formulation is a `SeawaterBuoyancy`. Every model field that needs a density — `:ρ` and the
`:N2` derived from it — is therefore evaluated with exactly the EOS the physics used, rather
than a hard-coded polynomial that could drift away from it.

Returns `nothing` when the model carries no such formulation (e.g. a `ConstantBuoyancy`
diagnostic model), so callers can skip density diagnostics instead of guessing.
"""
function model_equation_of_state(model)
    hasproperty(model, :buoyancy) || return nothing
    buoyancy = model.buoyancy
    hasproperty(buoyancy, :formulation) || return nothing
    formulation = buoyancy.formulation
    hasproperty(formulation, :equation_of_state) || return nothing
    return formulation.equation_of_state
end

"""
    model_gravitational_acceleration(model) -> Float64

Gravitational acceleration the model is integrating with, taken from its buoyancy formulation
so the published N² is consistent with the dynamics. Falls back to [`g_ref`](@ref) only when
the model does not advertise one.
"""
function model_gravitational_acceleration(model)
    candidates = if hasproperty(model, :buoyancy)
        buoyancy = model.buoyancy
        (hasproperty(buoyancy, :gravitational_acceleration) ? buoyancy.gravitational_acceleration : nothing,
         hasproperty(buoyancy, :formulation) && hasproperty(buoyancy.formulation, :gravitational_acceleration) ?
             buoyancy.formulation.gravitational_acceleration : nothing,
         hasproperty(buoyancy, :formulation) && hasproperty(buoyancy.formulation, :g) ?
             buoyancy.formulation.g : nothing)
    else
        (nothing, nothing, nothing)
    end

    for candidate in candidates
        candidate isa Real && return Float64(candidate)
    end

    return g_ref
end

"""
    reference_density(model) -> Float64

Reference density ρ₀ used to scale the potential-density gradient into buoyancy frequency.
Prefers the equation of state's own `reference_density`; falls back to `1025.0` kg m⁻³.
"""
function reference_density(model)
    eos = model_equation_of_state(model)
    if !isnothing(eos) && hasproperty(eos, :reference_density)
        ρ₀ = eos.reference_density
        ρ₀ isa Real && return Float64(ρ₀)
    end
    return 1025.0
end

"""
    fill_seawater_density!(ρ_field, model)

Fill `ρ_field` with the in-situ seawater density (kg m⁻³) implied by the model's own `T`/`S`
tracers and its equation of state, evaluated on `ρ_field`'s own grid.

The polynomial is `SeawaterPolynomials.ρ(Θ, Sᴬ, Z, eos)` — the real implementation behind
NumericalEarth's `total_density` alias, which is declared without methods in this build and
therefore cannot be called. Note the argument order: conservative temperature `Θ` first, absolute
salinity `Sᴬ` second. Swapping them yields a plausible-looking but physically wrong density
(≈ 1004 kg m⁻³ for shelf conditions that should read ≈ 1027 kg m⁻³), so the order is fixed here
rather than inferred at each call site. Depth `Z` is taken from the grid's own vertical coordinate
nodes, consistent with the convention used by [`write_grid_coordinates`](@ref).

Halos are filled alongside the interior because `:N2` differentiates vertically and so reads one
column beyond the interior at the top and bottom. Two details matter here:

* The loop runs over the *data* axes, not `axes(ρ_field, k)`. `axes` on a `Field` reports the
  interior (`1:Nz`), so iterating it leaves the halo columns holding whatever `new_data`
  initialised them to; the vertical difference at the first and last interior cells then reads
  uninitialised memory and reports N² off by ~1000×.
* The depth index is *clamped* into `1:Nz` when sampling the vertical coordinate. `znode` is
  undefined outside the grid, and extrapolating there produced density values orders of magnitude
  off. Clamping makes the halo a constant extension of the nearest interior cell, so the boundary
  gradient collapses to zero instead of being garbage.

The fill is always performed on the **host** and copied to the destination array, including on a
GPU run. Two reasons: the TEOS-10 polynomial carries a coefficient vector, so `eos` is not `isbits`
and cannot be captured inside a GPU kernel; and calling `ρ` elementwise over `T`/`S` scalar-indexes
device arrays, which CUDA disallows outright ("Scalar indexing is disallowed"). Since ρ is only
refreshed on the output schedule, the host round-trip is negligible next to a time step.
"""
function fill_seawater_density!(ρ_field, model)
    eos = model_equation_of_state(model)
    isnothing(eos) && return nothing

    T = model.tracers.T
    S = model.tracers.S

    grid = ρ_field.grid
    base_grid = grid isa ImmersedBoundaryGrid ? grid.underlying_grid : grid
    cpu_grid = architecture(base_grid) isa GPU ? on_architecture(CPU(), base_grid) : base_grid
    Nz = base_grid.Nz

    data = ρ_field.data
    size(data) == size(T.data) == size(S.data) || error(
        "Seawater density field $(size(data)) does not align with the model tracers " *
        "(T=$(size(T.data)), S=$(size(S.data))).")

    # Transfer unconditionally: `on_architecture(CPU(), …)` is a no-op for host arrays,
    # and testing the destination with `isa CuArray` is not reliable because a `Field`'s
    # `.data` can be an OffsetArray *wrapping* the device buffer rather than a CuArray.
    T_in = on_architecture(CPU(), T.data)
    S_in = on_architecture(CPU(), S.data)

    # Halo cells of a freshly assembled model hold *uninitialised* memory, which is not
    # merely garbage: reading it and feeding it to the TEOS-10 polynomial can overflow to
    # Inf/NaN and then throw a `DomainError` from a `sqrt` of a huge negative discriminant,
    # or corrupt the interpreter outright. Neither is recoverable, and the halo is not
    # something the diagnostics need — they are read from the interior and the halo is
    # masked by the immersed boundary. So the buffer starts life filled with the equation
    # of state's reference density and only the interior is ever evaluated.
    host = fill(Float64(eos.reference_density), size(data))

    # Interior index range per dimension. `indices(field, k)` reports a Colon, so the
    # halo width is derived from the data axes against the underlying grid's interior
    # size; the halo is symmetric, so half the excess is the halo on each side.
    ax1, ax2, ax3 = axes(data, 1), axes(data, 2), axes(data, 3)
    Nx, Ny = base_grid.Nx, base_grid.Ny
    ix = (first(ax1) + (length(ax1) - Nx) ÷ 2) : (first(ax1) + (length(ax1) - Nx) ÷ 2 + Nx - 1)
    iy = (first(ax2) + (length(ax2) - Ny) ÷ 2) : (first(ax2) + (length(ax2) - Ny) ÷ 2 + Ny - 1)
    iz = (first(ax3) + (length(ax3) - Nz) ÷ 2) : (first(ax3) + (length(ax3) - Nz) ÷ 2 + Nz - 1)

    # A halo column is still at a real depth, so the interior index is clamped into 1:Nz
    # when the vertical coordinate is sampled.
    k_phys(k) = clamp(k - first(iz) + 1, 1, Nz)

    @inbounds for k in iz, j in iy, i in ix
        z = Float64(Oceananigans.Grids.znode(k_phys(k), cpu_grid, Center()))
        Ti = Float64(T_in[i, j, k])
        Si = Float64(S_in[i, j, k])

        # Clamp into the polynomial's validity box; land cells legitimately hold T = S = 0,
        # and any residual uninitialised memory is caught by the isfinite guard.
        value = (isfinite(Ti) && isfinite(Si)) ?
                ρ(clamp(Ti, -2.0, 40.0), clamp(Si, 0.0, 42.0), z, eos) :
                Float64(eos.reference_density)

        host[i, j, k] = isfinite(value) ? value : Float64(eos.reference_density)
    end

    # Copy back in place so the caller's field object is updated, on CPU or device alike.
    copyto!(data, host)

    return ρ_field
end

"""
    native_stratification_diagnostics(model) -> Dict{Symbol, Any}

Model-native stratification and vorticity fields, ready to be written by a JLD2 writer.

* `:ρ` — in-situ seawater density (kg m⁻³) on the centre grid, from the model's own EOS.
* `:N2` — buoyancy frequency squared N² = −(g/ρ₀) ∂ρ/∂z (s⁻²), the scaled `∂z` operation
  on `:ρ`, using the model's own gravitational acceleration and reference density.
* `:ζ` — relative vertical vorticity ζ = ∂v/∂x − ∂u/∂y (s⁻¹), likewise on the centre grid.

These are `Field`/`AbstractOperation` objects that Oceananigans evaluates with the model's own
grid, stencils and stencilling, replacing the post-hoc finite-difference re-derivation
previously performed in the visualization layer.

`:ρ` is state derived from the prognostic tracers rather than a static field, so it must be
refreshed as the simulation advances. Call [`refresh_stratification_diagnostics!`](@ref) on each
tick of the output schedule; `setup_hydrodynamic_simulation` registers that as a callback, which
Oceananigans runs before the output writers on the same tick.

# Inputs
- `model`: Oceananigans `AbstractModel` carrying `T` and `S` tracers and `u`/`v` velocities.

# Outputs
- `Dict{Symbol, Any}`: keys `:ρ`, `:N2`, `:ζ` mapped to fields/operations. `:ρ` and `:N2` are
  omitted when the model carries no `T`/`S` tracers or no Boussinesq equation of state; `:ζ` is
  omitted when the model has no horizontal velocities.
"""
function native_stratification_diagnostics(model)
    out = Dict{Symbol, Any}()

    # Relative vertical vorticity ζ = ∂v/∂x − ∂u/∂y on the centre grid.
    # ∂x/∂y are the abstract operations exported by Oceananigans; the writer calls
    # `compute!` on them at each output time.
    if hasproperty(model, :velocities)
        u = model.velocities.u
        v = model.velocities.v
        out[:ζ] = ∂x(v) - ∂y(u)
    end

    # Seawater density and the stratification it implies. Both need the `T` and `S`
    # tracers plus the model's Boussinesq equation of state; without them there is no
    # physically meaningful density to publish, so the keys are omitted rather than
    # filled with a placeholder.
    #
    # NOTE: Native ρ/N² diagnostics are currently CPU-only because the TEOS-10 equation
    # of state is not isbits and cannot be called from GPU kernels. On GPU we skip these
    # diagnostics entirely (the GPU path has deeper stability issues in any case).
    if hasproperty(model, :tracers) && haskey(model.tracers, :T) && haskey(model.tracers, :S) &&
       !isnothing(model_equation_of_state(model)) &&
       !(architecture(model) isa GPU)
        cpu_model = on_architecture(CPU(), model)
        ρ_field_cpu = CenterField(cpu_model.grid)
        fill_seawater_density!(ρ_field_cpu, cpu_model)
        
        ρ_field = on_architecture(architecture(model), ρ_field_cpu)
        out[:ρ] = ρ_field

        # N² = ∂z b with buoyancy b = g (ρ₀ − ρ) / ρ₀, so N² = −(g/ρ₀) ∂ρ/∂z.
        # `∂z(ρ_field)` alone carries units of kg m⁻⁴, so the constant is applied here to
        # publish genuine buoyancy frequency in s⁻². Keeping it as a scalar-multiply
        # operation (rather than a pre-scaled Field) means the writer re-evaluates the whole
        # expression against the current ρ at every write.
        scale = -(model_gravitational_acceleration(model) / reference_density(model))
        out[:N2] = scale * ∂z(ρ_field)
    end

    return out
end

"""
    refresh_stratification_diagnostics!(out::Dict{Symbol, Any}, model)

Recompute the state-derived entries of [`native_stratification_diagnostics`](@ref) from the
model's current tracers, in place.

Only `:ρ` needs refreshing: `:N2` and `:ζ` are `∂z`/`∂x`/`∂y` operations over it or over the
velocities, and the output writer re-evaluates those with `compute!` at every write. A missing
`:ρ` (model without `T`/`S` or without an equation of state) is a no-op.
"""
function refresh_stratification_diagnostics!(out::Dict{Symbol, Any}, model)
    ρ_field = get(out, :ρ, nothing)
    isnothing(ρ_field) && return out

    # Native ρ/N² diagnostics are CPU-only (TEOS-10 not isbits). On GPU this callback
    # is a no-op since the keys were never added in native_stratification_diagnostics.
    architecture(model) isa GPU && return out

    cpu_model = on_architecture(CPU(), model)
    cpu_ρ = on_architecture(CPU(), ρ_field)
    fill_seawater_density!(cpu_ρ, cpu_model)
    
    copyto!(ρ_field.data, cpu_ρ.data)
    return out
end

"""
    write_grid_coordinates

Write the model's cell-centre geographic coordinates to a sidecar JLD2 file next to the
simulation output.

`write_grid_coordinates(model, output_path) -> String`

The main simulation file carries a `serialized/grid` entry, but deserialising it yields an
opaque `JLD2.ReconstructedStatic` rather than a usable `LatitudeLongitudeGrid`, because the
grid's type parameters cannot be resolved in the reading session. Anything that needs the
domain geometry — most importantly the Lagrangian flow interpolator — therefore reads these
plain vectors instead of attempting type reconstruction.

# Inputs
- `model`: Model whose grid coordinates are persisted.
- `output_path`: Path of the main simulation JLD2 file; the sidecar is derived from it.

# Outputs
- `String`: Path of the sidecar file written.

# Notes
Coordinates are geographic degrees for `lons`/`lats` and metres (negative down) for
`depths`, matching the conventions used throughout the package.
"""
function write_grid_coordinates(model, output_path::AbstractString)
    base, _ = splitext(output_path)
    sidecar = string(base, "_grid.jld2")

    base_g = model.grid isa ImmersedBoundaryGrid ? model.grid.underlying_grid : model.grid
    cpu_g = architecture(base_g) isa GPU ? on_architecture(CPU(), base_g) : base_g

    lons = collect(Float64, cpu_g.λᶜᵃᵃ)[cpu_g.Hx+1 : cpu_g.Hx+cpu_g.Nx]
    lats = collect(Float64, cpu_g.φᵃᶜᵃ)[cpu_g.Hy+1 : cpu_g.Hy+cpu_g.Ny]
    depths = [Float64(Oceananigans.Grids.znode(k, cpu_g, Center())) for k in 1:cpu_g.Nz]
    halo = (Int(cpu_g.Hx), Int(cpu_g.Hy), Int(cpu_g.Hz))
    grid_size = (Int(cpu_g.Nx), Int(cpu_g.Ny), Int(cpu_g.Nz))

    jldopen(sidecar, "w") do file
        file["lons"] = lons
        file["lats"] = lats
        file["depths"] = depths
        file["halo"] = halo
        file["size"] = grid_size
    end

    return sidecar
end



"""
    setup_hydrodynamic_simulation(
        model;
        Δt::Real = 2minutes,
        stop_time::Real = 12hours,
        adaptive_time_step::Bool = false,
        target_cfl::Real = 0.2,
        max_Δt::Real = 12.0,
        min_Δt::Real = 10.0,
        progress_schedule::Union{Real, Int} = 20,
        enable_output::Bool = true,
        output_dir::AbstractString = "outputs",
        output_filename::AbstractString = "nova_scotia_hydrodynamics.jld2",
        output_schedule::Union{Real, Int} = 100,
        overwrite_existing::Union{Nothing, Bool} = nothing,
        enable_checkpoint::Bool = true,
        checkpoint_dir::Union{Nothing, AbstractString} = nothing,
        checkpoint_prefix::AbstractString = "checkpoint_$(resolve_config_name())",
        checkpoint_schedule::Union{Nothing, Real, Int} = nothing,
        cleanup_checkpoints::Bool = true,
        pickup::Union{Bool, Symbol, String, Int} = :auto,
        watchdog::Bool = true,
        divergence_velocity_limit::Real = 20.0
    )

Configure an Oceananigans `Simulation` with progress callbacks, CFL monitoring,
numerical stability watchdogs, periodic prognostic state checkpointing, and JLD2
output writers for background hydrodynamic fields.

# Mathematical Formulation & Time Integration
Advances discrete time \$t^{n+1} = t^n + \\Delta t\$ using fractional step or
multistage schemes subject to the CFL condition:
```math
\\text{CFL} = \\max \\left( \\frac{|u| \\Delta t}{\\Delta x},
\\frac{|v| \\Delta t}{\\Delta y} \\right) \\le C_{\\text{target}}
```

# Inputs
- `model`: Configured `HydrostaticFreeSurfaceModel`.
- `Δt::Real`: Initial numerical time step in seconds (default 2 minutes).
- `stop_time::Real`: Simulation duration in seconds (default 12 hours).
- `adaptive_time_step::Bool`: Whether to dynamically scale `Δt` based on CFL.
- `target_cfl::Real`: Target CFL number when `adaptive_time_step` is enabled.
- `max_Δt::Real`: Maximum allowed time step in seconds (default `12.0` to ensure
  high-order WENO stability across immersed step bathymetry).
- `min_Δt::Real`: Minimum allowed time step in seconds (default `0.1`). If the wizard attempts to scale below this floor to preserve CFL, it implies spatial divergence.
- `progress_schedule::Union{Real, Int}`: Logging frequency (iterations or seconds).
- `enable_output::Bool`: Whether to attach a JLD2 timeseries output writer.
- `output_dir::AbstractString`: Directory where outputs will be saved.
- `output_filename::AbstractString`: Output file name.
- `output_schedule::Union{Real, Int}`: Output saving frequency (iterations or seconds).
- `overwrite_existing::Union{Nothing, Bool}`: Whether to overwrite pre-existing output.
  When `nothing` (default) and `pickup` is active, automatically preserves and appends.
- `enable_checkpoint::Bool`: Whether to attach Oceananigans `Checkpointer`.
- `checkpoint_dir::Union{Nothing, AbstractString}`: Directory for checkpoint archives.
  Defaults to `joinpath(output_dir, "checkpoints")`.
- `checkpoint_prefix::AbstractString`: Checkpoint filename prefix (default `"checkpoint_\$(resolve_config_name())"`).
- `checkpoint_schedule::Union{Nothing, Real, Int}`: Checkpoint frequency (defaults to
  `output_schedule` if unspecified).
- `cleanup_checkpoints::Bool`: Automatically prune older intermediate checkpoints.
- `pickup::Union{Bool, Symbol, String, Int}`: Checkpoint pickup specification.
  `:auto` (default) detects the latest existing checkpoint in `checkpoint_dir`.
- `watchdog::Bool`: Whether to attach a NaN/divergence early detection callback.
- `include_diagnostics::Bool`: Whether to include secondary diagnostic fields
  (mixing coefficients ν, κ, buoyancy frequency N², and vertical vorticity ζ) in
  the JLD2 output snapshots (default `true`). Set `false` to reduce file size.
- `divergence_velocity_limit::Real`: Maximum permissible fluid velocity threshold before
  raising divergence error (default 20.0 m/s).

# Outputs
- `Simulation`: Ready-to-run Oceananigans simulation instance.

# References
- Ramadhan, A., et al. (2020). Oceananigans.jl: Fast and friendly geophysical
  fluid dynamics on GPUs. *Journal of Open Source Software*, 5(53), 2018.
"""
function setup_hydrodynamic_simulation(
    model;
    Δt::Real = 2minutes,
    stop_time::Real = 12hours,
    adaptive_time_step::Bool = false,
    target_cfl::Real = 0.2,
    max_Δt::Real = 12.0,
    min_Δt::Real = 0.1,
    progress_schedule::Union{Real, Int} = 20,
    enable_output::Bool = true,
    output_path::AbstractString = joinpath("outputs", "nova_scotia_hydrodynamics.jld2"),
    output_schedule::Union{Real, Int} = 100,
    include_diagnostics::Bool = true,
    overwrite_existing::Union{Nothing, Bool} = nothing,
    enable_checkpoint::Bool = true,
    checkpoint_dir::Union{Nothing, AbstractString} = nothing,
    checkpoint_prefix::AbstractString = "checkpoint_$(resolve_config_name())",
    checkpoint_schedule::Union{Nothing, Real, Int} = nothing,
    cleanup_checkpoints::Bool = true,
    pickup::Union{Bool, Symbol, String, Int} = :auto,
    watchdog::Bool = true,
    divergence_velocity_limit::Real = 20.0
)
    mkpath(dirname(output_path))
    full_output_path = output_path

    # Resolve checkpoint directory
    cp_dir = if isnothing(checkpoint_dir) || isempty(checkpoint_dir)
        joinpath(dirname(output_path), "checkpoints")
    else
        String(checkpoint_dir)
    end
    if enable_checkpoint
        mkpath(cp_dir)
    end

    resolved_cp_prefix = if isempty(strip(checkpoint_prefix)) || checkpoint_prefix == "checkpoint"
        "checkpoint_$(resolve_config_name())"
    else
        String(checkpoint_prefix)
    end

    latest_cp = find_latest_checkpoint(cp_dir, prefix = resolved_cp_prefix)

    # Determine resolved pickup target
    resolved_pickup = if pickup === :auto
        if !isnothing(latest_cp)
            m_dim = size(model.velocities.u.data)
            if verify_checkpoint_compatibility(
                latest_cp,
                m_dim;
                divergence_velocity_limit = divergence_velocity_limit
            )
                @info "Detected existing hydrodynamic checkpoint for pickup: $(latest_cp)"
                latest_cp
            else
                false
            end
        else
            false
        end
    else
        pickup
    end

    # Determine whether to overwrite or append to existing diagnostic output file
    resolved_overwrite = if !isnothing(overwrite_existing)
        overwrite_existing
    elseif resolved_pickup != false && isfile(full_output_path)
        @info "Resuming simulation: appending to existing timeseries at $(full_output_path)."
        false
    else
        true
    end

    sim = Simulation(model, Δt = Δt, stop_time = stop_time)

    # 1. Progress reporting callback
    prog_sched = if progress_schedule isa Int
        IterationInterval(progress_schedule)
    else
        TimeInterval(progress_schedule)
    end

    progress_fn(s) = begin
        u_max = maximum(abs, interior(s.model.velocities.u))
        v_max = maximum(abs, interior(s.model.velocities.v))
        cfl_val = compute_advective_cfl(s.model, s.Δt)
        # Report the vertical velocity, the three directional Courant numbers and the
        # free-surface range alongside the advective CFL. `compute_advective_cfl` returns only
        # the MAXIMUM of the three, so on this grid it was impossible to tell from the log
        # which direction was actually limiting the step -- and a CFL of 6 against a 0.20
        # target is a scheme running far outside its stability limit, so knowing which
        # direction is responsible is the whole diagnosis.
        w_int = Array(interior(s.model.velocities.w))
        w_max, loc_w = findmax(abs, w_int)
        w_loc_txt = try
            lon_w, lat_w, dep_w = velocity_peak_location(s, loc_w)
            " (at $(format_geographic_location(lon_w, lat_w, dep_w)))"
        catch
            ""
        end
        # The grid-metric breakdown is best-effort: the metric field names differ between the
        # underlying grid of an `ImmersedBoundaryGrid` and a bare `LatitudeLongitudeGrid`, and
        # guessing them wrong would turn a diagnostic into the very failure it is meant to
        # report. `w_max` and the free-surface range are the numbers that matter here, and both
        # are always available, so a failure to read the metrics costs detail, not the run.
        metrics = try
            base_g = s.model.grid isa ImmersedBoundaryGrid ? s.model.grid.underlying_grid : s.model.grid
            dz_min = min_vertical_spacing(base_g)
            r_earth = Float64(base_g.radius)
            dlon_deg = minimum(diff(collect(base_g.λᶠᵃᵃ[1:base_g.Nx + 1])))
            dlat_deg = minimum(diff(collect(base_g.φᵃᶠᵃ[1:base_g.Ny + 1])))
            lat_max_abs = max(abs(base_g.φᵃᶠᵃ[1]), abs(base_g.φᵃᶠᵃ[base_g.Ny + 1]))
            cflz_val = 0.0
            if hasproperty(base_g, :z) && hasproperty(base_g.z, :Δᵃᵃᶜ)
                dz_vec_p = to_cpu_offset_array(base_g.z.Δᵃᵃᶜ)
                for k in 1:base_g.Nz
                    dz_k = Float64(dz_vec_p[k])
                    w_k = max(maximum(abs, @view(w_int[:, :, k])),
                              maximum(abs, @view(w_int[:, :, k + 1])))
                    cflz_val = max(cflz_val, (w_k * s.Δt) / max(1.0, dz_k))
                end
            else
                cflz_val = (w_max * s.Δt) / max(1.0, dz_min)
            end
            (dx_m = r_earth * cosd(lat_max_abs) * deg2rad(dlon_deg),
             dy_m = r_earth * deg2rad(dlat_deg),
             dz_m = dz_min,
             cflx = (u_max * s.Δt) / max(1.0, r_earth * cosd(lat_max_abs) * deg2rad(dlon_deg)),
             cfly = (v_max * s.Δt) / max(1.0, r_earth * deg2rad(dlat_deg)),
             cflz = cflz_val)
        catch
            nothing
        end
        # Free-surface excursion, in metres. A barotropic acceleration shows up here first:
        # if eta grows without bound the barotropic velocity follows, and the divergence
        # watchdog (which only inspects u and v) reports the consequence, not the cause.
        eta_txt = try
            if hasproperty(s.model, :free_surface)
                fs = s.model.free_surface
                η_field = hasproperty(fs, :displacement) ? fs.displacement :
                          hasproperty(fs, :η) ? fs.η : nothing
                if η_field !== nothing
                    " | η: $(round(minimum(interior(η_field)), digits=3))..$(round(maximum(interior(η_field)), digits=3)) m"
                else
                    ""
                end
            else
                ""
            end
        catch
            ""
        end
        metrics_txt = isnothing(metrics) ? "" :
            " | Δx=$(round(metrics.dx_m, digits=1)) Δy=$(round(metrics.dy_m, digits=1)) " *
            "Δz=$(round(metrics.dz_m, digits=2)) m (x=$(round(metrics.cflx, digits=3)), " *
            "y=$(round(metrics.cfly, digits=3)), z=$(round(metrics.cflz, digits=3)))"
        @info(
            "Iter: $(iteration(s)) | Time: $(prettytime(s)) | " *
            "Δt: $(round(s.Δt, digits=1))s | max(|u|): $(round(u_max, digits=4)) m/s | " *
            "max(|v|): $(round(v_max, digits=4)) m/s | " *
            "max(|w|): $(round(w_max, digits=4)) m/s$(w_loc_txt) | " *
            "CFL: $(round(cfl_val, digits=3))" * metrics_txt * eta_txt
        )
    end
    sim.callbacks[:progress] = Callback(progress_fn, prog_sched)

    # 2. Adaptive time step management
    #
    # This is deliberately NOT Oceananigans' `TimeStepWizard`. The wizard derives its Courant
    # number from the horizontal grid metrics alone, and on this model that is blind to the
    # direction that actually limits the step.
    #
    # Concretely, on the snowcrab configuration the wizard reported a horizontal CFL of about
    # 0.026 (u = 3.85 m/s, dt = 26.4 s, dx ~ 3.9 km) against a target of 0.2, and therefore
    # INCREASED dt from 18.0 s to 26.4 s -- while the true stability was being violated in the
    # vertical, where max(|w|) = 2.4 m/s across a 10 m surface cell gives a Courant number near
    # 6. Growing dt in the face of that is what let max(|u|) climb to the 20 m/s watchdog, and
    # max(|w|) reach 7 m/s, which no ocean flow produces.
    #
    # `compute_advective_cfl` already computes the correct three-directional maximum, including
    # the smallest vertical cell, so the controller below drives dt from that instead. It runs
    # every iteration rather than every 5, and is allowed to halve dt in a single step: the
    # growth being controlled is exponential, so a controller limited to a 0.5x change every
    # fifth iteration cannot catch up once it starts falling behind.
    if adaptive_time_step
        adapt_fn = function (s)
            cfl = compute_advective_cfl(s.model, s.Δt)
            if isfinite(cfl) && cfl > 0
                # Scale so the next step lands on the target, then bound the change to a
                # factor of two in either direction: unbounded scaling lets a single noisy
                # field maximum collapse dt to the floor, and a factor-of-two cap per step
                # still removes a CFL of 6 in about three iterations.
                s.Δt = clamp(s.Δt * (target_cfl / cfl), 0.5 * s.Δt, 2.0 * s.Δt)
                s.Δt = clamp(s.Δt, Float64(min_Δt), Float64(max_Δt))
            end
            return nothing
        end
        sim.callbacks[:wizard] = Callback(adapt_fn, IterationInterval(5))
    end

    # 3. Numerical stability watchdog callback
    if watchdog
        stability_check(s) = begin
            # `u` and `v` sit on staggered face axes, so their interiors are *not* the same
            # shape: `u` is (nx+1, ny, nz) and `v` is (nx, ny+1, nz). Broadcasting the two
            # against each other therefore always fails with a DimensionMismatch, so both are
            # first restricted to the footprint they share. Only the high end of each axis is
            # dropped, so the remaining indices still address the same metric entries below.
            u_int = interior(s.model.velocities.u)
            v_int = interior(s.model.velocities.v)
            nx_c = min(size(u_int, 1), size(v_int, 1))
            ny_c = min(size(u_int, 2), size(v_int, 2))
            spd = max.(
                abs.(u_int[1:nx_c, 1:ny_c, :]),
                abs.(v_int[1:nx_c, 1:ny_c, :])
            )
            # Where is the maximum? A bare magnitude cannot distinguish a localised problem
            # (the Bay of Fundy narrows, a steep bank, a sponge edge) from a domain-wide one,
            # and those call for completely different fixes.
            spd_max, loc = findmax(spd)
            if isnan(spd_max) || isinf(spd_max) || spd_max > divergence_velocity_limit
                lon_loc, lat_loc, dep_loc = velocity_peak_location(s, loc)
                loc_str = format_geographic_location(lon_loc, lat_loc, dep_loc)
                error(
                    "Numerical divergence detected at iteration $(iteration(s)), " *
                    "time $(prettytime(s)). Velocity magnitude u_max = $(spd_max) m/s " *
                    "(limit: $(divergence_velocity_limit) m/s) at $(loc_str)."
                )
            elseif spd_max > 5.0 && (iteration(s) % 50 == 0)
                lon_loc, lat_loc, dep_loc = velocity_peak_location(s, loc)
                loc_str = format_geographic_location(lon_loc, lat_loc, dep_loc)
                @warn(
                    "Elevated interior velocity magnitude: max(|u|,|v|) = " *
                    "$(round(spd_max, digits=2)) m/s at iteration $(iteration(s)) ($(prettytime(s))), " *
                    "located at $(loc_str)."
                )
            end
        end
        sim.callbacks[:watchdog] = Callback(stability_check, IterationInterval(50))
    end

    # 4. Field output writing
    if enable_output
        outputs_dict = Dict{Symbol, Any}(
            :u => model.velocities.u,
            :v => model.velocities.v,
            :w => model.velocities.w
        )
        for tracer_name in keys(model.tracers)
            outputs_dict[tracer_name] = model.tracers[tracer_name]
        end
        if hasproperty(model, :free_surface) && hasproperty(model.free_surface, :η)
            outputs_dict[:η] = model.free_surface.η
        end

        if include_diagnostics
            # Model-native vertical mixing, written on the tracer (centre) grid so the
            # consumer never has to de-stagger or guess at halos for these fields.
            merge!(outputs_dict, native_mixing_diagnostics(model))

            # Model-native stratification (ρ, N²) and vorticity (ζ), evaluated by
            # Oceananigans on the model's own grid rather than re-derived downstream.
            strat_dict = native_stratification_diagnostics(model)
            merge!(outputs_dict, strat_dict)
        else
            strat_dict = Dict{Symbol, Any}()
        end

        out_sched = if output_schedule isa Int
            IterationInterval(output_schedule)
        else
            TimeInterval(output_schedule)
        end

        # ρ is derived from the prognostic `T`/`S` tracers, so it has to be recomputed as
        # the simulation advances; `∂z(ρ)` is an operation and is recomputed by the writer
        # itself. Oceananigans runs `TimeStepCallsite` callbacks before the output writers on
        # each tick (see `Simulations/run.jl`), so sharing the writer's schedule guarantees ρ
        # is current at the moment it is written rather than one interval behind.
        if include_diagnostics && haskey(strat_dict, :ρ)
            sim.callbacks[:stratification] = Callback(
                s -> refresh_stratification_diagnostics!(strat_dict, s.model),
                out_sched
            )
        end

        writer = JLD2Writer(
            model,
            outputs_dict,
            filename = full_output_path,
            schedule = out_sched,
            overwrite_files = resolved_overwrite
        )
        sim.output_writers[:fields] = writer

        # Persist the grid coordinate vectors as plain numeric arrays.
        #
        # The writer also stores a `serialized/grid` entry, but deserialising it yields an
        # opaque `JLD2.ReconstructedStatic` rather than a usable grid (the grid's type
        # parameters cannot be resolved in the reading session), so consumers such as
        # `create_flow_interpolator_from_jld2` cannot read coordinates off it. Writing the
        # vectors explicitly is version-independent and removes the dependency on JLD2 type
        # reconstruction entirely.
        write_grid_coordinates(model, full_output_path)
    end

    # 5. Prognostic state checkpointing
    if enable_checkpoint
        cp_sched_val = (isnothing(checkpoint_schedule) || (checkpoint_schedule isa Real && checkpoint_schedule <= 0)) ?
            output_schedule : checkpoint_schedule
        cp_sched = if cp_sched_val isa Int
            IterationInterval(cp_sched_val)
        else
            TimeInterval(cp_sched_val)
        end

        checkpointer = Checkpointer(
            model,
            schedule = cp_sched,
            dir = cp_dir,
            prefix = resolved_cp_prefix,
            overwrite_files = true,
            cleanup = cleanup_checkpoints
        )
        sim.output_writers[:checkpointer] = checkpointer
    end

    return sim
end

"""
    run_hydrodynamic_simulation!(
        simulation;
        pickup::Union{Bool, Symbol, String, Int} = :auto,
        checkpoint_at_end::Bool = true,
        verbose::Bool = true
    ) -> Simulation

Execute time integration of the ocean hydrodynamic simulation with error diagnostics,
automated checkpoint pickup, and graceful emergency checkpointing on interruption.

# Mathematical Formulation & Time Integration
Advances prognostic primitive equation variables:
```math
\\boldsymbol{q}^{n+1} = \\boldsymbol{q}^n + \\int_{t^n}^{t^{n+1}} \\mathcal{F}(\\boldsymbol{q}, t) \\, dt
```
When `pickup` is specified or auto-detected, restores prognostic state `q^k`
from discrete checkpoint iteration `k` without recomputing preceding time steps.

# Inputs
- `simulation::Simulation`: Oceananigans simulation to execute.
- `pickup::Union{Bool, Symbol, String, Int}`: Checkpoint pickup specification.
  - `:auto` (default): Picks up from latest checkpoint in `:checkpointer` if present.
  - `true`: Resumes from most recently modified checkpoint in `:checkpointer`.
  - `filepath::String`: Resumes from specified checkpoint archive path.
  - `false`: Integrates from model's current initialized state without checkpoint pickup.
- `checkpoint_at_end::Bool`: Automatically serialize final state upon stopping (default `true`).
- `verbose::Bool`: Whether to log progress, elapsed wall-clock time, and completion.

# Outputs
- `Simulation`: Completed or gracefully paused simulation instance.

# References
- Ramadhan, A., et al. (2020). *JOSS*, 5(53), 2018.
"""
function run_hydrodynamic_simulation!(
    simulation;
    pickup::Union{Bool, Symbol, String, Int} = :auto,
    checkpoint_at_end::Bool = true,
    verbose::Bool = true
)
    # Resolve pickup target
    actual_pickup = if pickup === :auto
        if haskey(simulation.output_writers, :checkpointer)
            cp_writer = simulation.output_writers[:checkpointer]
            candidates = list_candidate_checkpoints(cp_writer.dir, prefix = cp_writer.prefix)
            m_dim = size(simulation.model.velocities.u.data)
            valid_cp = nothing
            for cp in candidates
                if verify_checkpoint_compatibility(cp, m_dim)
                    valid_cp = cp
                    break
                else
                    @warn "Skipping invalid or incompatible candidate checkpoint: $(cp)"
                end
            end
            if !isnothing(valid_cp)
                @info "Picking up hydrodynamic simulation from validated checkpoint: $(valid_cp)"
                valid_cp
            else
                false
            end
        else
            false
        end
    else
        pickup
    end

    if verbose
        @info "Starting hydrodynamic simulation..."
        @info "Initial Δt: $(simulation.Δt) s, Stop time: $(prettytime(simulation.stop_time))"
        if actual_pickup != false
            @info "Pickup enabled: $(actual_pickup)"
        end
    end

    start_wall_time = time()
    try
        run!(simulation; pickup = actual_pickup, checkpoint_at_end = checkpoint_at_end)
    catch err
        if err isa InterruptException
            @warn(
                "Hydrodynamic simulation interrupted at iteration $(iteration(simulation)), " *
                "time: $(prettytime(simulation))."
            )
            # Perform emergency state checkpoint if numerically stable
            if haskey(simulation.output_writers, :checkpointer)
                try
                    stable, max_spd = is_model_velocity_stable(simulation.model)
                    if stable
                        checkpoint(simulation)
                        cp_w = simulation.output_writers[:checkpointer]
                        @info "Emergency restart checkpoint safely saved to directory: $(cp_w.dir)"
                    else
                        @warn "Skipping emergency checkpoint: model velocity diverges ($(round(max_spd, digits=2)) m/s)."
                    end
                catch cp_err
                    @error "Failed to save emergency checkpoint during interruption: $(cp_err)"
                end
            end
            @info "Simulation state safely preserved on disk. Resume anytime with --restart."
            return simulation

        else
            @error(
                "Simulation failed at iteration $(iteration(simulation)), " *
                "time: $(prettytime(simulation))",
                exception = (err, catch_backtrace())
            )
            rethrow(err)
        end
    end

    elapsed_time = time() - start_wall_time
    if verbose
        @info "Simulation completed successfully in $(round(elapsed_time, digits=2)) seconds."
    end
    return simulation
end

"""
    _interior_centers(raw, N, H; name = "field") -> Array{Float64}

Map an array as written by `JLD2Writer` onto the `N = (N_1, ..., N_D)` interior cell centres.

The stored length ``s_d`` along each dimension identifies the storage, given the interior
count ``N_d`` and the halo width ``H_d`` from the grid sidecar:

| ``s_d``             | storage                | operation                                     |
|:--------------------|:-----------------------|:----------------------------------------------|
| ``N_d``             | interior centres       | identity                                      |
| ``N_d + 1``         | interior faces         | ``\\phi^c_i = (\\phi^f_i + \\phi^f_{i+1}) / 2``     |
| ``N_d + 2H_d``      | centres with halo      | indices ``H_d + 1 : H_d + N_d``               |
| ``N_d + 2H_d + 1``  | faces with halo        | faces ``H_d + 1 : H_d + N_d + 1``, then average |

Assumes faces in a `Bounded` direction (``N_d + 1`` interior faces). A face-located field in a
`Periodic` direction has ``N_d`` faces and is indistinguishable from a centred field by shape;
it would be returned without the half-cell average.

# Inputs
- `raw::AbstractArray`: stored array, `ndims(raw) == D`.
- `N::NTuple{D, Int}`: interior cell counts.
- `H::NTuple{D, Int}`: halo widths.
- `name`: label used in the error message.

# Outputs
- `Array{Float64, D}` of size `N`. Throws if any ``s_d`` matches none of the rows above, rather
  than truncating to an arbitrary sub-block.
"""
function _interior_centers(raw::AbstractArray, N::NTuple{D, Int}, H::NTuple{D, Int};
                           name::AbstractString = "field") where {D}
    ndims(raw) == D || error(
        "$(name): stored array has $(ndims(raw)) dimension(s), expected $(D).")
    out = Float64.(raw)
    for d in 1:D
        s, n, h = size(out, d), N[d], H[d]
        out = if s == n
            out
        elseif s == n + 1
            0.5 .* (selectdim(out, d, 1:n) .+ selectdim(out, d, 2:(n + 1)))
        elseif h > 0 && s == n + 2h
            collect(selectdim(out, d, (h + 1):(h + n)))
        elseif h > 0 && s == n + 2h + 1
            0.5 .* (selectdim(out, d, (h + 1):(h + n)) .+
                    selectdim(out, d, (h + 2):(h + n + 1)))
        else
            error("$(name): stored size $(size(raw)) does not match interior $(N) with " *
                  "halo $(H) along dimension $(d) (length $(s); expected $(n), $(n + 1), " *
                  "$(n + 2h) or $(n + 2h + 1)). Check the grid sidecar against the archive.")
        end
    end
    return out
end

"""
    create_flow_interpolator_from_jld2(
        jld2_filepath::AbstractString;
        variables::Tuple = (:u, :v, :w, :T)
    )

Construct a high-performance 4D spatiotemporal interpolator function `(lon, lat, z, t)`
from saved Oceananigans JLD2 simulation outputs.

# Mathematical Formulation
Given discrete grid nodes \$(x_i, y_j, z_k)\$ and time levels \$t_m\$, queries at
arbitrary Lagrangian particle positions \$(\\lambda, \\phi, z, t)\$ are evaluated via
trilinear spatial interpolation followed by linear temporal interpolation:
```math
f(\\boldsymbol{x}, t) = (1 - \\theta) f(\\boldsymbol{x}, t_m) + \\theta f(\\boldsymbol{x}, t_{m+1})
```
where \$\\theta = (t - t_m) / (t_{m+1} - t_m)\$.

# Inputs
- `jld2_filepath::AbstractString`: Path to the simulation `.jld2` output file.
- `variables::Tuple`: Tuple of variable symbols to load (`:u, :v, :w, :T`).
- `t_window::Union{Nothing, Tuple{<:Real, <:Real}}`: Optional `(t_start, t_end)` in **seconds**
  restricting which snapshots are loaded. Queries outside the window are clamped to its edges.
  Particle tracking usually covers a small fraction of a long simulation (e.g. a 60-day larval
  window inside a 2-year run), and loading only that window is what keeps the interpolator within
  memory. `nothing` (the default) loads every snapshot.
- `max_snapshots::Union{Nothing, Integer}`: Optional hard cap on the number of snapshots retained,
  applied by evenly subsampling the selected window if the window still exceeds this many. A final
  backstop against running out of memory on a very long or very high-cadence record.
- `time_range_out::Union{Nothing, Vector{Float64}}`: Optional two-element output vector, resized to
  `[t_start, t_end]` (seconds) to report the span actually retained. Use this to size a run from the
  hydrodynamics rather than assuming a duration: queries outside the retained span are clamped to
  the edge snapshot, so tracking past `t_end` would integrate a frozen field.

# Outputs
- `Function`: Callable `(lon, lat, z, t) -> NamedTuple` returning interpolated field values.

# Memory
A production `snowcrab` simulation stores thousands of snapshots on a 345x245x20 grid. Materialising
all of them as `Float64` for `u, v, w, T` needs hundreds of gigabytes, so an unrestricted load
fails with `OutOfMemoryError` and the caller silently substitutes an analytical flow — which is a
scientifically wrong result, not a graceful degradation. Pass `t_window` (and/or `max_snapshots`) so
the resident set matches the part of the record actually being interrogated.

# Storage layout
Fields are written with halos and on the staggered C-grid. Each stored array is reduced to
interior cell centres by [`_interior_centers`](@ref), using the interior size and halo widths
from the `<base>_grid.jld2` sidecar.

# Multi-part runs
Parts are merged on the physical time axis `timeseries/t`. Where two parts carry the same time
(a seam, or a run restarted from t = 0), the snapshot from the most recently written part is
kept.

# References
- Marshall, J., et al. (1997). *J. Geophys. Res. Oceans*, 102(C3), 5753-5766.
"""
function create_flow_interpolator_from_jld2 end

"""
    resolve_hydro_model_paths(filepath::AbstractString) -> Vector{String}

Return every archive belonging to one simulation, in chronological part order.

A run that is extended writes a NEW file rather than appending, because `JLD2Writer`
re-serialises the model properties (`serialized/coriolis`, `buoyancy`, `closure`, and the
grid entries) unconditionally whenever it opens a file. Reopening an archive it previously
wrote therefore raises `a group or dataset named rotation_rate is already present`. There is
no append mode in this writer: `overwrite_existing` only decides whether the file is removed
first. So each part is written once and never reopened, and the parts must be stitched
together on read.

Naming is `<base>_partN.jld2`, ordered **numerically**. Sorting the suffixes as strings
would place `_part10` before `_part2`, and the resulting time axis would run backwards at
that point -- which looks like a corrupt simulation rather than a lexicographic slip.

The un-suffixed file is part 1. A run writes the plain `<base>.jld2` on its first invocation
and `_partN` files on each subsequent extension, so the resolver returns the base **and** the
parts. Treating the base as separate and returning only the parts would silently discard the
first increment of every extended run -- the commonest case, and the one a short run would
never reveal.

A run that was never extended has no `_part*` files, so the base is returned on its own.
That keeps every existing archive and every existing call site working unchanged.
"""
function resolve_hydro_model_paths(filepath::AbstractString)
    base, ext = splitext(filepath)
    isfile(filepath) || error("Simulation output file not found at: $(filepath)")
    dir = dirname(filepath)
    isdir(dir) || return [filepath]

    stem = basename(base)
    # Escape the stem before interpolating it into a pattern: a stem containing regex
    # metacharacters would otherwise match (or fail to match) the wrong files.
    esc = replace(stem, r"([\\^\$.|?*+()\[\]{}])" => s"\\\1")
    # `\$` is escaped because a bare `$` immediately before the closing quote is parsed as an
    # interpolation rather than as a regex anchor.
    pat = Regex("^" * esc * "_part([0-9]+)\$")

    # Parts live in a `parts/` subdirectory, not alongside the base archive. A long run can
    # accumulate a dozen multi-GB files, and interleaving them with the inputs, checkpoints
    # and figures makes the output directory hard to read and easy to copy by mistake.
    pdir = joinpath(dir, "parts")
    isdir(pdir) || return [filepath]

    parts = String[]
    for f in readdir(pdir; join = true)
        name, fext = splitext(f)
        fext == ext || continue
        # Match on the BASE NAME. The pattern is anchored with `^` to the stem, so testing it
        # against the full path -- which starts with the directory -- can never match, and the
        # resolver silently returned the single un-extended path instead of every part.
        m = match(pat, basename(name))
        m === nothing && continue
        push!(parts, f)
    end
    isempty(parts) && return [filepath]

    num = Dict(f => parse(Int, match(pat, basename(splitext(f)[1])).captures[1]) for f in parts)
    sort!(parts, by = f -> num[f])
    # The base file is the first increment; the parts follow it.
    all_parts = vcat([filepath], parts)
    println("Reassembling $(length(all_parts)) simulation part(s): " *
            join(basename.(all_parts), ", "))
    return all_parts
end

function create_flow_interpolator_from_jld2(
    jld2_filepath::AbstractString;
    variables::Tuple = (:u, :v, :w, :T),
    domain_lon::Union{Nothing, Tuple{<:Real, <:Real}} = nothing,
    domain_lat::Union{Nothing, Tuple{<:Real, <:Real}} = nothing,
    t_window::Union{Nothing, Tuple{<:Real, <:Real}} = nothing,
    max_snapshots::Union{Nothing, Integer} = nothing,
    time_range_out::Union{Nothing, Vector{Float64}} = nothing
)
if !isfile(jld2_filepath)
        error("Simulation output file not found at: $(jld2_filepath)")
    end

# A run extended past its first length writes a NEW `_partN` archive rather than
    # reopening the previous one: `JLD2Writer` re-serialises `serialized/coriolis`,
    # `buoyancy`, `closure` and the grid entries every time it opens a file, so reopening an
    # archive it wrote before raises `a group or dataset named rotation_rate is already
    # present`. `overwrite_existing` only decides whether the file is deleted first, so there
    # is no append mode. The parts are stitched here, in one place, rather than in each
    # reader that opens this file -- otherwise the compatibility check and the time-span
    # query would each describe part 1 alone.
    parts = resolve_hydro_model_paths(jld2_filepath)
    multi = length(parts) > 1

    # ---- Pass 1: the time axes of every part, and no field data.
    #
    # The combined axis must exist before any array is sized, for two reasons pulling the
    # same way: the window and the snapshot cap must be applied to the WHOLE run -- subsample
    # each part first and the concatenation has an unevenly spaced axis, which the
    # interpolator accepts silently -- and selection must stay ahead of allocation, which is
    # what bounds peak memory on a long run. Only `timeseries/t` and the key lists are read,
    # so this costs one pass of metadata rather than a pass over the fields.
    local part_keys = Vector{Vector{String}}(undef, length(parts))
    local part_times = Vector{Vector{Float64}}(undef, length(parts))
    for (pi, p) in enumerate(parts)
        jldopen(p, "r") do file
            haskey(file, "timeseries/u") || error(
                "JLD2 file $(p) is missing required 'timeseries/u' group.")
            g = file["timeseries/u"]
            ks = filter(k -> !isnothing(tryparse(Float64, k)), collect(keys(g)))
            isempty(ks) && error(
                "JLD2 timeseries 'timeseries/u' in $(p) contains no numeric time keys.")
            sort!(ks, by = k -> parse(Float64, k))
            # The snapshot KEYS are not times -- they are elapsed-second stamps written by the
            # checkpointer. The physical time of each snapshot lives in `timeseries/t`, and
            # selecting on the keys would select on checkpointer indices, so a `t_window` in
            # seconds would pick the wrong snapshots or none.
            t = if haskey(file, "timeseries/t")
                o = file["timeseries/t"]
                o isa JLD2.Group ? [Float64(o[k]) for k in ks] : collect(Float64, o)
            else
                [parse(Float64, k) for k in ks]
            end
            length(t) == length(ks) || (t = t[1:min(length(t), length(ks))])
            part_keys[pi] = ks
            part_times[pi] = t
        end
    end

    # Flatten to (part, index) pairs, ordered by TIME. Parts are written in order, but
    # sorting on the time value is what actually guarantees a monotonic axis.
    want_all = [(pi, k) for pi in eachindex(parts) for k in eachindex(part_keys[pi])]
    all_t = [part_times[pi][k] for (pi, k) in want_all]
    # Ties in time are ordered latest part first, so the de-duplication below keeps the
    # most recently written snapshot. At a true seam the two are the same state; after a
    # restart from t = 0 the later part is the one that ran to completion.
    perm = sortperm([(all_t[i], -want_all[i][1]) for i in eachindex(all_t)])
    want_all = want_all[perm]
    all_t = all_t[perm]

    # Drop repeated times. A part ends at the instant its successor resumes from, so a seam
    # can carry the same time twice; two snapshots at one instant give the temporal
    # interpolator a zero-length interval, which is a division by zero rather than a warning.
    uniq = trues(length(all_t))
    for i in 2:length(all_t)
        all_t[i] == all_t[i - 1] && (uniq[i] = false)
    end
    ndup = length(all_t) - sum(uniq)
    ndup > 0 && println("  dropped $(ndup) duplicate time(s) across parts " *
                        "(kept the most recently written part at each).")
    want_all = want_all[uniq]
    all_t = all_t[uniq]
    isempty(all_t) && error(
        "No time snapshots found across $(length(parts)) simulation part(s).")

    nt_total = length(all_t)
    t_lo, t_hi = all_t[1], all_t[end]

    keep = trues(nt_total)
    if !isnothing(t_window)
        lo, hi = Float64(t_window[1]), Float64(t_window[2])
        keep .= (all_t .>= lo) .& (all_t .<= hi)
        if !any(keep)
            keep[argmin(abs.(all_t .- clamp(lo, t_lo, t_hi)))] = true
        end
    end

    selected = findall(keep)
    if !isnothing(max_snapshots) && max_snapshots > 0 && length(selected) > max_snapshots
        s2 = unique!(round.(Int, range(1, length(selected), length = max_snapshots)))
        selected = selected[s2]
    end
    isempty(selected) && error(
        "No snapshots retained from $(length(parts)) simulation part(s): the selection is " *
        "empty (t_window=$(t_window), max_snapshots=$(max_snapshots)).")

    want = want_all[selected]
    t_vec = all_t[selected]
    nt = length(want)
    if nt < nt_total || multi
        @info "create_flow_interpolator_from_jld2: retained $nt of $nt_total snapshots " *
              "from $(length(parts)) part(s) (t_window=$(t_window), " *
              "max_snapshots=$(max_snapshots)); covering " *
              "$(round(t_vec[1], digits=1)) .. $(round(t_vec[end], digits=1)) s " *
              "($(round((t_vec[end] - t_vec[1]) / 86400, digits=2)) days)"
    end
    if !isnothing(time_range_out)
        resize!(time_range_out, 2)
        time_range_out[1] = t_vec[1]
        time_range_out[2] = t_vec[end]
    end

    # Grid geometry comes from the sidecar written alongside the FIRST part, which holds plain
    # numeric vectors. `serialized/grid` is deliberately not read: deserialising it yields an
    # opaque `JLD2.ReconstructedStatic`, because the grid's type parameters cannot be
    # resolved in the reading session.
    first_part = parts[first(want)[1]]
    grid_path = string(first(splitext(first_part)), "_grid.jld2")
    coords = if isfile(grid_path)
        JLD2.jldopen(grid_path, "r") do cf
            (lons = collect(Float64, cf["lons"]), lats = collect(Float64, cf["lats"]),
             depths = collect(Float64, cf["depths"]),
             halo = Int.(collect(cf["halo"])), size = Int.(collect(cf["size"])))
        end
    else
        nothing
    end

    Nx = isnothing(coords) ? 0 : coords.size[1]
    Ny = isnothing(coords) ? 0 : coords.size[2]
    Nz = isnothing(coords) ? 0 : coords.size[3]
    Hx = isnothing(coords) ? 0 : coords.halo[1]
    Hy = isnothing(coords) ? 0 : coords.halo[2]
    Hz = isnothing(coords) ? 0 : coords.halo[3]
    if Nx == 0
        # No sidecar: halos are unknown and taken as zero. The interior size is read from a
        # cell-centred field when one exists, because `u` carries an extra x-face.
        jldopen(first_part, "r") do file
            key0 = part_keys[first(want)[1]][first(want)[2]]
            ref = haskey(file, "timeseries/T") ? "timeseries/T/$(key0)" :
                                                 "timeseries/u/$(key0)"
            s2 = size(file[ref])
            Nx, Ny, Nz = s2[1], s2[2], s2[3]
        end
    end

    # Every part must describe the SAME grid. Asserted rather than trusted: stitching two
    # different grids would either throw obscurely or, worse, interleave them into an array
    # that is internally consistent and physically meaningless. The comparison is between
    # STORED layouts (halo and staggering included), since that is what pass 2 reads;
    # comparing a stored shape with the interior size rejects every haloed archive.
    stored_layout(file, pi) = Tuple(
        haskey(file, "timeseries/$(v)") ? size(file["timeseries/$(v)/$(part_keys[pi][1])"]) :
                                          ()
        for v in ("u", "v", "w", "T"))
    ref_layout = jldopen(f -> stored_layout(f, 1), parts[1], "r")
    for (pi, p) in enumerate(parts)
        lay = pi == 1 ? ref_layout : jldopen(f -> stored_layout(f, pi), p, "r")
        lay == ref_layout || error(
            "Simulation part $(basename(p)) stores (u, v, w, T) with sizes $(lay) but " *
            "$(basename(parts[1])) stores $(ref_layout). Parts of one simulation must share " *
            "a grid; these cannot be stitched.")
    end
    NN = (Nx, Ny, Nz)
    HH = (Hx, Hy, Hz)

    lon_raw = if isnothing(coords)
        isnothing(domain_lon) && error(
            "No sidecar grid coordinates ($(grid_path)) and no `domain_lon` supplied. Pass " *
            "the configured study-domain range from the TOML `[domain]` section.")
        collect(range(Float64(domain_lon[1]), Float64(domain_lon[2]), length = Nx))
    else
        coords.lons
    end
    lat_raw = if isnothing(coords)
        isnothing(domain_lat) && error(
            "No sidecar grid coordinates ($(grid_path)) and no `domain_lat` supplied. Pass " *
            "the configured study-domain range from the TOML `[domain]` section.")
        collect(range(Float64(domain_lat[1]), Float64(domain_lat[2]), length = Ny))
    else
        coords.lats
    end
    dep_raw = isnothing(coords) ? collect(range(-1000.0, 0.0, length = Nz)) : coords.depths

    i_c = (1 + Hx):(Nx + Hx)
    j_c = (1 + Hy):(Ny + Hy)
    k_c = (1 + Hz):(Nz + Hz)
    lons_vec = length(lon_raw) >= (Nx + 2 * Hx) ? lon_raw[i_c] :
               (length(lon_raw) >= Nx ? lon_raw[1:Nx] :
                collect(range(first(lon_raw), last(lon_raw), length = Nx)))
    lats_vec = length(lat_raw) >= (Ny + 2 * Hy) ? lat_raw[j_c] :
               (length(lat_raw) >= Ny ? lat_raw[1:Ny] :
                collect(range(first(lat_raw), last(lat_raw), length = Ny)))
    deps_vec = length(dep_raw) >= (Nz + 2 * Hz) ? dep_raw[k_c] :
               (length(dep_raw) >= Nz ? dep_raw[1:Nz] :
                collect(range(first(dep_raw), last(dep_raw), length = Nz)))

    # ---- Pass 2: field data, one part at a time.
    #
    # Each part is opened once and only the selected snapshots are read from it, so a long run
    # never holds two parts' data resident at the same time.
    u_arr = Array{Float64}(undef, Nx, Ny, Nz, nt)
    v_arr = Array{Float64}(undef, Nx, Ny, Nz, nt)
    w_arr = Array{Float64}(undef, Nx, Ny, Nz, nt)
    T_arr = Array{Float64}(undef, Nx, Ny, Nz, nt)
    η_arr = Array{Float64}(undef, Nx, Ny, nt)
    have_eta = false

    for (m, (pi, k)) in enumerate(want)
        jldopen(parts[pi], "r") do file
            key = part_keys[pi][k]
            read3(v) = _interior_centers(file["timeseries/$(v)/$(key)"], NN, HH;
                                         name = "$(basename(parts[pi])):$(v)@$(key)")
            u_arr[:, :, :, m] = read3("u")
            v_arr[:, :, :, m] = read3("v")
            w_arr[:, :, :, m] = haskey(file, "timeseries/w") ? read3("w") : zeros(NN)
            T_arr[:, :, :, m] = haskey(file, "timeseries/T") ? read3("T") : zeros(NN)
            if haskey(file, "timeseries/η")
                have_eta = true
                raw_η = file["timeseries/η/$(key)"]
                # The free surface is a single z-level; drop that singleton dimension.
                if ndims(raw_η) == 3
                    size(raw_η, 3) == 1 || error(
                        "η@$(key): expected one vertical level, found size $(size(raw_η)).")
                    raw_η = raw_η[:, :, 1]
                end
                η_arr[:, :, m] = _interior_centers(raw_η, (Nx, Ny), (Hx, Hy);
                                                   name = "η@$(key)")
            end
        end
    end
    have_eta || (η_arr = Array{Float64}(undef, 0, 0, 0))
has_eta = have_eta
    sorted_keys = String[part_keys[pi][k] for (pi, k) in want]

    nx, ny, nz, nt = size(u_arr)

    function flow_interpolator(lon::Real, lat::Real, z::Real, t::Real)
        # Find spatial indices via binary search with boundary-safe clamping
        i_f = searchsortedlast(lons_vec, Float64(lon))
        j_f = searchsortedlast(lats_vec, Float64(lat))
        k_f = searchsortedlast(deps_vec, Float64(z))
        i = nx > 1 ? clamp(i_f, 1, nx - 1) : 1
        j = ny > 1 ? clamp(j_f, 1, ny - 1) : 1
        k = nz > 1 ? clamp(k_f, 1, nz - 1) : 1

        # Spatial fractional coordinates
        sx = (nx > 1 && (lons_vec[i+1] - lons_vec[i]) != 0.0) ?
             clamp((Float64(lon) - lons_vec[i]) / (lons_vec[i+1] - lons_vec[i]), 0.0, 1.0) : 0.0
        sy = (ny > 1 && (lats_vec[j+1] - lats_vec[j]) != 0.0) ?
             clamp((Float64(lat) - lats_vec[j]) / (lats_vec[j+1] - lats_vec[j]), 0.0, 1.0) : 0.0
        sz = (nz > 1 && (deps_vec[k+1] - deps_vec[k]) != 0.0) ?
             clamp((Float64(z) - deps_vec[k])  / (deps_vec[k+1]  - deps_vec[k]),  0.0, 1.0) : 0.0

        # Find temporal bracket
        m_f = searchsortedlast(t_vec, Float64(t))
        m   = nt > 1 ? clamp(m_f, 1, nt - 1) : 1
        θ   = (nt > 1 && (t_vec[m+1] - t_vec[m]) != 0.0) ?
              clamp((Float64(t) - t_vec[m]) / (t_vec[m+1] - t_vec[m]), 0.0, 1.0) : 0.0

        function sample_at_time(arr, time_idx)
            i_next = nx > 1 ? i + 1 : i
            j_next = ny > 1 ? j + 1 : j
            k_next = nz > 1 ? k + 1 : k

            v000 = arr[i,      j,      k,      time_idx]
            v100 = arr[i_next, j,      k,      time_idx]
            v010 = arr[i,      j_next, k,      time_idx]
            v110 = arr[i_next, j_next, k,      time_idx]
            v001 = arr[i,      j,      k_next, time_idx]
            v101 = arr[i_next, j,      k_next, time_idx]
            v011 = arr[i,      j_next, k_next, time_idx]
            v111 = arr[i_next, j_next, k_next, time_idx]

            return (1.0 - sx) * (1.0 - sy) * (1.0 - sz) * v000 +
                   sx         * (1.0 - sy) * (1.0 - sz) * v100 +
                   (1.0 - sx) * sy         * (1.0 - sz) * v010 +
                   sx         * sy         * (1.0 - sz) * v110 +
                   (1.0 - sx) * (1.0 - sy) * sz         * v001 +
                   sx         * (1.0 - sy) * sz         * v101 +
                   (1.0 - sx) * sy         * sz         * v011 +
                   sx         * sy         * sz         * v111
        end

        function sample_eta_at_time(time_idx)
            i_next = nx > 1 ? i + 1 : i
            j_next = ny > 1 ? j + 1 : j
            e00 = η_arr[i,      j,      time_idx]
            e10 = η_arr[i_next, j,      time_idx]
            e01 = η_arr[i,      j_next, time_idx]
            e11 = η_arr[i_next, j_next, time_idx]
            return (1.0 - sx) * (1.0 - sy) * e00 +
                   sx         * (1.0 - sy) * e10 +
                   (1.0 - sx) * sy         * e01 +
                   sx         * sy         * e11
        end

        function interp(arr)
            val0 = sample_at_time(arr, m)
            if nt > 1 && θ > 0.0
                val1 = sample_at_time(arr, m + 1)
                return (1.0 - θ) * val0 + θ * val1
            else
                return val0
            end
        end

        function interp_eta()
            if !has_eta
                return 0.0
            end
            val0 = sample_eta_at_time(m)
            if nt > 1 && θ > 0.0
                val1 = sample_eta_at_time(m + 1)
                return (1.0 - θ) * val0 + θ * val1
            else
                return val0
            end
        end

        return (
            u = interp(u_arr),
            v = interp(v_arr),
            w = interp(w_arr),
            T = interp(T_arr),
            η = interp_eta()
        )
    end

    return flow_interpolator
end
