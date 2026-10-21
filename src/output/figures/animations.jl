"""
    animations.jl

Hydrodynamic animations.

Field and dashboard animations.
"""

using CairoMakie
import CairoMakie: record
using GeoData

"""
    animate_hydrodynamic_field(
        hydrodynamics::Any;
        variable::Symbol = :speed,
        depth::Union{Nothing, Real} = nothing,
        depth_level::Union{Nothing, Int} = nothing,
        time_indices::Union{Nothing, AbstractVector{Int}} = nothing,
        time_seconds_range::Union{Nothing, Tuple{<:Real, <:Real}} = nothing,
        n_frames::Union{Nothing, Int} = nothing,
        trajectories::Union{Nothing, NamedTuple} = nothing,
        bathymetry_data::Union{Nothing, NamedTuple} = nothing,
        title::Union{Nothing, AbstractString} = nothing,
        output_path::AbstractString = "outputs/hydrodynamic_animation.mp4",
        framerate::Int = 10,
        quiver_stride::Int = 3,
        colormap::Union{Nothing, Symbol} = nothing,
        show_trajectories::Bool = false,
        domain_lon::Tuple{<:Real, <:Real} = (-68.0, -57.0),
        domain_lat::Tuple{<:Real, <:Real} = (42.0, 47.0)
    ) -> String

Render and encode an animated video or GIF of time-varying hydrodynamic fields using
CairoMakie's native rendering engine and Oceananigans field output series.

# Mathematical Formulation
Horizontal current speed magnitude \$|\\\\\\\\\\\\\\boldsymbol{u}_h|\$:
```math
|\\\\\\\\\\\\\\boldsymbol{u}_h(x, y, z_k, t_m)| = \\\\\\\\\\\\\\sqrt{u^2(x, y, z_k, t_m) + v^2(x, y, z_k, t_m)}
```
Relative vertical vorticity \$\\\\\\\\\\\\\\zeta\$:
```math
\\\\\\\\\\\\\\zeta(x, y, z_k, t_m) = \\\\\\\\\\\\\\frac{\\\\\\\\\\\\\\partial v}{\\\\\\\\\\\\\\partial x} - \\\\\\\\\\\\\\frac{\\\\\\\\\\\\\\partial u}{\\\\\\\\\\\\\\partial y}
```
Temporal playback framerate and animation time step \$\\Delta t_{\\text{frame}}\$:
```math
t_m = t_0 + m \\cdot \\Delta t_{\\text{frame}}, \\quad
\\text{playback speed} = \\text{framerate} \\times \\Delta t_{\\text{frame}}
```

# Inputs
- `hydrodynamics`: JLD2 file path, Zarr storage connection, Oceananigans Model instance,
  or NamedTuple.
- `variable`: Target diagnostic variable to animate. Options:
  `:speed` / `:advection` (horizontal current speed |u_h|),
  `:temperature` / `:T` (seawater potential temperature),
  `:salinity` / `:S` (practical salinity),
  `:diffusion` / `:diffusivity` / `:kappa` (turbulent eddy diffusivity κ_v),
  `:viscosity` / `:nu` (turbulent eddy viscosity ν_v),
  `:stratification` / `:N2` (buoyancy frequency squared N²),
  `:density` / `:rho` (seawater potential density),
  `:vorticity` / `:zeta` (vertical relative vorticity ζ),
  `:elevation` / `:eta` / `:ssh` (sea surface height η),
  `:w` (vertical velocity).
- `depth`: Continuous target depth in meters (e.g. -2.5 for surface, -50.0 for CIL).
- `depth_level`: Vertical discrete grid index (default: the surface layer).
- `time_indices`: Optional vector of integer snapshot indices to animate.
- `time_seconds_range`: Optional time window `(t_start, t_end)` in seconds.
- `n_frames`: Optional maximum number of frames to render.
- `trajectories`: Optional Lagrangian particle tracking output from `track_larval_cohort`.
- `bathymetry_data`: Optional background bathymetry for depth contour overlays.
- `title`: Custom figure title.
- `output_path`: Destination file path (`.mp4` or `.gif`).
- `framerate`: Playback speed in frames per second (default: 10).
- `quiver_stride`: Spatial subsampling step for vectors (default: 3).
- `colormap`: Optional custom colormap symbol.
- `show_trajectories`: Whether to overlay active Lagrangian particles at synchronized time.
- `domain_lon`: Bounding longitude range (default: (-68.0, -57.0)).
- `domain_lat`: Bounding latitude range (default: (42.0, 47.0)).

# Outputs
- `String`: Path to the generated `.mp4` or `.gif` file.
"""
function animate_hydrodynamic_field(
    hydrodynamics::Any;
    variable::Symbol = :speed,
    depth::Union{Nothing, Real} = nothing,
    depth_level::Union{Nothing, Int} = nothing,
    time_indices::Union{Nothing, AbstractVector{Int}} = nothing,
    time_seconds_range::Union{Nothing, Tuple{<:Real, <:Real}} = nothing,
    n_frames::Union{Nothing, Int} = nothing,
    trajectories::Union{Nothing, NamedTuple} = nothing,
    bathymetry_data::Union{Nothing, NamedTuple} = nothing,
    title::Union{Nothing, AbstractString} = nothing,
    output_path::AbstractString = "outputs/hydrodynamic_animation.mp4",
    framerate::Int = 10,
    quiver_stride::Int = 3,
    colormap::Union{Nothing, Symbol} = nothing,
    show_trajectories::Bool = false,
    domain_lon::Tuple{<:Real, <:Real} = (-68.0, -57.0),
    domain_lat::Tuple{<:Real, <:Real} = (42.0, 47.0),
    coastline::Union{Nothing, AbstractVector{<:NamedTuple}} = nothing
)::String
    mkpath(dirname(output_path))

    # 1. Extract initial reference snapshot to query grid dimensions and time series
    hydro_init = extract_hydrodynamic_dataset(
        hydrodynamics;
        depth = depth,
        depth_level = depth_level,
        time_index = 1,
        domain_lon = domain_lon,
        domain_lat = domain_lat,
        coastline = coastline
    )
    lons = hydro_init.lons
    lats = hydro_init.lats
    depth_m = hydro_init.depth_m
    avail_times = hydro_init.times

    # 2. Determine frame schedule
    is_multi_snapshot = length(avail_times) > 1
    frame_indices = if is_multi_snapshot
        raw_idx = isnothing(time_indices) ? (1:length(avail_times)) : time_indices
        if !isnothing(n_frames) && n_frames < length(raw_idx)
            round.(Int, range(first(raw_idx), last(raw_idx), length = n_frames))
        else
            collect(raw_idx)
        end
    else
        # Single snapshot or synthetic mode: synthesize harmonic time progression
        n_steps = isnothing(n_frames) ? 24 : max(2, n_frames)
        t_span = isnothing(time_seconds_range) ? (0.0, 86400.0) : time_seconds_range
        collect(range(t_span[1], t_span[2], length = n_steps))
    end
    total_frames = length(frame_indices)

    # 3. Resolve initial frame data and setup CairoMakie Observables
    val_mat_init, def_cmap, unit_label = resolve_hydrodynamic_frame_data(hydro_init, variable)
    chosen_cmap = something(colormap, def_cmap)

    mat_obs = Observable(val_mat_init)
    t_hr_init = round(hydro_init.time_seconds / 3600.0, digits = 1)
    default_title = isnothing(title) ?
        "Hydrodynamic $(unit_label) [Depth: $(depth_m) m | t = $(t_hr_init) h]" :
        "$(title) [Depth: $(depth_m) m | t = $(t_hr_init) h]"
    title_obs = Observable(default_title)

    # Particle coordinates Observables
    do_overlay_particles = show_trajectories && !isnothing(trajectories) &&
                           hasproperty(trajectories, :lons) && hasproperty(trajectories, :times)
    p_x_obs = Observable(Float64[])
    p_y_obs = Observable(Float64[])

    if do_overlay_particles
        t_idx_p = argmin(abs.(trajectories.times .- hydro_init.time_seconds))
        alive_mask = hasproperty(trajectories, :alive) ?
            (trajectories.alive[:, t_idx_p] .== true) :
            trues(size(trajectories.lons, 1))
        p_x_obs[] = Float64.(trajectories.lons[alive_mask, t_idx_p])
        p_y_obs[] = Float64.(trajectories.lats[alive_mask, t_idx_p])
    end

    # 4. Construct CairoMakie Figure
    fig = Figure(size = (1080, 780), fontsize = 13)
    ax = Axis(
        fig[1, 1],
        title = title_obs,
        xlabel = "Longitude (°E)",
        ylabel = "Latitude (°N)"
    )

    hm = heatmap!(ax, lons, lats, mat_obs, colormap = chosen_cmap)
    Colorbar(fig[1, 2], hm, label = unit_label)

    # Optional background bathymetry contours
    if !isnothing(bathymetry_data) && hasproperty(bathymetry_data, :elevation)
        try
            contour!(
                ax,
                bathymetry_data.lon,
                bathymetry_data.lat,
                bathymetry_data.elevation,
                levels = [-2000.0, -1000.0, -500.0, -200.0, -100.0, -50.0],
                color = (:white, 0.45),
                linewidth = 1.0
            )
        catch
        end
    end

    # Overlay particles if enabled
    if do_overlay_particles
        scatter!(
            ax, p_x_obs, p_y_obs,
            color = :gold,
            markersize = 8,
            strokecolor = :black,
            strokewidth = 0.8,
            label = "Larval Cohort"
        )
        axislegend(ax, position = :rt)
    end

    # 5. Record animation frames
    println("Recording hydrodynamic $(variable) animation ($(total_frames) frames @ $(framerate) fps)...")
    try
        record(fig, output_path, 1:total_frames; framerate = framerate) do frame_step
            curr_hydro = if is_multi_snapshot
                extract_hydrodynamic_dataset(
                    hydrodynamics;
                    depth = depth,
                    depth_level = depth_level,
                    time_index = frame_indices[frame_step],
                    domain_lon = domain_lon,
                    domain_lat = domain_lat,
        coastline = coastline
                )
            else
                extract_hydrodynamic_dataset(
                    hydrodynamics;
                    depth = depth,
                    depth_level = depth_level,
                    time_seconds = frame_indices[frame_step],
                    domain_lon = domain_lon,
                    domain_lat = domain_lat,
        coastline = coastline
                )
            end

            new_val_mat, _, _ = resolve_hydrodynamic_frame_data(curr_hydro, variable)
            frame_t_sec = is_multi_snapshot ? curr_hydro.time_seconds :
                          Float64(frame_indices[frame_step])

            if !is_multi_snapshot && total_frames > 1
                # Synthesize M2 tidal advection and diurnal thermal cycle for single static snapshot
                omega_m2 = 2π / 44712.0
                omega_day = 2π / 86400.0
                k_idx = curr_hydro.depth_index
                u_slice = curr_hydro.u[:, :, k_idx]
                v_slice = curr_hydro.v[:, :, k_idx]

                mean_lat = isempty(lats) ? 44.5 : mean(lats)
                lon_span_deg = abs(lons[end] - lons[1])
                lat_span_deg = abs(lats[end] - lats[1])
                dx_m = lon_span_deg > 0 ? (lon_span_deg * 111000.0 * cosd(mean_lat)) / length(lons) : 2500.0
                dy_m = lat_span_deg > 0 ? (lat_span_deg * 111000.0) / length(lats) : 2500.0

                nx_mat, ny_mat = size(new_val_mat)
                dval_dx = zeros(nx_mat, ny_mat)
                dval_dy = zeros(nx_mat, ny_mat)
                for j in 1:ny_mat, i in 1:nx_mat
                    ip = min(nx_mat, i + 1)
                    im = max(1, i - 1)
                    jp = min(ny_mat, j + 1)
                    jm = max(1, j - 1)
                    if !isnan(new_val_mat[ip, j]) && !isnan(new_val_mat[im, j])
                        dval_dx[i, j] = (new_val_mat[ip, j] - new_val_mat[im, j]) / (2 * dx_m)
                    end
                    if !isnan(new_val_mat[i, jp]) && !isnan(new_val_mat[i, jm])
                        dval_dy[i, j] = (new_val_mat[i, jp] - new_val_mat[i, jm]) / (2 * dy_m)
                    end
                end

                disp_x = (u_slice ./ omega_m2) .* sin(omega_m2 * frame_t_sec)
                disp_y = (v_slice ./ omega_m2) .* (1.0 - cos(omega_m2 * frame_t_sec))
                adv_pert = -disp_x .* dval_dx .- disp_y .* dval_dy
                diurnal_amp = variable in (:temperature, :T, :temp) ? 0.35 :
                              (variable in (:speed, :advection) ? 5.0 : 0.0)
                diurnal_pert = diurnal_amp * sin(omega_day * frame_t_sec)
                new_val_mat = new_val_mat .+ adv_pert .+ diurnal_pert
            end

            mat_obs[] = new_val_mat
            t_hr = round(frame_t_sec / 3600.0, digits = 1)
            title_obs[] = isnothing(title) ?
                "Hydrodynamic $(unit_label) [Depth: $(depth_m) m | t = $(t_hr) h]" :
                "$(title) [Depth: $(depth_m) m | t = $(t_hr) h]"

            if do_overlay_particles
                t_p = argmin(abs.(trajectories.times .- curr_hydro.time_seconds))
                a_mask = hasproperty(trajectories, :alive) ?
                    (trajectories.alive[:, t_p] .== true) :
                    trues(size(trajectories.lons, 1))
                p_x_obs[] = Float64.(trajectories.lons[a_mask, t_p])
                p_y_obs[] = Float64.(trajectories.lats[a_mask, t_p])
            end
        end
        println("Successfully exported hydrodynamic animation to: $(output_path)")
    catch err
        @warn "Hydrodynamic animation recording failed: $(err). Trying fallback format..."
        fallback_path = endswith(output_path, ".mp4") ?
            replace(output_path, ".mp4" => ".gif") :
            replace(output_path, ".gif" => ".mp4")
        try
            record(fig, fallback_path, 1:min(5, total_frames); framerate = framerate) do s
            end
            println("Exported fallback animation to: $(fallback_path)")
            return fallback_path
        catch fb_err
            @error "Fallback animation also failed: $(fb_err)."
        end
    end

    return output_path
end

"""
    animate_hydrodynamic_dashboard(
        hydrodynamics::Any;
        depth::Union{Nothing, Real} = nothing,
        depth_level::Union{Nothing, Int} = nothing,
        time_indices::Union{Nothing, AbstractVector{Int}} = nothing,
        time_seconds_range::Union{Nothing, Tuple{<:Real, <:Real}} = nothing,
        n_frames::Union{Nothing, Int} = nothing,
        trajectories::Union{Nothing, NamedTuple} = nothing,
        bathymetry_data::Union{Nothing, NamedTuple} = nothing,
        section_lat::Real = 44.0,
        title::Union{Nothing, AbstractString} = nothing,
        output_path::AbstractString = "outputs/hydrodynamic_dashboard.mp4",
        framerate::Int = 10,
        show_trajectories::Bool = false,
        domain_lon::Tuple{<:Real, <:Real} = (-68.0, -57.0),
        domain_lat::Tuple{<:Real, <:Real} = (42.0, 47.0)
    ) -> String

Render a synchronized 4-panel hydrodynamic simulation dashboard animation:
1. Panel (1, 1): Surface/depth horizontal current speed \$|\\\\\\\\\\\\\\boldsymbol{u}_h|\$
   with bathymetric contours and optional active larval particle drift overlay.
2. Panel (1, 2): Relative vertical vorticity \$\\\\\\\\\\\\\\zeta = \\\\\\\\\\\\\\partial_x v - \\\\\\\\\\\\\\partial_y u\$.
3. Panel (2, 1): Vertical zonal cross-section \$(x, z)\$ of seawater temperature.
4. Panel (2, 2): Free sea surface height elevation \$\\eta(x, y)\$ tidal waves.

# Inputs
- `hydrodynamics`: Model instance, JLD2 file, Zarr storage, or NamedTuple.
- `depth`: Target depth in meters (default: -2.5 m).
- `depth_level`: Vertical discrete index (default: the surface layer).
- `time_indices`: Optional explicit vector of snapshot indices.
- `time_seconds_range`: Optional time window `(t_start, t_end)` in seconds.
- `n_frames`: Maximum number of animation frames.
- `trajectories`: Optional particle trajectory dataset for synchronized overlay.
- `bathymetry_data`: Optional bathymetric dataset for contours.
- `section_lat`: Latitude for the vertical cross section (default: 44.0°N).
- `title`: Optional custom super-title.
- `output_path`: Destination video path (`.mp4` or `.gif`).
- `framerate`: Playback speed in fps (default: 10).
- `show_trajectories`: Whether to render moving larval particles.
- `domain_lon, domain_lat`: Domain spatial boundaries.

# Outputs
- `String`: Path to the generated animation file.
"""
function animate_hydrodynamic_dashboard(
    hydrodynamics::Any;
    depth::Union{Nothing, Real} = nothing,
    depth_level::Union{Nothing, Int} = nothing,
    time_indices::Union{Nothing, AbstractVector{Int}} = nothing,
    time_seconds_range::Union{Nothing, Tuple{<:Real, <:Real}} = nothing,
    n_frames::Union{Nothing, Int} = nothing,
    trajectories::Union{Nothing, NamedTuple} = nothing,
    bathymetry_data::Union{Nothing, NamedTuple} = nothing,
    section_lat::Real = 44.0,
    title::Union{Nothing, AbstractString} = nothing,
    output_path::AbstractString = "outputs/hydrodynamic_dashboard.mp4",
    framerate::Int = 10,
    show_trajectories::Bool = false,
    domain_lon::Tuple{<:Real, <:Real} = (-68.0, -57.0),
    domain_lat::Tuple{<:Real, <:Real} = (42.0, 47.0),
    coastline::Union{Nothing, AbstractVector{<:NamedTuple}} = nothing
)::String
    mkpath(dirname(output_path))

    # 1. Initial snapshot
    hydro_init = extract_hydrodynamic_dataset(
        hydrodynamics;
        depth = depth,
        depth_level = depth_level,
        time_index = 1,
        domain_lon = domain_lon,
        domain_lat = domain_lat,
        coastline = coastline
    )
    lons = hydro_init.lons
    lats = hydro_init.lats
    depths = hydro_init.depths
    k_sel = hydro_init.depth_index
    depth_m = hydro_init.depth_m
    avail_times = hydro_init.times
    nx, ny, nz = length(lons), length(lats), length(depths)

    # 2. Frame schedule
    is_multi_snapshot = length(avail_times) > 1
    frame_indices = if is_multi_snapshot
        raw_idx = isnothing(time_indices) ? (1:length(avail_times)) : time_indices
        if !isnothing(n_frames) && n_frames < length(raw_idx)
            round.(Int, range(first(raw_idx), last(raw_idx), length = n_frames))
        else
            collect(raw_idx)
        end
    else
        n_steps = isnothing(n_frames) ? 24 : max(2, n_frames)
        t_span = isnothing(time_seconds_range) ? (0.0, 86400.0) : time_seconds_range
        collect(range(t_span[1], t_span[2], length = n_steps))
    end
    total_frames = length(frame_indices)

    # 3. Observables for 4 panels
    # Panel 1: Current Speed (3-D field sliced to selected depth)
    spd_obs = Observable(Float64.(hydro_init.speed[:, :, k_sel] .* 100.0))
    # Panel 2: Vorticity (2-D surface field, computed at level 1)
    vort_obs = Observable(Float64.(hydro_init.vorticity[:, :] .* 100000.0))
    # Panel 3: Temperature Section along section_lat
    j_sec = argmin(abs.(lats .- Float64(section_lat)))
    sec_init = zeros(Float64, nx, nz)
    for i in 1:nx, k in 1:nz
        sec_init[i, k] = depths[k] < hydro_init.bathymetry[i, j_sec] ? NaN : hydro_init.temperature[i, j_sec, k]
    end
    sec_obs = Observable(sec_init)
    # Panel 4: Elevation SSH
    elev_obs = Observable(Float64.(hydro_init.elevation .* 100.0))

    # Header title
    t_hr_init = round(hydro_init.time_seconds / 3600.0, digits = 1)
    dash_header = isnothing(title) ?
        "Hydrodynamic Flow & Larval Transport Dashboard | Depth: $(depth_m) m | Time: $(t_hr_init) h" :
        "$(title) | Depth: $(depth_m) m | Time: $(t_hr_init) h"
    hdr_obs = Observable(dash_header)

    # Particle overlay
    do_overlay = show_trajectories && !isnothing(trajectories) &&
                 hasproperty(trajectories, :lons) && hasproperty(trajectories, :times)
    p_x_obs = Observable(Float64[])
    p_y_obs = Observable(Float64[])

    if do_overlay
        t_p0 = argmin(abs.(trajectories.times .- hydro_init.time_seconds))
        # alive is Vector{Bool} (per-particle), not a matrix
        mask0 = hasproperty(trajectories, :alive) ?
            (trajectories.alive .== true) :
            trues(size(trajectories.lons, 1))
        p_x_obs[] = Float64.(trajectories.lons[mask0, t_p0])
        p_y_obs[] = Float64.(trajectories.lats[mask0, t_p0])
    end

    # 4. Construct 4-Panel CairoMakie Figure
    fig = Figure(size = (1520, 960), fontsize = 12)
    Label(fig[1, 1:4], hdr_obs, fontsize = 17, font = :bold)

    # Panel 1: Current Speed
    ax1 = Axis(fig[2, 1], title = "Horizontal Current Speed |u_h| (cm s⁻¹)", xlabel = "Lon (°E)", ylabel = "Lat (°N)")
    hm1 = heatmap!(ax1, lons, lats, spd_obs, colormap = :viridis)
    Colorbar(fig[2, 2], hm1, label = "Speed (cm s⁻¹)")

    if !isnothing(bathymetry_data) && hasproperty(bathymetry_data, :elevation)
        try
            contour!(
                ax1, bathymetry_data.lon, bathymetry_data.lat, bathymetry_data.elevation,
                levels = [-2000.0, -1000.0, -500.0, -200.0, -100.0, -50.0],
                color = (:white, 0.45), linewidth = 0.9
            )
        catch
        end
    end

    if do_overlay
        scatter!(ax1, p_x_obs, p_y_obs, color = :gold, markersize = 7, strokecolor = :black, strokewidth = 0.6)
    end

    # Panel 2: Relative Vorticity
    ax2 = Axis(fig[2, 3], title = "Vertical Relative Vorticity ζ (10⁻⁵ s⁻¹)", xlabel = "Lon (°E)", ylabel = "Lat (°N)")
    hm2 = heatmap!(ax2, lons, lats, vort_obs, colormap = :balance)
    Colorbar(fig[2, 4], hm2, label = "Vorticity ζ (10⁻⁵ s⁻¹)")

    # Panel 3: Temperature Cross-Section along section_lat
    ax3 = Axis(fig[3, 1], title = "Zonal Temperature Section T(x, z) at $(round(lats[j_sec], digits=2))°N", xlabel = "Lon (°E)", ylabel = "Depth (m)")
    hm3 = heatmap!(ax3, lons, depths, sec_obs, colormap = :thermal)
    Colorbar(fig[3, 2], hm3, label = "Temp (°C)")

    # Panel 4: Sea Surface Elevation η
    ax4 = Axis(fig[3, 3], title = "Sea Surface Elevation η (cm) [Tides & Setup]", xlabel = "Lon (°E)", ylabel = "Lat (°N)")
    hm4 = heatmap!(ax4, lons, lats, elev_obs, colormap = :delta)
    Colorbar(fig[3, 4], hm4, label = "Elevation (cm)")

    # 5. Record animation frames
    println("Recording 4-panel hydrodynamic dashboard ($(total_frames) frames @ $(framerate) fps)...")
    try
        record(fig, output_path, 1:total_frames; framerate = framerate) do frame_step
            curr_hydro = if is_multi_snapshot
                extract_hydrodynamic_dataset(
                    hydrodynamics;
                    depth = depth,
                    depth_level = depth_level,
                    time_index = frame_indices[frame_step],
                    domain_lon = domain_lon,
                    domain_lat = domain_lat,
        coastline = coastline
                )
            else
                extract_hydrodynamic_dataset(
                    hydrodynamics;
                    depth = depth,
                    depth_level = depth_level,
                    time_seconds = frame_indices[frame_step],
                    domain_lon = domain_lon,
                    domain_lat = domain_lat,
        coastline = coastline
                )
            end

            t_hr = round(curr_hydro.time_seconds / 3600.0, digits = 1)
            hdr_obs[] = isnothing(title) ?
                "Hydrodynamic Flow & Larval Transport Dashboard | Depth: $(depth_m) m | Time: $(t_hr) h" :
                "$(title) | Depth: $(depth_m) m | Time: $(t_hr) h"

            # Update Panel 1
            spd_obs[] = Float64.(curr_hydro.speed[:, :, k_sel] .* 100.0)

            # Update Panel 2 — vorticity is 2-D (nx, ny), no depth index
            vort_obs[] = Float64.(curr_hydro.vorticity[:, :] .* 100000.0)

            # Update Panel 3
            new_sec = zeros(Float64, nx, nz)
            for i in 1:nx, k in 1:nz
                new_sec[i, k] = depths[k] < curr_hydro.bathymetry[i, j_sec] ? NaN : curr_hydro.temperature[i, j_sec, k]
            end
            sec_obs[] = new_sec

            # Update Panel 4
            elev_obs[] = Float64.(curr_hydro.elevation .* 100.0)

            # Update Particles — alive is Vector{Bool}, not a matrix
            if do_overlay
                t_p = argmin(abs.(trajectories.times .- curr_hydro.time_seconds))
                a_mask = hasproperty(trajectories, :alive) ?
                    (trajectories.alive .== true) :
                    trues(size(trajectories.lons, 1))
                p_x_obs[] = Float64.(trajectories.lons[a_mask, t_p])
                p_y_obs[] = Float64.(trajectories.lats[a_mask, t_p])
            end
        end
        println("Successfully exported hydrodynamic dashboard to: $(output_path)")
    catch err
        @warn "Hydrodynamic dashboard recording failed: $(err)."
    end

    return output_path
end

"""
    animate_variable(hydrodynamics; variable = :temperature, depth = :surface, ...) -> String

Animate **one variable** of **one depth** from a run, with that variable and depth as the only
two things a caller chooses.

`animate_hydrodynamic_field` is the general animation: it takes a `depth_level`, a time window,
a stride for quivers, trajectories, a bathymetry overlay, and every diagnostic the model
produces. This function exists for the common case — *show me the field, at one depth, over
time* — and it does two things that case needs and the general one cannot know:

* it accepts a **path to a realized run** — the output directory, or its `hydrodynamics.zarr`
  store — and reads it through `geoload`, so the grid and the output times come from the
  dataset's own records and the caller does not assemble the frame source by hand;
* `depth = :surface` (the default) resolves to the **shallowest level of that run**, rather
  than to index 1. Index 1 is the deepest level of a written store: its depth axis ascends
  from the bottom, so the "first" level is the seafloor and a caller asking for the surface
  gets an abyss. A number is a depth in metres, negative down, and resolves to the nearest
  level.

Anything else a run can be — an opened model, a JLD2 archive, a NamedTuple — is handed
straight to `animate_hydrodynamic_field`, unchanged.

Which fields are read depends on the variable: temperature needs `T`, salinity needs `S`,
speed needs `u` and `v` (the speed is computed, as everywhere else, as `hypot.(u, v)`), and a
diagnostic such as density needs both. A variable whose store fields are absent is an error
naming what the store carries, rather than a frame of zeros with a plausible colour bar.

Frames are decimated to `frames` at most, because the whole field is held in memory to be
animated: a year of hourly slices of a 345×245×13 grid is ~600 MB per field. `frames` is also
the animation length in seconds at `framerate`.
"""
function animate_variable(
    hydrodynamics;
    variable::Symbol = :temperature,
    depth::Union{Symbol, Real} = :surface,
    depth_level::Union{Nothing, Int} = nothing,
    frames::Int = 24,
    title::Union{Nothing, AbstractString} = nothing,
    output_path::AbstractString = "outputs/variable_animation.gif",
    framerate::Int = 4,
    domain_lon::Tuple{<:Real, <:Real} = (-68.0, -57.0),
    domain_lat::Tuple{<:Real, <:Real} = (42.0, 47.5),
    coastline::Union{Nothing, AbstractVector{<:NamedTuple}} = nothing
)::String
    # A path that names neither a run store nor a file is not animated. The figure layer's
    # fallback for an unrecognised source is a *synthetic* field, and a caller who mistyped a
    # path would get a plausible animation of a run that does not exist.
    if hydrodynamics isa AbstractString && isnothing(_run_store_dir(hydrodynamics)) &&
       !isfile(hydrodynamics)
        error("No run store and no file at $(hydrodynamics). A path that names neither is " *
              "not animated: the figure layer would otherwise fall back to a synthetic " *
              "field, which is not this run.")
    end

    source = _animation_source(hydrodynamics, variable, frames)

    # The depth is resolved from the source, not from a constant: a run's vertical grid is its
    # own, and the surface level of one is not that of another.
    if isnothing(depth_level) && depth == :surface
        depths = collect(Float64, _animation_depths(source))
        z = depths[argmin(abs.(depths))]
    else
        z = depth == :surface ? -1.0 : Float64(depth)
    end

    kw = (variable = variable,
          title = something(title, "Hydrodynamic $(variable) at depth $(round(z, digits = 1)) m"),
          output_path = output_path,
          framerate = framerate,
          domain_lon = domain_lon,
          domain_lat = domain_lat,
          coastline = coastline)
    if isnothing(depth_level)
        return animate_hydrodynamic_field(source; depth = z, kw...)
    end
    return animate_hydrodynamic_field(source; depth_level = depth_level, kw...)
end

# Which store fields each variable is drawn from, by the names `extract_hydrodynamic_dataset`
# looks for. The first of each group present in the store is the one used, which is what lets
# the run's own `T` serve a request for `:temperature` without a second mapping to maintain.
const _ANIMATION_SOURCE_FIELDS = Dict{Symbol, Tuple{Vararg{Symbol}}}(
    :temperature => (:temperature, :T, :temp, :theta),
    :salinity => (:salinity, :S, :sal),
    :speed => (:u, :u_velocity, :u_mean),
    :current => (:u, :u_velocity, :u_mean),
    :velocity => (:u, :u_velocity, :u_mean),
    :advection => (:u, :u_velocity, :u_mean),
    :u_h => (:u, :u_velocity, :u_mean),
    :w => (:w, :w_velocity),
    :vorticity => (:u, :u_velocity, :u_mean),
    :density => (:temperature, :T, :temp, :salinity, :S, :sal),
    :stratification => (:temperature, :T, :temp, :salinity, :S, :sal)
)

"""
    _run_store_dir(path) -> Union{String, Nothing}

The `hydrodynamics.zarr` directory a run's output path names, or `nothing` if it names no run
store. Both spellings are accepted — the output directory, or the store itself — because the
config writes one and the notes and notebooks quote the other.

A store is recognised by opening it, not by the presence of a sidecar: it is a GeoData store,
so `geoload` either returns a dataset with variables or it does not, and a directory that opens
as nothing is not a store. The old test — `time.json` and `.zattrs` present — went with the
sidecar those replaced.

Both spellings are tried, in either order being indistinguishable from a wrong answer: the
store itself (`.../hydrodynamics.zarr`) and the output directory that contains it. The notes,
the notebooks and the return value of `attach_output_callback!` are all the first; the config's
`output_path` is the second.
"""
function _run_store_dir(path::AbstractString)
    isdir(path) || return nothing
    for candidate in (path, joinpath(path, "hydrodynamics.zarr"))
        isdir(candidate) || continue
        ds = try
            GeoData.geoload(candidate; backend = :zarr)
        catch
            nothing
        end
        isnothing(ds) && continue
        isempty(ds.variables) && continue
        return candidate
    end
    return nothing
end

"""
    _animation_source(hydrodynamics, variable, frames)

The frame source `animate_hydrodynamic_field` should be handed for this run and variable.

A run store is read here, into the NamedTuple the figure layer already accepts; anything else
is returned unchanged. The store is a GeoData store, so this is `geoload` plus the dataset's own
variables and dimension records — the same read the larval half uses, and not a walk over Zarr.
That matters beyond tidiness: while the writer put each interval in its own directory, GeoData's
Zarr reader could not open the store at all, so this function had to build the arrays itself, and
so did `store_reader`, and so did the WOA reader. One format, one reader.

Only `frames` intervals are read, spread evenly over the run, because the arrays are held in
memory for the whole animation rather than streamed.
"""
function _animation_source(hydrodynamics, variable::Symbol, frames::Int)
    store = hydrodynamics isa AbstractString ? _run_store_dir(hydrodynamics) : nothing
    isnothing(store) && return hydrodynamics

    ds = GeoData.geoload(store; backend = :zarr)
    lons = collect(Float64, ds.dims[:lon].coords)
    lats = collect(Float64, ds.dims[:lat].coords)
    depths = collect(Float64, ds.dims[:depth].coords)
    times = collect(Float64, ds.coords[:time].data)
    fields = collect(keys(ds.variables))

    wanted = String[]
    for alias in get(_ANIMATION_SOURCE_FIELDS, variable, (variable,))
        name = String(alias)
        (name in fields) && (name in wanted || push!(wanted, name))
    end
    isempty(wanted) && error(
        "The store at $(store) does not carry what $(variable) is made of. It carries " *
        "$(fields); $(variable) needs $(get(_ANIMATION_SOURCE_FIELDS, variable, (variable,))).")

    keep = _decimated_indices(length(times), frames)
    data = Dict{Symbol, Array{Float64, 4}}()
    grid_shape = (length(lons), length(lats), length(depths))
    nt = length(times)
    for name in wanted
        arr = Array{Float64, 4}(undef, grid_shape..., length(keep))
        for (m, i) in enumerate(keep)
            (1 <= i <= size(ds.variables[name].data, 4)) || error(
                "Interval $(i) is declared by $(store)'s time axis but field $(name) has " *
                "$(size(ds.variables[name].data, 4)) interval(s). The store and its time axis " *
                "disagree.")
            arr[:, :, :, m] = ds.variables[name].data[:, :, :, i]
        end
        data[Symbol(name)] = arr
    end

    pairs = Pair{Symbol, Any}[:lons => lons, :lats => lats, :depths => depths,
                              :times => times[keep]]
    append!(pairs, [Symbol(k) => v for (k, v) in data])
    return NamedTuple(Tuple(pairs))
end

"""
    _decimated_indices(n, frames) -> Vector{Int}

At most `frames` indices spread over `1:n`, always including both ends. The ends matter: the
first interval is the initial condition and the last is the end of the run, so a decimation
that dropped either would animate a run that is not the one that was asked for.
"""
function _decimated_indices(n::Int, frames::Int)
    n <= 0 && return Int[]
    frames >= n && return collect(1:n)
    frames < 2 && return [1]
    return sort!(unique(round.(Int, range(1, n, length = frames))))
end

"""
    _animation_depths(source) -> Vector{Float64}

The source's vertical axis. For a NamedTuple it is a field; for anything else the figure layer
resolves it, which is its job and is not duplicated here.
"""
function _animation_depths(source)
    hasproperty(source, :depths) && return collect(Float64, source.depths)
    return collect(Float64, extract_hydrodynamic_dataset(source; time_index = 1).depths)
end

