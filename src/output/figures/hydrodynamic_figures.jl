"""
    hydrodynamic_figures.jl

Hydrodynamic figures.

Eulerian hydrodynamic panels: advection, tracers, stratification, diffusion,
sections, timeseries, Hovmoller, bottom temperature, and the T-S diagram.
"""

using CairoMakie
import CairoMakie: record

"""
    plot_hydrodynamic_advection(
        hydrodynamics::Any;
        depth::Union{Nothing, Real} = nothing,
        depth_level::Union{Nothing, Int} = 1,
        time_seconds::Union{Nothing, Real} = nothing,
        time_index::Union{Nothing, Int} = 1,
        bathymetry_data::Union{Nothing, NamedTuple} = nothing,
        title::AbstractString = "Ocean Hydrodynamic Advection Velocity Field",
        output_path::Union{Nothing, AbstractString} = "outputs/hydrodynamic_advection.png",
        quiver_stride::Int = 2,
        colormap::Symbol = :turbo
    ) -> Figure

Plot 2D horizontal advection current velocity vector arrows \$\\\\\\\\\\\\\\boldsymbol{u}_h = (u, v)\$
over current speed magnitude \$|\\\\\\\\\\\\\\boldsymbol{u}_h| = \\\\\\\\\\\\\\sqrt{u^2 + v^2}\$ at a specific depth and time.

# Mathematical Formulation
Horizontal velocity field \$\\\\\\\\\\\\\\boldsymbol{u}_h(x, y, z_k, t_m) = (u, v)\$ with speed:
```math
|\\\\\\\\\\\\\\boldsymbol{u}_h| = \\\\\\\\\\\\\\sqrt{u^2 + v^2}
```
Flow orientation angle:
```math
\\theta = \\operatorname{atan2}(v, u)
```

# Inputs
- `hydrodynamics`: Model instance, JLD2 path, Zarr storage, or NamedTuple.
- `depth`: Target continuous depth in meters (e.g. `-25.0` or `25.0`).
- `depth_level`: Vertical discrete index (default: 1).
- `time_seconds`: Timestamp in seconds.
- `time_index`: Snapshot index.
- `bathymetry_data`: Optional background elevation dataset.
- `title`: Figure title.
- `output_path`: Destination file path.
- `quiver_stride`: Spatial subsampling step for vector arrows.
- `colormap`: CairoMakie colormap symbol (default `:turbo`).

# Outputs
- `Figure`: CairoMakie figure object.
"""
function plot_hydrodynamic_advection(
    hydrodynamics::Any;
    depth::Union{Nothing, Real} = nothing,
    depth_level::Union{Nothing, Int} = 1,
    time_seconds::Union{Nothing, Real} = nothing,
    time_index::Union{Nothing, Int} = 1,
    bathymetry_data::Union{Nothing, NamedTuple} = nothing,
    title::AbstractString = "Ocean Hydrodynamic Advection Velocity Field",
    output_path::Union{Nothing, AbstractString} = "outputs/hydrodynamic_advection.png",
    quiver_stride::Int = 2,
    colormap::Symbol = :turbo
)
    hydro = extract_hydrodynamic_dataset(
        hydrodynamics;
        depth = depth,
        depth_level = depth_level,
        time_seconds = time_seconds,
        time_index = time_index
    )
    lons = hydro.lons
    lats = hydro.lats
    k = hydro.depth_index
    depth_m = hydro.depth_m
    t_hr = round(hydro.time_seconds / 3600.0, digits = 1)

    u_k = hydro.u[:, :, k]
    v_k = hydro.v[:, :, k]
    spd_k = hydro.speed[:, :, k] .* 100.0

    fig = Figure(size = (980, 720), fontsize = 13)
    ax = Axis(
        fig[1, 1],
        title = "$(title)\n[Depth: $(depth_m) m | Time: $(t_hr) h]",
        xlabel = "Longitude (°E)",
        ylabel = "Latitude (°N)"
    )

    hm = heatmap!(ax, lons, lats, spd_k, colormap = colormap;
                   colorrange = _field_colorrange(spd_k))
    Colorbar(fig[1, 2], hm, label = "Current Speed (|u_h|, cm/s)")

    if !isnothing(bathymetry_data) && hasproperty(bathymetry_data, :elevation)
        contour!(
            ax,
            bathymetry_data.lon,
            bathymetry_data.lat,
            bathymetry_data.elevation,
            levels = [-2000.0, -1000.0, -500.0, -200.0, -100.0, -50.0],
            color = (:white, 0.4),
            linewidth = 1.0
        )
    end

    nx = length(lons)
    ny = length(lats)
    pts = Point2f[]
    dirs = Vec2f[]
    step_x = max(1, quiver_stride)
    step_y = max(1, quiver_stride)

    for i in 1:step_x:nx, j in 1:step_y:ny
        u_val = u_k[i, j]
        v_val = v_k[i, j]
        if !isnan(u_val) && !isnan(v_val)
            push!(pts, Point2f(lons[i], lats[j]))
            push!(dirs, Vec2f(u_val * 0.4, v_val * 0.4))
        end
    end

    if !isempty(pts)
        if isdefined(Makie, :arrows2d!)
            arrows2d!(
                ax, pts, dirs,
                tipwidth = 6, tiplength = 10, lengthscale = 1.0,
                tipcolor = :white, shaftcolor = :white
            )
        else
            arrows!(
                ax, pts, dirs,
                tipwidth = 6, tiplength = 10, lengthscale = 1.0,
                tipcolor = :white, shaftcolor = :white
            )
        end
    end

    if !isnothing(output_path)
        mkpath(dirname(output_path))
        save(output_path, fig)
    end

    return fig
end

"""
    plot_hydrodynamic_tracers(
        hydrodynamics::Any;
        depth::Union{Nothing, Real} = nothing,
        depth_level::Union{Nothing, Int} = 1,
        time_seconds::Union{Nothing, Real} = nothing,
        time_index::Union{Nothing, Int} = 1,
        title::AbstractString = "Hydrodynamic Model Seawater Tracers",
        output_path::Union{Nothing, AbstractString} = "outputs/hydrodynamic_tracers.png"
    ) -> Figure

Render a two-panel spatial distribution of active seawater tracer fields
(Temperature \$T\$ and Practical Salinity \$S\$) from the hydrodynamic model
at a specific depth and time.

# Mathematical Formulations
- **Thermal Tracer Distribution**: \$T(x, y, z_k, t_m)\$ [°C].
- **Haline Tracer Distribution**: \$S(x, y, z_k, t_m)\$ [PSU].

# Inputs
- `hydrodynamics`: Model, JLD2 file, Zarr storage, or NamedTuple.
- `depth`: Target depth in meters (e.g. `-50.0`).
- `depth_level`: Vertical index (default: 1).
- `time_seconds`: Timestamp in seconds.
- `time_index`: Snapshot index.
- `title`: Figure title.
- `output_path`: Destination path.

# Outputs
- `Figure`: CairoMakie figure object.
"""
function plot_hydrodynamic_tracers(
    hydrodynamics::Any;
    depth::Union{Nothing, Real} = nothing,
    depth_level::Union{Nothing, Int} = 1,
    time_seconds::Union{Nothing, Real} = nothing,
    time_index::Union{Nothing, Int} = 1,
    title::AbstractString = "Hydrodynamic Model Seawater Tracers",
    output_path::Union{Nothing, AbstractString} = "outputs/hydrodynamic_tracers.png"
)
    hydro = extract_hydrodynamic_dataset(
        hydrodynamics;
        depth = depth,
        depth_level = depth_level,
        time_seconds = time_seconds,
        time_index = time_index
    )
    lons = hydro.lons
    lats = hydro.lats
    k = hydro.depth_index
    depth_m = hydro.depth_m
    t_hr = round(hydro.time_seconds / 3600.0, digits = 1)

    t_mat = hydro.temperature[:, :, k]
    s_mat = hydro.salinity[:, :, k]

    fig = Figure(size = (1180, 540), fontsize = 12)

    # Panel 1: Temperature
    ax1 = Axis(
        fig[1, 1],
        title = "Seawater Temperature T (°C)\n[Depth: $(depth_m) m | t = $(t_hr) h]",
        xlabel = "Longitude (°E)",
        ylabel = "Latitude (°N)"
    )
    hm1 = heatmap!(ax1, lons, lats, t_mat, colormap = :thermal;
                     colorrange = _field_colorrange(t_mat))
    Colorbar(fig[1, 2], hm1, label = "Temperature (°C)")

    # Panel 2: Salinity
    ax2 = Axis(
        fig[1, 3],
        title = "Practical Salinity S (PSU)\n[Depth: $(depth_m) m | t = $(t_hr) h]",
        xlabel = "Longitude (°E)",
        ylabel = "Latitude (°N)"
    )
    hm2 = heatmap!(ax2, lons, lats, s_mat, colormap = :viridis;
                     colorrange = _field_colorrange(s_mat))
    Colorbar(fig[1, 4], hm2, label = "Salinity (PSU)")

    if !isnothing(output_path)
        mkpath(dirname(output_path))
        save(output_path, fig)
    end

    return fig
end

"""
    plot_hydrodynamic_stratification(
        hydrodynamics::Any;
        depth::Union{Nothing, Real} = nothing,
        depth_level::Union{Nothing, Int} = 1,
        time_seconds::Union{Nothing, Real} = nothing,
        time_index::Union{Nothing, Int} = 1,
        stations::Union{Nothing, AbstractVector} = nothing,
        title::AbstractString = "Hydrodynamic Seawater Stratification & Pycnocline Structure",
        output_path::Union{Nothing, AbstractString} = "outputs/hydrodynamic_stratification.png"
    ) -> Figure

Render a three-panel spatial and vertical analysis of ocean stratification:
1. Horizontal map of Brunt-Väisälä buoyancy frequency squared \$N^2\$ (\$10^{-4} s^{-2}\$).
2. Horizontal map of Practical Salinity \$S\$ (PSU).
3. 1D vertical water-column profiles of \$S(z), T(z), N^2(z)\$ at representative shelf stations.

# Mathematical Formulation
Buoyancy frequency squared (gravitational stability metric):
```math
N^2 = -\\\\\\\\\\\\\\frac{g}{\\\\\\\\\\\\\\rho_0} \\\\\\\\\\\\\\frac{\\\\\\\\\\\\\\partial \\\\\\\\\\\\\\rho}{\\\\\\\\\\\\\\partial z}
    \\approx g \\left( \\alpha \\\\\\\\\\\\\\frac{\\\\\\\\\\\\\\partial T}{\\\\\\\\\\\\\\partial z} - \\beta \\\\\\\\\\\\\\frac{\\\\\\\\\\\\\\partial S}{\\\\\\\\\\\\\\partial z} \\right)
```
Vertical salinity gradient (halocline strength):
```math
\\\\\\\\\\\\\\frac{\\\\\\\\\\\\\\partial S}{\\\\\\\\\\\\\\partial z} = \\\\\\\\\\\\\\frac{S(z_1) - S(z_2)}{\\Delta z}
```

# Inputs
- `hydrodynamics`: Model, JLD2 file, Zarr storage, or NamedTuple.
- `depth`: Depth in meters (default: surface).
- `depth_level`: Vertical level index.
- `time_seconds`: Simulation time in seconds.
- `time_index`: Snapshot index.
- `stations`: Optional vector of named coordinates `[(lon=..., lat=..., name=...)]`.
- `title`: Figure title.
- `output_path`: Destination path.

# Outputs
- `Figure`: CairoMakie figure object.
"""
function plot_hydrodynamic_stratification(
    hydrodynamics::Any;
    depth::Union{Nothing, Real} = nothing,
    depth_level::Union{Nothing, Int} = 1,
    time_seconds::Union{Nothing, Real} = nothing,
    time_index::Union{Nothing, Int} = 1,
    stations::Union{Nothing, AbstractVector} = nothing,
    title::AbstractString = "Hydrodynamic Seawater Stratification & Pycnocline Structure",
    output_path::Union{Nothing, AbstractString} = "outputs/hydrodynamic_stratification.png"
)
    hydro = extract_hydrodynamic_dataset(
        hydrodynamics;
        depth = depth,
        depth_level = depth_level,
        time_seconds = time_seconds,
        time_index = time_index
    )
    lons = hydro.lons
    lats = hydro.lats
    depths = hydro.depths
    k = hydro.depth_index
    depth_m = hydro.depth_m
    t_hr = round(hydro.time_seconds / 3600.0, digits = 1)

    n2_mat = hydro.stratification[:, :, k] .* 10000.0 # 10^-4 s^-2
    s_mat  = hydro.salinity[:, :, k]

    # Default representative shelf stations if not provided
    station_list = if !isnothing(stations) && !isempty(stations)
        stations
    else
        [
            (lon = -63.5, lat = 44.5, name = "Coastal (Halifax)"),
            (lon = -61.5, lat = 43.5, name = "Mid-Shelf (Emerald)"),
            (lon = -59.0, lat = 42.8, name = "Outer Slope (Laurentian)")
        ]
    end

    fig = Figure(size = (1500, 520), fontsize = 12)

    # Panel 1: Buoyancy Frequency N²
    ax1 = Axis(
        fig[1, 1],
        title = "Buoyancy Frequency N² (10⁻⁴ s⁻²)\n[Depth: $(depth_m) m | t = $(t_hr) h]",
        xlabel = "Longitude (°E)",
        ylabel = "Latitude (°N)"
    )
    hm1 = heatmap!(ax1, lons, lats, n2_mat, colormap = :ice;
                     colorrange = _field_colorrange(n2_mat))
    Colorbar(fig[1, 2], hm1, label = "N² (10⁻⁴ s⁻²)")

    # Panel 2: Practical Salinity S
    ax2 = Axis(
        fig[1, 3],
        title = "Practical Salinity S (PSU)\n[Depth: $(depth_m) m | t = $(t_hr) h]",
        xlabel = "Longitude (°E)",
        ylabel = "Latitude (°N)"
    )
    hm2 = heatmap!(ax2, lons, lats, s_mat, colormap = :viridis;
                     colorrange = _field_colorrange(s_mat))
    Colorbar(fig[1, 4], hm2, label = "Salinity (PSU)")

    # Panel 3: Vertical 1D Profiles at stations
    ax3 = Axis(
        fig[1, 5],
        title = "Vertical Stratification S(z) Profiles",
        xlabel = "Practical Salinity S (PSU)",
        ylabel = "Depth (m)"
    )

    colors = [:darkblue, :forestgreen, :darkorange, :purple]
    markers = [:circle, :diamond, :rect, :star5]

    for (st_idx, st) in enumerate(station_list)
        c_lon = st.lon
        c_lat = st.lat
        st_name = hasproperty(st, :name) ? st.name : "Station \$(st_idx)"
        col = colors[mod1(st_idx, length(colors))]
        mkr = markers[mod1(st_idx, length(markers))]

        i_st = argmin(abs.(lons .- c_lon))
        j_st = argmin(abs.(lats .- c_lat))

        # Mark station location on map panels
        scatter!(ax1, [lons[i_st]], [lats[j_st]], color = col, marker = mkr, markersize = 12, strokecolor = :white, strokewidth = 1.5)
        scatter!(ax2, [lons[i_st]], [lats[j_st]], color = col, marker = mkr, markersize = 12, strokecolor = :white, strokewidth = 1.5)

        # Plot vertical 1D salinity profile S(z)
        s_profile = hydro.salinity[i_st, j_st, :]
        lines!(ax3, s_profile, depths, color = col, linewidth = 2.4, label = st_name)
        scatter!(ax3, s_profile, depths, color = col, marker = mkr, markersize = 8)
    end

    axislegend(ax3, position = :rb)

    if !isnothing(output_path)
        mkpath(dirname(output_path))
        save(output_path, fig)
    end

    return fig
end

"""
    plot_hydrodynamic_diffusion(
        hydrodynamics::Any;
        depth::Union{Nothing, Real} = nothing,
        depth_level::Union{Nothing, Int} = 1,
        time_seconds::Union{Nothing, Real} = nothing,
        time_index::Union{Nothing, Int} = 1,
        title::AbstractString = "Hydrodynamic Turbulent Eddy Diffusivity & Viscosity",
        output_path::Union{Nothing, AbstractString} = "outputs/hydrodynamic_diffusion.png"
    ) -> Figure

Render a three-panel visualization of ocean turbulent mixing fields:
1. Horizontal map of turbulent vertical eddy diffusivity \$\\\\\\\\\\\\\\kappa_v\$ (\$m^2 s^{-1}\$).
2. Horizontal map of turbulent vertical eddy viscosity \$\\\\\\\\\\\\\\nu_v\$ (\$m^2 s^{-1}\$).
3. Vertical 1D profiles of \$\\\\\\\\\\\\\\kappa_v(z)\$ illustrating mixed-layer mixing and pycnocline barrier.

# Mathematical Formulation
Parameterization via gradient Richardson number \$Ri\$:
```math
Ri = \\\\\\\\\\\\\\frac{N^2}{\\left(\\\\\\\\\\\\\\frac{\\\\\\\\\\\\\\partial u}{\\\\\\\\\\\\\\partial z}\\right)^2 + \\left(\\\\\\\\\\\\\\frac{\\\\\\\\\\\\\\partial v}{\\\\\\\\\\\\\\partial z}\\right)^2}, \\quad
\\\\\\\\\\\\\\kappa_v(z) = \\\\\\\\\\\\\\kappa_{\\text{bg}} + \\\\\\\\\\\\\\frac{\\\\\\\\\\\\\\kappa_{\\text{max}}}{(1 + 5 Ri)^2} + \\\\\\\\\\\\\\kappa_{\\text{surf}} \\exp(z / h_{\\text{mix}})
```

# Inputs
- `hydrodynamics`: Model instance, JLD2 file, Zarr storage, or NamedTuple.
- `depth`: Depth in meters.
- `depth_level`: Vertical level index.
- `time_seconds`: Timestamp in seconds.
- `time_index`: Snapshot index.
- `title`: Figure title.
- `output_path`: Destination file path.

# Outputs
- `Figure`: CairoMakie figure object.
"""
function plot_hydrodynamic_diffusion(
    hydrodynamics::Any;
    depth::Union{Nothing, Real} = nothing,
    depth_level::Union{Nothing, Int} = 1,
    time_seconds::Union{Nothing, Real} = nothing,
    time_index::Union{Nothing, Int} = 1,
    title::AbstractString = "Hydrodynamic Turbulent Eddy Diffusivity & Viscosity",
    output_path::Union{Nothing, AbstractString} = "outputs/hydrodynamic_diffusion.png"
)
    hydro = extract_hydrodynamic_dataset(
        hydrodynamics;
        depth = depth,
        depth_level = depth_level,
        time_seconds = time_seconds,
        time_index = time_index
    )
    lons = hydro.lons
    lats = hydro.lats
    depths = hydro.depths
    k = hydro.depth_index
    depth_m = hydro.depth_m
    t_hr = round(hydro.time_seconds / 3600.0, digits = 1)

    diff_mat = hydro.diffusion[:, :, k] .* 10000.0 # 10^-4 m^2/s
    visc_mat = hydro.viscosity[:, :, k] .* 10000.0

    # Concatenated rather than `\$`-escaped: the previous title printed the interpolation source
    # verbatim, leaving depth_m and t_hr computed but unused.
    frame_note = "[Depth: $(round(depth_m, digits=1)) m | t = $(t_hr) h]"

    fig = Figure(size = (1500, 520), fontsize = 12)

    # Panel 1: Eddy Diffusivity κ_v
    ax1 = Axis(
        fig[1, 1],
        title = "Eddy Diffusivity κ_v (10⁻⁴ m² s⁻¹)\n" * frame_note,
        xlabel = "Longitude (°E)",
        ylabel = "Latitude (°N)"
    )
    hm1 = heatmap!(ax1, lons, lats, diff_mat, colormap = :plasma;
                   colorrange = _field_colorrange(diff_mat))
    Colorbar(fig[1, 2], hm1, label = "κ_v (10⁻⁴ m² s⁻¹)")

    # Panel 2: Eddy Viscosity ν_v
    ax2 = Axis(
        fig[1, 3],
        title = "Eddy Viscosity ν_v (10⁻⁴ m² s⁻¹)\n" * frame_note,
        xlabel = "Longitude (°E)",
        ylabel = "Latitude (°N)"
    )
    hm2 = heatmap!(ax2, lons, lats, visc_mat, colormap = :magma;
                   colorrange = _field_colorrange(visc_mat))
    Colorbar(fig[1, 4], hm2, label = "ν_v (10⁻⁴ m² s⁻¹)")

    # Panel 3: Vertical 1D Diffusivity Profiles
    ax3 = Axis(
        fig[1, 5],
        title = "Vertical Diffusivity Profiles κ_v(z)",
        xlabel = "Eddy Diffusivity (10⁻⁴ m² s⁻¹)",
        ylabel = "Depth (m)"
    )

    sample_points = [
        (lon = -63.5, lat = 44.5, name = "Coastal Mixed Zone", col = :royalblue),
        (lon = -61.5, lat = 43.5, name = "Mid-Shelf Pycnocline", col = :forestgreen),
        (lon = -59.0, lat = 42.8, name = "Slope Boundary", col = :crimson)
    ]

    for pt in sample_points
        i_pt = argmin(abs.(lons .- pt.lon))
        j_pt = argmin(abs.(lats .- pt.lat))
        k_prof = hydro.diffusion[i_pt, j_pt, :] .* 10000.0

        scatter!(ax1, [lons[i_pt]], [lats[j_pt]], color = pt.col, marker = :diamond, markersize = 12, strokecolor = :white, strokewidth = 1.5)
        lines!(ax3, k_prof, depths, color = pt.col, linewidth = 2.4, label = pt.name)
        scatter!(ax3, k_prof, depths, color = pt.col, marker = :diamond, markersize = 8)
    end

    axislegend(ax3, position = :rb)

    if !isnothing(output_path)
        mkpath(dirname(output_path))
        save(output_path, fig)
    end

    return fig
end

"""
    assert_plottable(data, description, variable) -> data

Validate that a masked 2D field contains at least two distinct finite values.

Makie's `contourf` recipe requires a finite value range; when a field is entirely `NaN`
(typically because the transect lies wholly landward, or the upstream record is empty) it
raises deep inside the compute pipeline with an opaque `linspace(NaN, NaN)` error. This
function converts that into an actionable message naming the transect and variable.

# Inputs
- `data`: Masked 2D array, possibly containing `NaN` for land/below-seafloor cells.
- `description`: Human-readable description of what was being plotted.
- `variable`: Variable symbol, included in the error message.

# Outputs
- `data`: returned unchanged, for call-site chaining.

# Throws
- `ErrorException` if no finite values are present, or if all finite values are identical.
"""
function assert_plottable(data, description::AbstractString, variable)
    finite_vals = filter(isfinite, data)
    if isempty(finite_vals)
        error(
            "Cannot render $(description): field '$(variable)' contains no finite values. " *
            "The transect lies entirely below the seafloor or landward of the wet domain, " *
            "or the source hydrodynamic record is empty/missing."
        )
    end
    if length(unique(finite_vals)) == 1
        error(
            "Cannot render $(description): field '$(variable)' is spatially uniform " *
            "($(first(finite_vals))); contouring requires a non-zero value range. This " *
            "usually indicates an uninitialised or single-level source record."
        )
    end
    return data
end

"""
    plot_hydrodynamic_section(
        hydrodynamics::Any;
        variable::Symbol = :temperature,
        transect_type::Symbol = :zonal,
        coordinate::Real = 44.0,
        time_seconds::Union{Nothing, Real} = nothing,
        time_index::Union{Nothing, Int} = 1,
        title::Union{Nothing, AbstractString} = nothing,
        output_path::Union{Nothing, AbstractString} = "outputs/hydrodynamic_section.png"
    ) -> Figure

Render a vertical cross-section (depth \$z\$ versus horizontal distance) along a specified
transect line for any active hydrodynamic field (temperature, salinity, stratification,
advection currents, turbulent diffusion) with seafloor bathymetry masking.

# Inputs
- `hydrodynamics`: Hydrodynamic model instance, JLD2 file, Zarr storage, or NamedTuple
                   containing (lons, lats, depths, bathymetry, u, v, w, temperature, salinity,
                   elevation, diffusion, viscosity, speed).
- `variable`: Variable symbol to plot. Options:
  - `:temperature` / `:T` - Seawater potential temperature (°C)
  - `:salinity` / `:S` - Practical salinity (PSU)
  - `:density` / `:rho` - Potential density (kg m⁻³)
  - `:stratification` / `:N2` - Buoyancy frequency squared N² (10⁻⁴ s⁻²)
  - `:diffusion` / `:kappa` - Vertical eddy diffusivity κ_v (10⁻⁴ m² s⁻¹)
  - `:viscosity` / `:nu` - Vertical eddy viscosity ν_v (10⁻⁴ m² s⁻¹)
  - `:speed` - Horizontal current speed √(u²+v²) (cm s⁻¹)
  - `:u` - Zonal velocity (cm s⁻¹)
  - `:v` - Meridional velocity (cm s⁻¹)
  - `:w` - Vertical velocity (mm s⁻¹)
- `transect_type`: Transect orientation:
  - `:zonal` / `:lat` / `:latitude` / `:east_west` - East-west section at fixed latitude
  - `:meridional` / `:lon` / `:longitude` / `:north_south` - North-south section at fixed longitude
- `coordinate`: Fixed latitude (°N) for zonal transects, or fixed longitude (°E) for meridional transects.
- `time_seconds`: Simulation time in seconds (selects nearest snapshot if not exact).
- `time_index`: Snapshot index (1-based) as alternative to time_seconds.
- `title`: Optional custom title string.
- `output_path`: Destination file path (directory created if needed).

# Outputs
- `Figure`: CairoMakie figure object with cross-section plot, bathymetry overlay, and colorbar.

# Notes
- Values below seafloor are masked to NaN (rendered transparent).
- Cross-section data is extracted by finding the grid index nearest to `coordinate`.
- Uses 16 contour levels by default; colormap chosen per variable.
"""
function plot_hydrodynamic_section(
    hydrodynamics::Any;
    variable::Symbol = :temperature,
    transect_type::Symbol = :zonal,
    section_type::Union{Nothing, Symbol} = nothing,
    coordinate::Real = 44.0,
    time_seconds::Union{Nothing, Real} = nothing,
    time_index::Union{Nothing, Int} = 1,
    title::Union{Nothing, AbstractString} = nothing,
    output_path::Union{Nothing, AbstractString} = "outputs/hydrodynamic_section.png"
)
    t_mode = !isnothing(section_type) ? section_type : transect_type
    is_zonal = t_mode in (:zonal, :lat, :latitude, :east_west)
    hydro = extract_hydrodynamic_dataset(
        hydrodynamics;
        time_seconds = time_seconds,
        time_index = time_index
    )
    lons = hydro.lons
    lats = hydro.lats
    depths = hydro.depths
    t_hr = round(hydro.time_seconds / 3600.0, digits = 1)

    nx = length(lons)
    ny = length(lats)
    nz = length(depths)

    # Resolve variable 3D field and colorbar properties
    field_3d, cmap, var_label = if variable == :salinity || variable == :S
        (hydro.salinity, :viridis, "Practical Salinity (PSU)")
    elseif variable == :density || variable == :rho
        (hydro.density, :dense, "Potential Density (kg m⁻³)")
    elseif variable == :stratification || variable == :N2
        (hydro.stratification .* 10000.0, :ice, "N² (10⁻⁴ s⁻²)")
    elseif variable == :diffusion || variable == :kappa
        (hydro.diffusion .* 10000.0, :plasma, "Diffusivity κ_v (10⁻⁴ m² s⁻¹)")
    elseif variable == :viscosity || variable == :nu
        (hydro.viscosity .* 10000.0, :magma, "Viscosity ν_v (10⁻⁴ m² s⁻¹)")
    elseif variable == :speed
        (hydro.speed .* 100.0, :turbo, "Current Speed (cm s⁻¹)")
    elseif variable == :u
        (hydro.u .* 100.0, :balance, "Zonal Velocity u (cm s⁻¹)")
    elseif variable == :v
        (hydro.v .* 100.0, :balance, "Meridional Velocity v (cm s⁻¹)")
    elseif variable == :w
        (hydro.w .* 1000.0, :balance, "Vertical Velocity w (mm s⁻¹)")
    else
        (hydro.temperature, :thermal, "Temperature T (°C)")
    end

    fig = Figure(size = (1050, 520), fontsize = 12)

    # Bathymetry is optional: when the source record carries no seabed, the full water
    # column is rendered rather than masking it against fabricated topography.
    has_bathy = !isnothing(hydro.bathymetry) && ndims(hydro.bathymetry) == 2 &&
                size(hydro.bathymetry) == (nx, ny)

    if is_zonal
        j_fixed = argmin(abs.(lats .- Float64(coordinate)))
        coord_val = lats[j_fixed]
        sec_data = zeros(Float64, nx, nz)
        b_section = has_bathy ? hydro.bathymetry[:, j_fixed] : fill(NaN, nx)

        for i in 1:nx, k in 1:nz
            z_val = depths[k]
            b_val = b_section[i]
            if isfinite(b_val) && z_val < b_val
                sec_data[i, k] = NaN
            else
                sec_data[i, k] = field_3d[i, j_fixed, k]
            end
        end

        assert_plottable(
            sec_data,
            "hydrodynamic cross-section along latitude $(round(coord_val, digits=2))°N",
            variable
        )

        fig_title = isnothing(title) ?
            "Hydrodynamic Vertical Cross-Section along Latitude $(round(coord_val, digits=2))°N [t = $(t_hr) h]" : title
        ax = Axis(fig[1, 1], title = fig_title, xlabel = "Longitude (°E)", ylabel = "Depth (m)")

        co = contourf!(ax, lons, depths, sec_data, colormap = cmap, levels = 16)
        has_bathy && lines!(ax, lons, b_section, color = :black, linewidth = 2.5)
        Colorbar(fig[1, 2], co, label = var_label)
    else
        i_fixed = argmin(abs.(lons .- Float64(coordinate)))
        coord_val = lons[i_fixed]
        sec_data = zeros(Float64, ny, nz)
        b_section = has_bathy ? hydro.bathymetry[i_fixed, :] : fill(NaN, ny)

        for j in 1:ny, k in 1:nz
            z_val = depths[k]
            b_val = b_section[j]
            if isfinite(b_val) && z_val < b_val
                sec_data[j, k] = NaN
            else
                sec_data[j, k] = field_3d[i_fixed, j, k]
            end
        end

        assert_plottable(
            sec_data,
            "hydrodynamic cross-section along longitude $(round(coord_val, digits=2))°E",
            variable
        )

        fig_title = isnothing(title) ?
            "Hydrodynamic Vertical Cross-Section along Longitude $(round(coord_val, digits=2))°E [t = $(t_hr) h]" : title
        ax = Axis(fig[1, 1], title = fig_title, xlabel = "Latitude (°N)", ylabel = "Depth (m)")

        co = contourf!(ax, lats, depths, sec_data, colormap = cmap, levels = 16)
        has_bathy && lines!(ax, lats, b_section, color = :black, linewidth = 2.5)
        Colorbar(fig[1, 2], co, label = var_label)
    end

    if !isnothing(output_path)
        mkpath(dirname(output_path))
        save(output_path, fig)
    end

    return fig
end

"""
    _field_colorrange(A::AbstractMatrix; rel_pad = 0.05) -> Tuple{Float64, Float64}

Colour limits for a hydrodynamics field, guarded against roundoff.

Makie's default autoscale spans `minimum` to `maximum`. When a field is effectively constant — the
model's fixed background diffusivity, or a level at rest — that range collapses to floating-point
noise, and the heatmap renders that noise at full contrast. The result reads as a physical plume
or band that is pure arithmetic: a κ field spanning 1e-8 m²/s was displayed with a dramatic
colour ramp and a sharp diagonal "front".

If the data range is negligible compared with the field's magnitude, widen the limits to a
symmetric band around the median so the panel renders flat, which is what the data actually says.
"""
function _field_colorrange(A::AbstractMatrix; rel_pad::Real = 0.05)
    v = filter(isfinite, vec(A))
    isempty(v) && return (0.0, 1.0)
    lo, hi = minimum(v), maximum(v)
    if hi - lo > max(abs(lo), abs(hi), 1.0) * 1e-6     # genuine structure
        return (lo, hi)
    end
    c = median(v)
    d = max(abs(c) * rel_pad, eps(c) * 1e3, 1e-12)
    return (c - d, c + d)
end

"""
    plot_hydrodynamic_timeseries(
        hydrodynamics::Any;
        lon::Real = -63.5,
        lat::Real = 44.0,
        depths::AbstractVector{<:Real} = [-2.5, -25.0, -50.0, -100.0],
        variable::Symbol = :temperature,
        title::Union{Nothing, AbstractString} = nothing,
        output_path::Union{Nothing, AbstractString} = "outputs/hydrodynamic_timeseries.png"
    ) -> Figure

Plot temporal evolution of any hydrodynamic parameter at a specified geographical station
across multiple depth levels throughout the simulation horizon.

# Inputs
- `hydrodynamics`: Model instance, JLD2 file, Zarr storage, or NamedTuple.
- `station`: Optional `(lon, lat)` coordinate tuple for the station.
- `lon, lat`: Mooring / station coordinates (used if `station = nothing`).
- `depths`: Vector of depth levels in meters.
- `variable`: Target variable (`:temperature`, `:salinity`, `:speed`, `:diffusion`,
  `:stratification`).
- `title`: Optional custom title.
- `output_path`: Destination file path.

# Outputs
- `Figure`: CairoMakie figure object.
"""
function plot_hydrodynamic_timeseries(
    hydrodynamics::Any;
    station::Union{Nothing, Tuple{<:Real, <:Real}} = nothing,
    lon::Real = -63.5,
    lat::Real = 44.0,
    depths::AbstractVector{<:Real} = [-2.5, -25.0, -50.0, -100.0],
    variable::Symbol = :temperature,
    title::Union{Nothing, AbstractString} = nothing,
    output_path::Union{Nothing, AbstractString} = "outputs/hydrodynamic_timeseries.png"
)
    hydro = extract_hydrodynamic_dataset(hydrodynamics)
    target_lon = !isnothing(station) ? Float64(station[1]) : Float64(lon)
    target_lat = !isnothing(station) ? Float64(station[2]) : Float64(lat)
    i_st = argmin(abs.(hydro.lons .- target_lon))
    j_st = argmin(abs.(hydro.lats .- target_lat))
    st_lon = round(hydro.lons[i_st], digits = 2)
    st_lat = round(hydro.lats[j_st], digits = 2)

    fig = Figure(size = (920, 480), fontsize = 12)
    var_title = uppercase(string(variable)[1:1]) * string(variable)[2:end]
    default_title = "Hydrodynamic $(var_title) Time-Series at ($(st_lon)°E, $(st_lat)°N)"
    ax = Axis(fig[1, 1], title = something(title, default_title), xlabel = "Simulation Time (hours)", ylabel = string(variable))

    colors = [:royalblue, :forestgreen, :darkorange, :purple, :crimson]

    # Walk the REAL snapshots. An earlier version took a single static value and added a
    # synthesised M2 cosine, which produced a convincing-looking "time series" containing no
    # simulated variation at all; when only one snapshot existed it also invented the time axis.
    # A diagnostic that fabricates the signal it is supposed to report is worse than none, so the
    # series is now built by re-extracting at successive time indices.
    n_snap = isdefined(hydro, :n_snapshots) ? hydro.n_snapshots : length(hydro.times)
    max_frames = 200
    snap_idx = n_snap <= max_frames ? collect(1:n_snap) :
               unique(round.(Int, range(1, n_snap, length = max_frames)))
    length(snap_idx) < 2 && (snap_idx = [1, max(1, n_snap)])

    series = [Float64[] for _ in depths]
    frame_hours = Float64[]
    for (si, k) in enumerate(snap_idx)
        frame = si == 1 ? hydro :
                extract_hydrodynamic_dataset(hydrodynamics; time_index = k)
        push!(frame_hours, first(frame.times) / 3600.0)
        for (d_idx, d_val) in enumerate(depths)
            kd = resolve_depth_index(frame.depths, d_val, nothing)
            push!(series[d_idx], _timeseries_value(frame, i_st, j_st, kd, variable))
        end
    end

    for (d_idx, d_val) in enumerate(depths)
        k = resolve_depth_index(hydro.depths, d_val, nothing)
        col = colors[mod1(d_idx, length(colors))]
        lines!(ax, frame_hours, series[d_idx], color = col, linewidth = 2.2,
               label = "Depth $(hydro.depths[k]) m")
    end

    axislegend(ax, position = :rt)

    if !isnothing(output_path)
        mkpath(dirname(output_path))
        save(output_path, fig)
    end

    return fig
end

"""
    plot_hydrodynamic_field(
        hydrodynamics::Any,
        variable::Symbol;
        depth::Union{Nothing, Real} = nothing,
        depth_level::Union{Nothing, Int} = 1,
        time_seconds::Union{Nothing, Real} = nothing,
        time_index::Union{Nothing, Int} = 1,
        title::Union{Nothing, AbstractString} = nothing,
        output_path::Union{Nothing, AbstractString} = nothing,
        colormap::Union{Nothing, Symbol} = nothing
    ) -> Figure

Flexible unified renderer for 2D spatial distribution of any hydrodynamic variable:
- `:temperature` / `:T`: Temperature (°C).
- `:salinity` / `:S`: Practical Salinity (PSU).
- `:density` / `:rho`: Potential Density (\$kg m^{-3}\$).
- `:stratification` / `:N2`: Buoyancy frequency squared (\$10^{-4} s^{-2}\$).
- `:salinity_stratification`: Vertical salinity gradient (\$PSU m^{-1}\$).
- `:diffusion` / `:kappa`: Turbulent eddy diffusivity (\$10^{-4} m^2 s^{-1}\$).
- `:viscosity` / `:nu`: Turbulent eddy viscosity (\$10^{-4} m^2 s^{-1}\$).
- `:speed`: Horizontal current speed (\$cm s^{-1}\$).
- `:w`: Vertical velocity (\$mm s^{-1}\$).
- `:vorticity`: Relative vertical vorticity (\$10^{-5} s^{-1}\$).
- `:elevation`: Sea surface height (cm).

# Inputs
- `hydrodynamics`: Model, JLD2 file, Zarr storage, or NamedTuple.
- `variable`: Symbol indicating field to render.
- `depth`: Depth in meters.
- `depth_level`: Vertical level index.
- `time_seconds`: Timestamp in seconds.
- `time_index`: Snapshot index.
- `title`: Optional custom figure title.
- `output_path`: Optional file destination.
- `colormap`: Optional custom colormap symbol.

# Outputs
- `Figure`: CairoMakie figure object.
"""
function plot_hydrodynamic_field(
    hydrodynamics::Any,
    variable::Symbol;
    depth::Union{Nothing, Real} = nothing,
    depth_level::Union{Nothing, Int} = 1,
    time_seconds::Union{Nothing, Real} = nothing,
    time_index::Union{Nothing, Int} = 1,
    title::Union{Nothing, AbstractString} = nothing,
    output_path::Union{Nothing, AbstractString} = nothing,
    colormap::Union{Nothing, Symbol} = nothing
)
    hydro = extract_hydrodynamic_dataset(
        hydrodynamics;
        depth = depth,
        depth_level = depth_level,
        time_seconds = time_seconds,
        time_index = time_index
    )
    lons = hydro.lons
    lats = hydro.lats
    k = hydro.depth_index
    depth_m = hydro.depth_m
    t_hr = round(hydro.time_seconds / 3600.0, digits = 1)

    mat_2d, default_cmap, unit_label = if variable == :salinity || variable == :S
        (hydro.salinity[:, :, k], :viridis, "Practical Salinity (PSU)")
    elseif variable == :density || variable == :rho
        (hydro.density[:, :, k], :dense, "Potential Density (kg m⁻³)")
    elseif variable == :stratification || variable == :N2
        (hydro.stratification[:, :, k] .* 10000.0, :ice, "N² (10⁻⁴ s⁻²)")
    elseif variable == :salinity_stratification
        (hydro.salinity_stratification[:, :, k], :viridis, "∂S/∂z (PSU m⁻¹)")
    elseif variable == :diffusion || variable == :kappa
        (hydro.diffusion[:, :, k] .* 10000.0, :plasma, "Diffusivity κ_v (10⁻⁴ m² s⁻¹)")
    elseif variable == :viscosity || variable == :nu
        (hydro.viscosity[:, :, k] .* 10000.0, :magma, "Viscosity ν_v (10⁻⁴ m² s⁻¹)")
    elseif variable == :speed
        (hydro.speed[:, :, k] .* 100.0, :turbo, "Current Speed (cm s⁻¹)")
    elseif variable == :w
        (hydro.w[:, :, k] .* 1000.0, :balance, "Vertical Velocity w (mm s⁻¹)")
    elseif variable == :vorticity
        (hydro.vorticity .* 100000.0, :balance, "Vorticity ζ (10⁻⁵ s⁻¹)")
    elseif variable == :elevation
        (hydro.elevation .* 100.0, :balance, "Sea Surface Height η (cm)")
    else
        (hydro.temperature[:, :, k], :thermal, "Temperature T (°C)")
    end

    active_cmap = something(colormap, default_cmap)
    fig = Figure(size = (950, 720), fontsize = 13)
    fig_title = isnothing(title) ?
        "Hydrodynamic $(variable) [Depth: $(depth_m) m | t = $(t_hr) h]" : title

    ax = Axis(fig[1, 1], title = fig_title, xlabel = "Longitude (°E)", ylabel = "Latitude (°N)")
    hm = heatmap!(ax, lons, lats, mat_2d, colormap = active_cmap)
    Colorbar(fig[1, 2], hm, label = unit_label)

    if !isnothing(output_path)
        mkpath(dirname(output_path))
        save(output_path, fig)
    end

    return fig
end


