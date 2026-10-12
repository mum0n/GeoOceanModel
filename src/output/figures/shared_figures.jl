"""
    shared_figures.jl

Shared figure utilities.

Helpers used by more than one figure family: the depth mask and the generic
multi-panel dashboard.
"""

using CairoMakie
import CairoMakie: record

"""
    mask_bathy!(mat::AbstractMatrix, bathy::AbstractMatrix, k::Int) -> AbstractMatrix

Mask matrix values below bathymetry (land cells) to NaN for transparent rendering.

# Inputs
- `mat`: Matrix to mask in-place
- `bathy`: Bathymetry matrix (same dimensions as mat)
- `k`: Vertical level index (unused but kept for API consistency)

# Outputs
- `mat`: Modified matrix with land cells set to NaN
"""
function mask_bathy!(mat::AbstractMatrix, bathy::AbstractMatrix, k::Int)
    nx, ny = size(mat)
    for i in 1:nx, j in 1:ny
        if !isnan(bathy[i, j]) && bathy[i, j] >= 0.0
            mat[i, j] = NaN
        end
    end
    return mat
end

"""
    plot_multi_panel_dashboard(
        hydrodynamics::Any;
        trajectories::Union{Nothing, NamedTuple} = nothing,
        bathymetry::Union{Nothing, NamedTuple} = nothing,
        connectivity::Union{Nothing, NamedTuple} = nothing,
        time_seconds::Union{Nothing, Real} = nothing,
        time_index::Union{Nothing, Int} = 1,
        output_path::Union{Nothing, AbstractString} = "outputs/multi_panel_dashboard.png",
        title::Union{Nothing, AbstractString} = nothing
    ) -> Figure

Generate a comprehensive 6-panel dashboard combining key hydrodynamic and Lagrangian diagnostics.

Panels:
1. Horizontal current speed (|u_h|) with vector field quiver overlay
2. Potential temperature (T) with isotherm contours
3. Vertical eddy diffusivity (κ_v) log-scale
4. Buoyancy frequency squared (N²) stratification
5. Vertical cross-section along central latitude (temperature)
6. Particle trajectories with stage coloration (if trajectories provided)

# Inputs
- `hydrodynamics`: JLD2 file, Zarr storage, model instance, or NamedTuple from `extract_hydrodynamic_dataset`
- `trajectories`: Optional Lagrangian tracking output from `track_larval_cohort`
- `bathymetry`: Optional bathymetry NamedTuple for background contours
- `connectivity`: Optional connectivity matrix from `compute_empirical_connectivity`
- `time_seconds`: Simulation time in seconds (nearest snapshot selected)
- `time_index`: Snapshot index (1-based, overrides `time_seconds`)
- `output_path`: Destination file path
- `title`: Optional custom title

# Outputs
- `Figure`: CairoMakie figure with 6 panels and shared colorbars

# Notes
- Seafloor masking applied to all panels (values below bathymetry set to NaN)
- Cross-section panel uses central latitude of domain
- Particle panel shows all stages with developmental color mapping
"""
function plot_multi_panel_dashboard(
    hydrodynamics::Any;
    trajectories::Union{Nothing, NamedTuple} = nothing,
    bathymetry::Union{Nothing, NamedTuple} = nothing,
    connectivity::Union{Nothing, NamedTuple} = nothing,
    time_seconds::Union{Nothing, Real} = nothing,
    time_index::Union{Nothing, Int} = 1,
    output_path::Union{Nothing, AbstractString} = "outputs/multi_panel_dashboard.png",
    title::Union{Nothing, AbstractString} = nothing
)::Figure
    hydro = extract_hydrodynamic_dataset(hydrodynamics; time_seconds = time_seconds, time_index = time_index)
    lons = hydro.lons
    lats = hydro.lats
    depths = hydro.depths
    bathy = hydro.bathymetry
    k_surf = 1  # surface layer

    t_hr = round(hydro.time_seconds / 3600.0, digits = 1)
    fig_title = isnothing(title) ?
        "Multi-Panel Hydrodynamic & Lagrangian Dashboard [t = $(t_hr) h]" : title

    fig = Figure(size = (1800, 1400), fontsize = 11)
    Label(fig[0, 1:3], fig_title, fontsize = 16, font = :bold)

    # Panel 1: Current Speed + Vectors
    ax1 = Axis(fig[1, 1], title = "Surface Current Speed |u_h| (cm/s)",
               xlabel = "Longitude (°E)", ylabel = "Latitude (°N)", aspect = DataAspect())
    speed_surf = Float64.(hydro.speed[:, :, k_surf] .* 100.0)
    mask_bathy!(speed_surf, bathy, k_surf)
    hm1 = heatmap!(ax1, lons, lats, speed_surf, colormap = :viridis, nan_color = :transparent)
    quiver!(ax1, lons[1:3:end], lats[1:3:end],
            Float64.(hydro.u[:, :, k_surf][1:3:end, 1:3:end] .* 100.0),
            Float64.(hydro.v[:, :, k_surf][1:3:end, 1:3:end] .* 100.0),
            color = :white, arrowsize = 8)
    Colorbar(fig[1, 2], hm1, label = "cm/s", height = Relative(0.8))

    # Panel 2: Temperature + Isotherms
    ax2 = Axis(fig[1, 2], title = "Surface Temperature (°C)",
               xlabel = "Longitude (°E)", ylabel = "Latitude (°N)", aspect = DataAspect())
    temp_surf = Float64.(hydro.temperature[:, :, k_surf])
    mask_bathy!(temp_surf, bathy, k_surf)
    hm2 = heatmap!(ax2, lons, lats, temp_surf, colormap = :thermal, nan_color = :transparent)
    contour!(ax2, lons, lats, temp_surf, levels = -2:1:20, color = :white, linewidth = 0.5)
    Colorbar(fig[1, 3], hm2, label = "°C", height = Relative(0.8))

    # Panel 3: Vertical Eddy Diffusivity (log scale)
    ax3 = Axis(fig[2, 1], title = "Vertical Eddy Diffusivity κ_v (log10, m²/s)",
               xlabel = "Longitude (°E)", ylabel = "Latitude (°N)", aspect = DataAspect())
    diff_surf = Float64.(hydro.diffusion[:, :, k_surf])
    mask_bathy!(diff_surf, bathy, k_surf)
    diff_log = log10.(clamp.(diff_surf, 1e-6, 1e-2))
    hm3 = heatmap!(ax3, lons, lats, diff_log, colormap = :turbid, nan_color = :transparent)
    Colorbar(fig[2, 2], hm3, label = "log10(κ_v)", height = Relative(0.8))

    # Panel 4: Stratification N²
    ax4 = Axis(fig[2, 2], title = "Buoyancy Frequency N² (10⁻⁴ s⁻²)",
               xlabel = "Longitude (°E)", ylabel = "Latitude (°N)", aspect = DataAspect())
    strat_surf = Float64.(hydro.stratification[:, :, k_surf] .* 10000.0)
    mask_bathy!(strat_surf, bathy, k_surf)
    hm4 = heatmap!(ax4, lons, lats, strat_surf, colormap = :ice, nan_color = :transparent)
    Colorbar(fig[2, 3], hm4, label = "10⁻⁴ s⁻²", height = Relative(0.8))

    # Panel 5: Vertical Cross-Section (Temperature)
    ax5 = Axis(fig[3, 1], title = "Temperature Cross-Section (Central Latitude)",
               xlabel = "Longitude (°E)", ylabel = "Depth (m)")
    j_mid = argmin(abs.(lats .- mean(lats)))
    nx, nz = length(lons), length(depths)
    sec_temp = zeros(Float64, nx, nz)
    b_section = bathy[:, j_mid]
    for i in 1:nx, k in 1:nz
        z_val = depths[k]
        b_val = b_section[i]
        if !isnan(b_val) && z_val < b_val
            sec_temp[i, k] = NaN
        else
            sec_temp[i, k] = hydro.temperature[i, j_mid, k]
        end
    end
    co5 = contourf!(ax5, lons, depths, sec_temp, colormap = :thermal, levels = 16)
    lines!(ax5, lons, b_section, color = :black, linewidth = 2.5)
    Colorbar(fig[3, 2], co5, label = "°C", height = Relative(0.8))

    # Panel 6: Particle Trajectories (if provided)
    if !isnothing(trajectories)
        ax6 = Axis(fig[3, 2:3], title = "Larval Trajectories (Stage Coloration)",
                   xlabel = "Longitude (°E)", ylabel = "Latitude (°N)", aspect = DataAspect())
        n_p, n_t = size(trajectories.lons)
        stage_colors = Dict(
            :zoea1 => :lightblue, :zoea2 => :dodgerblue, :megalopa => :orange,
            :instar1 => :green, :dead => :red, :settled => :purple
        )
        for p in 1:n_p
            if !isnothing(trajectories.alive) && !trajectories.alive[p]
                continue
            end
            stage_final = trajectories.stages[p, end]
            col = get(stage_colors, stage_final, :gray)
            lines!(ax6, trajectories.lons[p, :], trajectories.lats[p, :],
                   color = (col, 0.6), linewidth = 1.5)
            scatter!(ax6, trajectories.lons[p, end], trajectories.lats[p, end],
                     color = col, markersize = 6, strokecolor = :white, strokewidth = 0.5)
        end
        # Bathymetry contours
        contour!(ax6, lons, lats, bathy, levels = [-200, -500, -1000, -2000],
                 color = :gray, linewidth = 0.8, linestyle = :dash)
    else
        # Fallback: Connectivity matrix if provided
        if !isnothing(connectivity)
            ax6 = Axis(fig[3, 2], title = "Regional Connectivity P_ij",
                       xlabel = "Destination Stratum", ylabel = "Source Stratum")
            hm6 = heatmap!(ax6, connectivity.source_labels, connectivity.dest_labels,
                          connectivity.matrix, colormap = :blues)
            Colorbar(fig[3, 3], hm6, label = "P_ij")
        else
            ax6 = Axis(fig[3, 2], title = "Surface Salinity (PSU)")
            sal_surf = Float64.(hydro.salinity[:, :, k_surf])
            mask_bathy!(sal_surf, bathy, k_surf)
            hm6 = heatmap!(ax6, lons, lats, sal_surf, colormap = :haline, nan_color = :transparent)
            Colorbar(fig[3, 3], hm6, label = "PSU", height = Relative(0.8))
        end
    end

    if !isnothing(output_path)
        mkpath(dirname(output_path))
        save(output_path, fig)
    end

    return fig
end
