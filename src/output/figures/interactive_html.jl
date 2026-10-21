"""
    interactive_html.jl

Interactive Leaflet dashboard engine.

The standalone HTML dashboard and the JavaScript it embeds. One file, because
the embedded script calls the helpers and the helpers build the script;
separating them would split a template from the code that fills it.
"""

using CairoMakie
import CairoMakie: record

"""
    export_interactive_tracks_html(
        output_path::AbstractString;
        trajectories::Union{Nothing, NamedTuple} = nothing,
        runs_data::Union{Nothing, AbstractDict} = nothing,
        hydrodynamics::Union{Nothing, NamedTuple, AbstractDict, Any} = nothing,
        gridded_dispersal::Union{Nothing, NamedTuple} = nothing,
        connectivity::Union{Nothing, NamedTuple} = nothing,
        recruitment_metrics::Union{Nothing, NamedTuple} = nothing,
        bathymetry::Union{Nothing, NamedTuple} = nothing,
        strata_definitions::Union{Nothing, AbstractVector} = nothing,
        title::AbstractString = "Interactive Larval Dispersal & Demographic Connectivity Map"
    ) -> String

Generate a standalone, interactive HTML5 + Leaflet.js oceanographic dashboard
visualizing 4D Lagrangian particle tracks, developmental stages, DVM behaviors,
hydrodynamic Eulerian flow fields (advection currents, temperature, salinity,
free surface elevation, upwelling, bathymetry), and regional Crab Fishing Area (CFA) connectivity.

# Mathematical & Behavioral Visualization
- **4D Trajectories**: \$\\mathbf{x}_p(t) = (\\\\\\\\lambda_p(t), \\phi_p(t), z_p(t))\$ with
  temperature exposure \$T_p(t)\$ and cumulative degree-days \$DD_p(t)\$.
- **Hydrodynamic Eulerian Advection**: Horizontal current velocity vector field
  \$\\\\\\\\\\\\\\boldsymbol{u}_h = (u, v)\$, speed \$|\\\\\\\\\\\\\\boldsymbol{u}_h|\$, and flow direction.
- **Hydrodynamic Tracers**: Continuous thermal \$T(x,y,z)\$ and haline \$S(x,y,z)\$
  stratification fields rendered on responsive hardware-accelerated canvas overlays.
- **Upwelling & Sea Surface Height**: Vertical velocity \$w\$ and free surface elevation \$\\eta\$.
- **Developmental Molting Coloration**: Zoea I \$\\to\$ IV, Megalopa, Instar I (settled).
- **Playback Animation**: Interactive time-stepping simulation scrubber and velocity trails.

# Inputs
- `output_path::AbstractString`: Destination path (e.g. `"outputs/interactive_larval_tracks.html"`).
- `trajectories::NamedTuple`: Particle tracking trajectory NamedTuple.
- `runs_data`: Optional dictionary mapping run names to data bundles.
- `hydrodynamics`: Optional hydrodynamic model output (Oceananigans model, NamedTuple, or Dict).
- `gridded_dispersal`: Optional 2D spatial dispersal fields.
- `connectivity`: Optional connectivity transition matrix.
- `recruitment_metrics`: Optional summary recruitment metrics.
- `bathymetry`: Optional bathymetry dataset.
- `strata_definitions`: Optional spatial management boundaries.
- `title::AbstractString`: Dashboard header title.

# Outputs
- `String`: Path to the generated interactive HTML file.
"""
function export_interactive_tracks_html(
    output_path::AbstractString;
    trajectories::Union{Nothing, NamedTuple} = nothing,
    runs_data::Union{Nothing, AbstractDict} = nothing,
    hydrodynamics::Union{Nothing, NamedTuple, AbstractDict, Any} = nothing,
    gridded_dispersal::Union{Nothing, NamedTuple} = nothing,
    connectivity::Union{Nothing, NamedTuple} = nothing,
    recruitment_metrics::Union{Nothing, NamedTuple} = nothing,
    bathymetry::Union{Nothing, NamedTuple} = nothing,
    strata_definitions::Union{Nothing, AbstractVector} = nothing,
    title::AbstractString = "Interactive Larval Dispersal & Demographic Connectivity Map",
    max_tracks::Int = 400,
    max_points_per_track::Int = 900
)::String
    out_dir = dirname(output_path)
    if !isempty(out_dir) && !isdir(out_dir)
        mkpath(out_dir)
    end

    # 1. Structure runs dictionary
    runs_dict = Dict{String, Any}()
    if !isnothing(runs_data) && !isempty(runs_data)
        for (k, v) in runs_data
            key_str = string(k)
            t_obj = hasproperty(v, :trajectories) ? v.trajectories : (v isa NamedTuple && hasproperty(v, :lons) ? v : nothing)
            if !isnothing(t_obj)
                runs_dict[key_str] = (
                    trajectories = t_obj,
                    gridded_dispersal = hasproperty(v, :gridded_dispersal) ? v.gridded_dispersal : (hasproperty(v, :empirical_movement) ? v.empirical_movement : gridded_dispersal),
                    connectivity = hasproperty(v, :connectivity) ? v.connectivity : connectivity,
                    recruitment_metrics = hasproperty(v, :recruitment_metrics) ? v.recruitment_metrics : recruitment_metrics,
                    hydrodynamics = hasproperty(v, :hydrodynamics) ? v.hydrodynamics : (hasproperty(v, :model) ? v.model : hydrodynamics)
                )
            end
        end
    elseif !isnothing(trajectories)
        runs_dict["Active Simulation"] = (
            trajectories = trajectories,
            gridded_dispersal = gridded_dispersal,
            connectivity = connectivity,
            recruitment_metrics = recruitment_metrics,
            hydrodynamics = hydrodynamics
        )
    else
        error("Either `trajectories` or `runs_data` must be provided to `export_interactive_tracks_html`.")
    end

    # 2. Extract Management Strata and Centroids
    strata_list = if isnothing(strata_definitions)
        auto_polys = load_cfa_polygons("inputs")
        if !isempty(auto_polys)
            [
                (
                    name = string(s.name),
                    color = s.color,
                    is_polygon = true,
                    polygon = s.coordinates,
                    lon_min = s.lon[1],
                    lon_max = s.lon[2],
                    lat_min = s.lat[1],
                    lat_max = s.lat[2],
                    centroid_lat = mean([pt[1] for pt in s.coordinates]),
                    centroid_lon = mean([pt[2] for pt in s.coordinates])
                ) for s in auto_polys
            ]
        else
            [
                (name = "CFA 20-22 (Eastern NS)", color = "#3B82F6", is_polygon = false, polygon = Any[], lon_min = -62.0, lon_max = -57.0, lat_min = 44.5, lat_max = 47.5, centroid_lat = 46.0, centroid_lon = -59.5),
                (name = "CFA 23-24 (Middle Shelf)", color = "#10B981", is_polygon = false, polygon = Any[], lon_min = -64.5, lon_max = -60.0, lat_min = 43.0, lat_max = 45.5, centroid_lat = 44.25, centroid_lon = -62.25),
                (name = "CFA 4X (Southwest NS)", color = "#F59E0B", is_polygon = false, polygon = Any[], lon_min = -67.5, lon_max = -63.5, lat_min = 42.0, lat_max = 44.5, centroid_lat = 43.25, centroid_lon = -65.5),
                (name = "Offshore / Slope", color = "#8B5CF6", is_polygon = false, polygon = Any[], lon_min = -68.0, lon_max = -57.0, lat_min = 41.5, lat_max = 43.0, centroid_lat = 42.25, centroid_lon = -62.5)
            ]
        end
    else
        [
            (
                name = string(s.name),
                color = hasproperty(s, :color) ? string(s.color) : "#3B82F6",
                is_polygon = hasproperty(s, :coordinates) && length(s.coordinates) >= 3,
                polygon = hasproperty(s, :coordinates) ? s.coordinates :
                          (hasproperty(s, :lons) && hasproperty(s, :lats) ? [[s.lats[i], s.lons[i]] for i in 1:length(s.lons)] : Any[]),
                lon_min = hasproperty(s, :lon) ? Float64(s.lon[1]) : (hasproperty(s, :lon_min) ? Float64(s.lon_min) : -68.0),
                lon_max = hasproperty(s, :lon) ? Float64(s.lon[2]) : (hasproperty(s, :lon_max) ? Float64(s.lon_max) : -57.0),
                lat_min = hasproperty(s, :lat) ? Float64(s.lat[1]) : (hasproperty(s, :lat_min) ? Float64(s.lat_min) : 42.0),
                lat_max = hasproperty(s, :lat) ? Float64(s.lat[2]) : (hasproperty(s, :lat_max) ? Float64(s.lat_max) : 47.0),
                centroid_lat = hasproperty(s, :coordinates) && !isempty(s.coordinates) ? mean([pt[1] for pt in s.coordinates]) : (hasproperty(s, :lat) ? (s.lat[1]+s.lat[2])/2.0 : 44.0),
                centroid_lon = hasproperty(s, :coordinates) && !isempty(s.coordinates) ? mean([pt[2] for pt in s.coordinates]) : (hasproperty(s, :lon) ? (s.lon[1]+s.lon[2])/2.0 : -63.0)
            ) for s in strata_definitions
        ]
    end

    strata_json_entries = String[]
    for s in strata_list
        poly_str = if s.is_polygon && !isempty(s.polygon)
            "[" * join(["[$(pt[1]),$(pt[2])]" for pt in s.polygon], ",") * "]"
        else
            "[]"
        end
        s_entry = """{
            "name": "$(s.name)",
            "color": "$(s.color)",
            "is_polygon": $(s.is_polygon),
            "polygon": $(poly_str),
            "lon_min": $(s.lon_min),
            "lon_max": $(s.lon_max),
            "lat_min": $(s.lat_min),
            "lat_max": $(s.lat_max),
            "centroid_lat": $(s.centroid_lat),
            "centroid_lon": $(s.centroid_lon)
        }"""
        push!(strata_json_entries, s_entry)
    end
    strata_json = "[" * join(strata_json_entries, ",\n") * "]"

    # 3. Serialize each run
    run_json_blocks = String[]
    for (run_key, run_data) in runs_dict
        trajs = canonicalize_trajectories(run_data.trajectories)
        lons = trajs.lons
        lats = trajs.lats
        depths = trajs.depths
        stages = trajs.stages
        times = trajs.times
        n_p, n_t = size(lons)

        temps = trajs.temperatures
        dds = trajs.degree_days_timeseries
        survs = trajs.survival_probability

        statuses = [string(s) for s in trajs.settlement_status]
        alive = trajs.alive
        ages = trajs.settlement_age
        ids = trajs.ids

        # ── Payload decimation ────────────────────────────────────────────────
        # The HTML embeds every array as inline JSON, so a full-resolution run is
        # unusable in a browser: 500 larvae over a 730-day horizon at dt = 300 s is
        # ~210 000 steps x 7 arrays, which produced a 1.7 GB file that would not open at all.
        # Thin in time and cap the number of tracks so the page stays interactive.
        # Traces are decimated uniformly (endpoints always retained), and the number of
        # tracks shown is capped, but every *statistic* below is still computed from the
        # FULL-resolution arrays, so the summary numbers are unaffected.
        track_idx = if n_p > max_tracks
            unique(round.(Int, range(1, n_p, length = max_tracks)))
        else
            collect(1:n_p)
        end
        time_idx = if n_t > max_points_per_track
            unique(round.(Int, range(1, n_t, length = max_points_per_track)))
        else
            collect(1:n_t)
        end
        n_tracks_shown = length(track_idx)
        n_points_shown = length(time_idx)
        # Time axis must be thinned on the same indices as the per-track arrays, or the
        # browser indexes past the end of the series.
        t_days = [round(times[t] / 86400.0, digits = 3) for t in time_idx]
        decimated = (n_tracks_shown < n_p) || (n_points_shown < n_t)
        if decimated
            @info "export_interactive_tracks_html: decimated payload to " *
                  "$n_tracks_shown of $n_p tracks and $n_points_shown of $n_t time points " *
                  "(inline JSON cannot hold the full $n_p x $n_t). Statistics are computed " *
                  "from the full-resolution data and are unaffected."
        end

        p_entries = String[]
        for p in track_idx
            p_lons = [round(lons[p, t], digits = 4) for t in time_idx]
            p_lats = [round(lats[p, t], digits = 4) for t in time_idx]
            p_depths = [round(depths[p, t], digits = 0) for t in time_idx]
            p_temps = [round(temps[p, t], digits = 1) for t in time_idx]
            p_dds = [round(dds[p, t], digits = 1) for t in time_idx]
            p_survs = [round(survs[p, t], digits = 3) for t in time_idx]
            p_stages = [string(stages[p, t]) for t in time_idx]

            push!(p_entries, """{
                "id": $(ids[p]),
                "status": "$(statuses[p])",
                "alive": $(alive[p]),
                "settlement_age_days": $(round(ages[p] / 86400.0, digits = 2)),
                "lons": [$(join(p_lons, ","))],
                "lats": [$(join(p_lats, ","))],
                "depths": [$(join(p_depths, ","))],
                "temps": [$(join(p_temps, ","))],
                "dds": [$(join(p_dds, ","))],
                "survs": [$(join(p_survs, ","))],
                "stages": [$(join(["\"$s\"" for s in p_stages], ","))]
            }""")
        end
        particles_json = "[" * join(p_entries, ",\n") * "]"

        # Statistics
        n_settled = count(==("settled_successful"), statuses)
        n_alive = count(identity, alive)
        settle_pct = n_p > 0 ? round(100.0 * n_settled / n_p, digits = 1) : 0.0
        survival_pct = n_p > 0 ? round(100.0 * n_alive / n_p, digits = 1) : 0.0
        mean_pld = round(mean([a / 86400.0 for a in ages]), digits = 1)
        mean_temp_val = round(mean(temps), digits = 2)

        # Gridded settlement density, velocity quivers, and thermal field
        g_disp = run_data.gridded_dispersal
        dens_entries = String[]
        quiver_entries = String[]
        therm_entries = String[]

        if !isnothing(g_disp) && hasproperty(g_disp, :lon_centers) && hasproperty(g_disp, :lat_centers)
            lons_c = g_disp.lon_centers
            lats_c = g_disp.lat_centers
            nx_c = length(lons_c)
            ny_c = length(lats_c)
            step_stride = max(1, round(Int, nx_c / 25))

            has_dens = hasproperty(g_disp, :density) || hasproperty(g_disp, :settlement_density)
            dens_mat = hasproperty(g_disp, :density) ? g_disp.density : (hasproperty(g_disp, :settlement_density) ? g_disp.settlement_density : nothing)

            has_u = hasproperty(g_disp, :u_mean)
            has_v = hasproperty(g_disp, :v_mean)
            has_diff = hasproperty(g_disp, :diffusivity)
            has_t = hasproperty(g_disp, :mean_exposure_temperature)
            has_dd = hasproperty(g_disp, :mean_degree_days)

            for i in 1:step_stride:nx_c, j in 1:step_stride:ny_c
                lon_val = round(lons_c[i], digits = 4)
                lat_val = round(lats_c[j], digits = 4)

                if !isnothing(dens_mat) && !isnan(dens_mat[i, j]) && dens_mat[i, j] > 1e-4
                    d_val = round(dens_mat[i, j], digits = 5)
                    push!(dens_entries, "{\"lon\":$(lon_val),\"lat\":$(lat_val),\"density\":$(d_val)}")
                end

                if has_u && has_v && !isnan(g_disp.u_mean[i, j]) && !isnan(g_disp.v_mean[i, j])
                    u_cm = round(g_disp.u_mean[i, j] * 100.0, digits = 2)
                    v_cm = round(g_disp.v_mean[i, j] * 100.0, digits = 2)
                    spd = round(hypot(u_cm, v_cm), digits = 2)
                    diff_val = has_diff && !isnan(g_disp.diffusivity[i, j]) ? round(g_disp.diffusivity[i, j], digits = 1) : 10.0
                    if spd > 0.1
                        push!(quiver_entries, "{\"lon\":$(lon_val),\"lat\":$(lat_val),\"u\":$(u_cm),\"v\":$(v_cm),\"speed\":$(spd),\"diff\":$(diff_val)}")
                    end
                end

                if has_t && !isnan(g_disp.mean_exposure_temperature[i, j])
                    t_val = round(g_disp.mean_exposure_temperature[i, j], digits = 2)
                    dd_val = has_dd && !isnan(g_disp.mean_degree_days[i, j]) ? round(g_disp.mean_degree_days[i, j], digits = 1) : 40.0
                    push!(therm_entries, "{\"lon\":$(lon_val),\"lat\":$(lat_val),\"temp\":$(t_val),\"dd\":$(dd_val)}")
                end
            end
        end

        # Demographic connectivity flows
        conn = run_data.connectivity
        flow_entries = String[]
        if !isnothing(conn) && hasproperty(conn, :matrix) && hasproperty(conn, :strata_names)
            mat = conn.matrix
            s_names = conn.strata_names
            n_st = length(s_names)

            c_map = Dict(s.name => (s.centroid_lat, s.centroid_lon) for s in strata_list)

            for i in 1:n_st, j in 1:n_st
                p_ij = Float64(mat[i, j])
                if p_ij > 0.005
                    src_n = s_names[i]
                    dst_n = s_names[j]
                    src_c = get(c_map, src_n, (44.0, -63.0))
                    dst_c = get(c_map, dst_n, (44.0, -63.0))
                    push!(flow_entries, """{
                        "src": "$(src_n)",
                        "dst": "$(dst_n)",
                        "src_lat": $(round(src_c[1], digits=4)),
                        "src_lon": $(round(src_c[2], digits=4)),
                        "dst_lat": $(round(dst_c[1], digits=4)),
                        "dst_lon": $(round(dst_c[2], digits=4)),
                        "prob": $(round(p_ij, digits=3))
                    }""")
                end
            end
        end

        # 4. Hydrodynamic Model Outputs (Advection, Temperature, Salinity, Elevation, Upwelling, Bathymetry)
        hydro_source = hasproperty(run_data, :hydrodynamics) && !isnothing(run_data.hydrodynamics) ?
                       run_data.hydrodynamics :
                       (hasproperty(run_data, :model) && !isnothing(run_data.model) ? run_data.model : hydrodynamics)

        min_lon_p = minimum(lons) - 0.4
        max_lon_p = maximum(lons) + 0.4
        min_lat_p = minimum(lats) - 0.3
        max_lat_p = maximum(lats) + 0.3
        d_lon = (min_lon_p, max_lon_p)
        d_lat = (min_lat_p, max_lat_p)

        hydro_data = extract_hydrodynamic_dataset(
            hydro_source,
            domain_lon = d_lon,
            domain_lat = d_lat,
            grid_size = (35, 30)
        )

        h_lons = hydro_data.lons
        h_lats = hydro_data.lats
        h_depths = hydro_data.depths
        nx_h = length(h_lons)
        ny_h = length(h_lats)

        nz_h = length(h_depths)
        stride_h = max(1, round(Int, nx_h / 20))
        # The surface layer: the level nearest z = 0. The store's depth axis runs
        # deepest first, so index 1 is the deepest layer -- using it as "the surface"
        # opened the map on the abyss while calling it the surface.
        k_surface = argmin(abs.(h_depths))
        has_hydro_bathy = !isnothing(hydro_data.bathymetry) &&
                          ndims(hydro_data.bathymetry) == 2 &&
                          size(hydro_data.bathymetry) == (nx_h, ny_h)

        # Build depth_levels array for multi-depth interaction
        depth_levels_entries = String[]
        h_vec_entries_surface = String[]

        for k_idx in 1:nz_h
            vec_entries_k = String[]
            for i in 1:stride_h:nx_h, j in 1:stride_h:ny_h
                lon_v = round(h_lons[i], digits = 4)
                lat_v = round(h_lats[j], digits = 4)
                u_val = round(hydro_data.u[i, j, k_idx] * 100.0, digits = 2)
                v_val = round(hydro_data.v[i, j, k_idx] * 100.0, digits = 2)
                spd_val = round(hydro_data.speed[i, j, k_idx] * 100.0, digits = 2)
                w_val = round(hydro_data.w[i, j, k_idx] * 1000.0, digits = 3)
                t_val = round(hydro_data.temperature[i, j, k_idx], digits = 2)
                s_val = round(hydro_data.salinity[i, j, k_idx], digits = 2)
                strat_val = round(hydro_data.stratification[i, j, k_idx] * 10000.0, digits = 3)
                diff_val = round(hydro_data.diffusion[i, j, k_idx] * 10000.0, digits = 3)
                eta_val = round(hydro_data.elevation[i, j] * 100.0, digits = 1)
                # Seabed depth is optional metadata; emit NaN when the source record
                # carries no bathymetry so the Leaflet depth control is skipped rather
                # than fabricated from an invented flat plane.
                b_val = has_hydro_bathy ? round(hydro_data.bathymetry[i, j], digits = 1) : "null"
                dir_deg = round(mod(90.0 - rad2deg(atan(hydro_data.v[i, j, k_idx], hydro_data.u[i, j, k_idx])), 360.0), digits = 1)

                v_item = """{"lon":$(lon_v),"lat":$(lat_v),"u":$(u_val),"v":$(v_val),"speed":$(spd_val),"dir":$(dir_deg),"w":$(w_val),"temp":$(t_val),"sal":$(s_val),"strat":$(strat_val),"diff":$(diff_val),"eta":$(eta_val),"depth":$(b_val)}"""
                push!(vec_entries_k, v_item)
                if k_idx == k_surface
                    push!(h_vec_entries_surface, v_item)
                end
            end

            t_k_json = "[" * join(["[" * join([string(round(hydro_data.temperature[i, j, k_idx], digits = 2)) for j in 1:ny_h], ",") * "]" for i in 1:nx_h], ",") * "]"
            s_k_json = "[" * join(["[" * join([string(round(hydro_data.salinity[i, j, k_idx], digits = 2)) for j in 1:ny_h], ",") * "]" for i in 1:nx_h], ",") * "]"
            spd_k_json = "[" * join(["[" * join([string(round(hydro_data.speed[i, j, k_idx] * 100.0, digits = 2)) for j in 1:ny_h], ",") * "]" for i in 1:nx_h], ",") * "]"
            w_k_json = "[" * join(["[" * join([string(round(hydro_data.w[i, j, k_idx] * 1000.0, digits = 3)) for j in 1:ny_h], ",") * "]" for i in 1:nx_h], ",") * "]"
            strat_k_json = "[" * join(["[" * join([string(round(hydro_data.stratification[i, j, k_idx] * 10000.0, digits = 3)) for j in 1:ny_h], ",") * "]" for i in 1:nx_h], ",") * "]"
            diff_k_json = "[" * join(["[" * join([string(round(hydro_data.diffusion[i, j, k_idx] * 10000.0, digits = 3)) for j in 1:ny_h], ",") * "]" for i in 1:nx_h], ",") * "]"

            valid_t = filter(!isnan, hydro_data.temperature[:, :, k_idx])
            valid_s = filter(!isnan, hydro_data.salinity[:, :, k_idx])
            valid_spd = filter(!isnan, hydro_data.speed[:, :, k_idx] .* 100.0)
            valid_w = filter(!isnan, hydro_data.w[:, :, k_idx] .* 1000.0)
            valid_strat = filter(!isnan, hydro_data.stratification[:, :, k_idx] .* 10000.0)
            valid_diff = filter(!isnan, hydro_data.diffusion[:, :, k_idx] .* 10000.0)

            t_min_k, t_max_k = isempty(valid_t) ? (0.0, 15.0) : (round(minimum(valid_t), digits=2), round(maximum(valid_t), digits=2))
            s_min_k, s_max_k = isempty(valid_s) ? (30.0, 35.0) : (round(minimum(valid_s), digits=2), round(maximum(valid_s), digits=2))
            spd_max_k = isempty(valid_spd) ? 25.0 : round(maximum(valid_spd), digits=1)
            w_min_k, w_max_k = isempty(valid_w) ? (-0.5, 0.5) : (round(minimum(valid_w), digits=2), round(maximum(valid_w), digits=2))
            strat_min_k, strat_max_k = isempty(valid_strat) ? (0.0, 5.0) : (round(minimum(valid_strat), digits=2), round(maximum(valid_strat), digits=2))
            diff_min_k, diff_max_k = isempty(valid_diff) ? (0.1, 10.0) : (round(minimum(valid_diff), digits=2), round(maximum(valid_diff), digits=2))

            push!(depth_levels_entries, """{
                "depth_m": $(round(h_depths[k_idx], digits=1)),
                "vectors": [$(join(vec_entries_k, ","))],
                "temperature": {"min": $(t_min_k), "max": $(t_max_k), "unit": "°C", "data": $(t_k_json)},
                "salinity": {"min": $(s_min_k), "max": $(s_max_k), "unit": "PSU", "data": $(s_k_json)},
                "speed": {"min": 0.0, "max": $(spd_max_k), "unit": "cm/s", "data": $(spd_k_json)},
                "upwelling": {"min": $(w_min_k), "max": $(w_max_k), "unit": "mm/s", "data": $(w_k_json)},
                "stratification": {"min": $(strat_min_k), "max": $(strat_max_k), "unit": "10⁻⁴ s⁻²", "data": $(strat_k_json)},
                "diffusion": {"min": $(diff_min_k), "max": $(diff_max_k), "unit": "10⁻⁴ m²/s", "data": $(diff_k_json)}
            }""")
        end

        eta_grid_json = "[" * join(["[" * join([string(round(hydro_data.elevation[i, j] * 100.0, digits = 1)) for j in 1:ny_h], ",") * "]" for i in 1:nx_h], ",") * "]"
        # Bathymetry grid is optional: emit JSON nulls and a degenerate colour range when
        # the source record carries no seabed, so the map renders without a depth layer
        # instead of indexing a missing array.
        bathy_grid_json = has_hydro_bathy ?
            ("[" * join(["[" * join([string(round(hydro_data.bathymetry[i, j], digits = 1)) for j in 1:ny_h], ",") * "]" for i in 1:nx_h], ",") * "]") :
            ("[" * join(["[" * join(["null" for _ in 1:ny_h], ",") * "]" for _ in 1:nx_h], ",") * "]")

        valid_eta = filter(!isnan, hydro_data.elevation .* 100.0)
        valid_b = has_hydro_bathy ? filter(isfinite, hydro_data.bathymetry) : Float64[]
        eta_min = isempty(valid_eta) ? -10.0 : round(minimum(valid_eta), digits=1)
        eta_max = isempty(valid_eta) ? 10.0 : round(maximum(valid_eta), digits=1)
        b_min = isempty(valid_b) ? -3000.0 : round(minimum(valid_b), digits=1)
        b_max = isempty(valid_b) ? 0.0 : round(maximum(valid_b), digits=1)

        # Surface grids for backwards compatibility
        t_min_s = isempty(filter(!isnan, hydro_data.temperature[:, :, k_surface])) ? 0.0 : round(minimum(filter(!isnan, hydro_data.temperature[:, :, k_surface])), digits=2)
        t_max_s = isempty(filter(!isnan, hydro_data.temperature[:, :, k_surface])) ? 15.0 : round(maximum(filter(!isnan, hydro_data.temperature[:, :, k_surface])), digits=2)
        s_min_s = isempty(filter(!isnan, hydro_data.salinity[:, :, k_surface])) ? 30.0 : round(minimum(filter(!isnan, hydro_data.salinity[:, :, k_surface])), digits=2)
        s_max_s = isempty(filter(!isnan, hydro_data.salinity[:, :, k_surface])) ? 35.5 : round(maximum(filter(!isnan, hydro_data.salinity[:, :, k_surface])), digits=2)
        spd_max_s = isempty(filter(!isnan, hydro_data.speed[:, :, k_surface] .* 100.0)) ? 25.0 : round(maximum(filter(!isnan, hydro_data.speed[:, :, k_surface] .* 100.0)), digits=1)
        w_min_s = isempty(filter(!isnan, hydro_data.w[:, :, k_surface] .* 1000.0)) ? -0.5 : round(minimum(filter(!isnan, hydro_data.w[:, :, k_surface] .* 1000.0)), digits=2)
        w_max_s = isempty(filter(!isnan, hydro_data.w[:, :, k_surface] .* 1000.0)) ? 0.5 : round(maximum(filter(!isnan, hydro_data.w[:, :, k_surface] .* 1000.0)), digits=2)

        temp_s_json = "[" * join(["[" * join([string(round(hydro_data.temperature[i, j, k_surface], digits = 2)) for j in 1:ny_h], ",") * "]" for i in 1:nx_h], ",") * "]"
        sal_s_json = "[" * join(["[" * join([string(round(hydro_data.salinity[i, j, k_surface], digits = 2)) for j in 1:ny_h], ",") * "]" for i in 1:nx_h], ",") * "]"
        spd_s_json = "[" * join(["[" * join([string(round(hydro_data.speed[i, j, k_surface] * 100.0, digits = 2)) for j in 1:ny_h], ",") * "]" for i in 1:nx_h], ",") * "]"
        w_s_json = "[" * join(["[" * join([string(round(hydro_data.w[i, j, k_surface] * 1000.0, digits = 3)) for j in 1:ny_h], ",") * "]" for i in 1:nx_h], ",") * "]"

        hydro_json = """{
            "lons": [$(join([string(round(x, digits=4)) for x in h_lons], ","))],
            "lats": [$(join([string(round(y, digits=4)) for y in h_lats], ","))],
            "depths": [$(join([string(round(z, digits=1)) for z in h_depths], ","))],
            "depth_levels": [$(join(depth_levels_entries, ","))],
            "vectors": [$(join(h_vec_entries_surface, ","))],
            "grids": {
                "temperature": {"min": $(t_min_s), "max": $(t_max_s), "unit": "°C", "data": $(temp_s_json)},
                "salinity": {"min": $(s_min_s), "max": $(s_max_s), "unit": "PSU", "data": $(sal_s_json)},
                "speed": {"min": 0.0, "max": $(spd_max_s), "unit": "cm/s", "data": $(spd_s_json)},
                "elevation": {"min": $(eta_min), "max": $(eta_max), "unit": "cm", "data": $(eta_grid_json)},
                "upwelling": {"min": $(w_min_s), "max": $(w_max_s), "unit": "mm/s", "data": $(w_s_json)},
                "bathymetry": {"min": $(b_min), "max": $(b_max), "unit": "m", "data": $(bathy_grid_json)}
            }
        }"""

        run_json = """
        "$(run_key)": {
            "name": "$(run_key)",
            "times": [$(join(t_days, ","))],
            "stats": {
                "n_particles": $(n_p),
                "n_settled": $(n_settled),
                "n_alive": $(n_alive),
                "settle_pct": $(settle_pct),
                "survival_pct": $(survival_pct),
                "mean_pld_days": $(mean_pld),
                "mean_temp": $(mean_temp_val)
            },
            "particles": $(particles_json),
            "settlement_density": [$(join(dens_entries, ","))],
            "velocity_quivers": [$(join(quiver_entries, ","))],
            "thermal_field": [$(join(therm_entries, ","))],
            "connectivity_flows": [$(join(flow_entries, ","))],
            "hydrodynamics": $(hydro_json)
        }
        """
        push!(run_json_blocks, run_json)
    end

    runs_master_json = "{\n" * join(run_json_blocks, ",\n") * "\n}"
    default_run_key = first(keys(runs_dict))

    html_content = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>$(title)</title>
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" integrity="sha256-p4NxAoJBhIIN+hmNHrzRCf9tD/miZyoHS5obTRR9BMY=" crossorigin=""/>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-dark: #0b0f19;
            --panel-bg: rgba(15, 23, 42, 0.90);
            --panel-border: rgba(255, 255, 255, 0.12);
            --accent-blue: #38bdf8;
            --accent-cyan: #06b6d4;
            --accent-green: #10b981;
            --accent-orange: #f97316;
            --accent-red: #ef4444;
            --accent-purple: #a855f7;
            --accent-amber: #f59e0b;
            --text-main: #f1f5f9;
            --text-muted: #94a3b8;
            --font-sans: 'Inter', system-ui, -apple-system, sans-serif;
            --font-mono: 'JetBrains Mono', monospace;
        }

        * { box-sizing: border-box; margin: 0; padding: 0; }

        body {
            font-family: var(--font-sans);
            background: var(--bg-dark);
            color: var(--text-main);
            height: 100vh;
            display: flex;
            flex-direction: column;
            overflow: hidden;
        }

        header {
            background: rgba(11, 15, 25, 0.96);
            backdrop-filter: blur(12px);
            border-bottom: 1px solid var(--panel-border);
            padding: 10px 20px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            z-index: 1000;
        }

        .header-title h1 {
            font-size: 1.1rem;
            font-weight: 700;
            color: #fff;
            letter-spacing: -0.02em;
        }

        .header-title h1 span {
            background: linear-gradient(135deg, var(--accent-blue), #818cf8);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        .header-subtitle {
            font-size: 0.76rem;
            color: var(--text-muted);
            margin-top: 2px;
        }

        .header-controls {
            display: flex;
            align-items: center;
            gap: 16px;
        }

        .run-selector-box {
            display: flex;
            align-items: center;
            gap: 8px;
            background: rgba(255, 255, 255, 0.06);
            padding: 4px 10px;
            border-radius: 8px;
            border: 1px solid var(--panel-border);
        }

        .run-selector-box label {
            font-size: 0.72rem;
            text-transform: uppercase;
            font-weight: 600;
            color: var(--accent-blue);
        }

        .run-selector-box select {
            background: #0f172a;
            color: #fff;
            border: 1px solid rgba(255,255,255,0.2);
            border-radius: 6px;
            padding: 4px 8px;
            font-size: 0.82rem;
            font-weight: 500;
            cursor: pointer;
            outline: none;
        }

        .metric-chips {
            display: flex;
            gap: 10px;
            align-items: center;
        }

        .chip {
            background: var(--panel-bg);
            border: 1px solid var(--panel-border);
            border-radius: 8px;
            padding: 4px 10px;
            display: flex;
            flex-direction: column;
            align-items: flex-end;
        }

        .chip-label {
            font-size: 0.62rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: var(--text-muted);
        }

        .chip-value {
            font-size: 0.88rem;
            font-weight: 600;
            font-family: var(--font-mono);
            color: var(--accent-blue);
        }

        .chip-value.success { color: var(--accent-green); }
        .chip-value.warning { color: var(--accent-orange); }

        #workspace {
            position: relative;
            flex: 1;
            display: flex;
            width: 100%;
            height: calc(100vh - 60px);
        }

        #map {
            width: 100%;
            height: 100%;
            background: #080d1a;
        }

        .floating-panel {
            position: absolute;
            background: var(--panel-bg);
            backdrop-filter: blur(16px);
            border: 1px solid var(--panel-border);
            border-radius: 12px;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.45);
            z-index: 1000;
            padding: 14px;
            transition: all 0.2s ease;
        }

        #layer-manager-dock {
            top: 20px;
            right: 20px;
            width: 270px;
            max-height: calc(100vh - 200px);
            overflow-y: auto;
        }

        #layer-manager-dock::-webkit-scrollbar { width: 4px; }
        #layer-manager-dock::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.2); border-radius: 2px; }

        .layer-group-title {
            font-size: 0.72rem;
            text-transform: uppercase;
            color: var(--accent-blue);
            margin: 10px 0 6px 0;
            letter-spacing: 0.06em;
            font-weight: 700;
            border-bottom: 1px solid rgba(255,255,255,0.08);
            padding-bottom: 3px;
        }

        .layer-group-title:first-child { margin-top: 0; }

        .layer-toggle-group {
            display: flex;
            flex-direction: column;
            gap: 6px;
        }

        .layer-item {
            display: flex;
            align-items: center;
            justify-content: space-between;
            font-size: 0.78rem;
            color: var(--text-main);
            padding: 3px 0;
        }

        .layer-item label {
            display: flex;
            align-items: center;
            gap: 6px;
            cursor: pointer;
            user-select: none;
        }

        .layer-item input[type="checkbox"] {
            accent-color: var(--accent-blue);
            cursor: pointer;
        }

        #controls-dock {
            top: 20px;
            left: 20px;
            width: 260px;
            display: flex;
            flex-direction: column;
            gap: 10px;
        }

        .control-section h4 {
            font-size: 0.70rem;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            color: var(--accent-blue);
            margin-bottom: 5px;
        }

        .control-select {
            width: 100%;
            background: rgba(0, 0, 0, 0.4);
            border: 1px solid var(--panel-border);
            color: #fff;
            padding: 5px 8px;
            border-radius: 6px;
            font-size: 0.78rem;
            outline: none;
        }

        .slider-control-row {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 8px;
            font-size: 0.74rem;
            color: var(--text-muted);
        }

        #playback-dock {
            bottom: 24px;
            left: 50%;
            transform: translateX(-50%);
            width: 90%;
            max-width: 720px;
            display: flex;
            flex-direction: column;
            gap: 10px;
        }

        .playback-controls {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 12px;
        }

        .btn-group { display: flex; gap: 6px; }

        .control-btn {
            background: rgba(255, 255, 255, 0.08);
            border: 1px solid var(--panel-border);
            color: #fff;
            padding: 6px 14px;
            border-radius: 6px;
            font-size: 0.82rem;
            font-weight: 500;
            cursor: pointer;
            display: inline-flex;
            align-items: center;
            gap: 6px;
            transition: background 0.15s ease;
        }

        .control-btn:hover { background: rgba(255, 255, 255, 0.18); }
        .control-btn.active {
            background: var(--accent-blue);
            color: #000;
            font-weight: 600;
            border-color: var(--accent-blue);
        }

        .time-display {
            font-family: var(--font-mono);
            font-size: 0.88rem;
            color: #fff;
            background: rgba(0, 0, 0, 0.35);
            padding: 4px 10px;
            border-radius: 6px;
            border: 1px solid var(--panel-border);
        }

        .slider-row { display: flex; align-items: center; gap: 12px; }

        input[type="range"] {
            flex: 1;
            accent-color: var(--accent-blue);
            height: 6px;
            border-radius: 3px;
            background: rgba(255, 255, 255, 0.2);
            cursor: pointer;
        }

        #hud-dock {
            bottom: 24px;
            right: 20px;
            width: 260px;
            font-size: 0.78rem;
        }

        .hud-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 6px 10px;
            margin-top: 6px;
        }

        .hud-item { display: flex; flex-direction: column; }
        .hud-label { font-size: 0.62rem; color: var(--text-muted); text-transform: uppercase; }
        .hud-val { font-family: var(--font-mono); font-weight: 600; color: #fff; }

        #legend-dock {
            bottom: 125px;
            left: 20px;
            width: 250px;
            padding: 10px 12px;
            font-size: 0.74rem;
        }

        .legend-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            font-weight: 600;
            color: var(--accent-blue);
            margin-bottom: 6px;
            text-transform: uppercase;
            font-size: 0.68rem;
            letter-spacing: 0.05em;
        }

        .legend-bar {
            height: 12px;
            border-radius: 4px;
            width: 100%;
            margin-bottom: 4px;
            border: 1px solid rgba(255,255,255,0.2);
        }

        .legend-labels {
            display: flex;
            justify-content: space-between;
            font-family: var(--font-mono);
            font-size: 0.68rem;
            color: var(--text-muted);
        }

        .leaflet-tooltip {
            background: rgba(15, 23, 42, 0.94) !important;
            color: #fff !important;
            border: 1px solid rgba(255,255,255,0.25) !important;
            border-radius: 6px !important;
            font-family: var(--font-sans) !important;
            font-size: 0.76rem !important;
            box-shadow: 0 4px 16px rgba(0,0,0,0.5) !important;
            padding: 6px 10px !important;
            line-height: 1.35 !important;
        }
    </style>
</head>
<body>
    <header>
        <div class="header-title">
            <h1>Larval <span>Dispersal</span> & Ocean Hydrodynamics</h1>
            <div class="header-subtitle">Multi-Run Hydrodynamic Particle Tracking Dashboard (*Chionoecetes opilio*)</div>
        </div>
        <div class="header-controls">
            <div class="run-selector-box">
                <label for="run-select">Run:</label>
                <select id="run-select" onchange="switchRun(this.value)"></select>
            </div>
            <div class="metric-chips">
                <div class="chip">
                    <span class="chip-label">Cohort</span>
                    <span class="chip-value" id="chip-cohort">-</span>
                </div>
                <div class="chip">
                    <span class="chip-label">Settled</span>
                    <span class="chip-value success" id="chip-settled">-</span>
                </div>
                <div class="chip">
                    <span class="chip-label">Survival</span>
                    <span class="chip-value" id="chip-survival">-</span>
                </div>
                <div class="chip">
                    <span class="chip-label">Mean PLD</span>
                    <span class="chip-value warning" id="chip-pld">-</span>
                </div>
            </div>
        </div>
    </header>

    <div id="workspace">
        <div id="map"></div>

        <div class="floating-panel" id="controls-dock">
            <div class="control-section">
                <h4>Particle Palette</h4>
                <select id="color-mode-select" class="control-select">
                    <option value="stage">Larval Stage (Molting)</option>
                    <option value="depth">Drift Depth (m)</option>
                    <option value="temp">Ambient Temperature (°C)</option>
                    <option value="dd">Cumulative Degree-Days</option>
                </select>
            </div>

            <div class="control-section">
                <h4>Larval Cohort Filter</h4>
                <select id="outcome-filter-select" class="control-select">
                    <option value="all">All Particles</option>
                    <option value="settled">Settled Successfully</option>
                    <option value="pelagic">Remaining Pelagic</option>
                    <option value="dead">Thermal Mortality (Dead)</option>
                </select>
            </div>

            <div class="control-section">
                <h4>Focus Particle</h4>
                <select id="particle-select" class="control-select">
                    <option value="-1">Overview (All Larvae)</option>
                </select>
            </div>

            <div class="control-section">
                <h4>Hydrodynamic Depth Level</h4>
                <select id="hydro-depth-select" class="control-select">
                    <option value="0">Surface (0.0 m)</option>
                </select>
            </div>

            <div class="control-section">
                <h4>Hydrodynamic Layer Opacity</h4>
                <div class="slider-control-row">
                    <span>Alpha</span>
                    <input type="range" id="hydro-opacity-slider" min="10" max="100" value="65">
                    <span id="hydro-opacity-val" style="font-family:var(--font-mono)">65%</span>
                </div>
            </div>
        </div>

        <div class="floating-panel" id="layer-manager-dock">
            <div class="layer-group-title">🛰️ Lagrangian Tracking</div>
            <div class="layer-toggle-group">
                <div class="layer-item">
                    <label><input type="checkbox" id="layer-fulltracks" checked> 🛰️ Full Trajectories</label>
                </div>
                <div class="layer-item">
                    <label><input type="checkbox" id="layer-progtrails" checked> ✨ Progressive Trails</label>
                </div>
                <div class="layer-item">
                    <label><input type="checkbox" id="layer-heads" checked> 🔵 Active Larval Heads</label>
                </div>
                <div class="layer-item">
                    <label><input type="checkbox" id="layer-release" checked> 🟢 Spawning Releases</label>
                </div>
                <div class="layer-item">
                    <label><input type="checkbox" id="layer-settle" checked> 🔴 Settlement Sites</label>
                </div>
                <div class="layer-item">
                    <label><input type="checkbox" id="layer-density"> 🔥 Nursery Settlement Density</label>
                </div>
            </div>

            <div class="layer-group-title">🌊 Hydrodynamic Model Outputs</div>
            <div class="layer-toggle-group">
                <div class="layer-item">
                    <label><input type="checkbox" id="layer-hydro-advection" checked> 💨 Advection Current Vectors</label>
                </div>
                <div class="layer-item">
                    <label><input type="checkbox" id="layer-hydro-speed"> ⚡ Current Speed (|u_h|)</label>
                </div>
                <div class="layer-item">
                    <label><input type="checkbox" id="layer-hydro-temp"> 🌡️ Seawater Temperature (T)</label>
                </div>
                <div class="layer-item">
                    <label><input type="checkbox" id="layer-hydro-sal"> 🧂 Practical Salinity (S)</label>
                </div>
                <div class="layer-item">
                    <label><input type="checkbox" id="layer-hydro-strat"> 📈 Buoyancy Stratification (N²)</label>
                </div>
                <div class="layer-item">
                    <label><input type="checkbox" id="layer-hydro-diff"> 🌀 Turbulent Diffusivity (κ_v)</label>
                </div>
                <div class="layer-item">
                    <label><input type="checkbox" id="layer-hydro-eta"> 🌊 Free Surface Height (η)</label>
                </div>
                <div class="layer-item">
                    <label><input type="checkbox" id="layer-hydro-w"> ⬆️ Upwelling / Downwelling (w)</label>
                </div>
                <div class="layer-item">
                    <label><input type="checkbox" id="layer-hydro-bathy"> 🗺️ Seafloor Bathymetry</label>
                </div>
            </div>

            <div class="layer-group-title">🏛️ Demographics & Zones</div>
            <div class="layer-toggle-group">
                <div class="layer-item">
                    <label><input type="checkbox" id="layer-strata" checked> 🗺️ CFA Management Zones</label>
                </div>
                <div class="layer-item">
                    <label><input type="checkbox" id="layer-conn"> ↔️ CFA Connectivity Flows</label>
                </div>
            </div>
        </div>

        <div class="floating-panel" id="playback-dock">
            <div class="playback-controls">
                <div class="btn-group">
                    <button class="control-btn" id="play-btn">▶ Play</button>
                    <button class="control-btn" id="reset-btn">⏮ Reset</button>
                    <button class="control-btn speed-btn active" data-speed="1">1x</button>
                    <button class="control-btn speed-btn" data-speed="2">2x</button>
                    <button class="control-btn speed-btn" data-speed="4">4x</button>
                </div>
                <div class="time-display" id="time-readout">Day 0.00 / 0.00 d</div>
            </div>
            <div class="slider-row">
                <input type="range" id="time-slider" min="0" max="0" value="0" step="1">
            </div>
        </div>

        <div class="floating-panel" id="hud-dock">
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <b style="color:var(--accent-blue);" id="telemetry-hud">Telemetry HUD</b>
                <span id="hud-id" style="font-family:var(--font-mono); color:#fff;">Overview</span>
            </div>
            <div class="hud-grid">
                <div class="hud-item"><span class="hud-label">Stage / Mode</span><span class="hud-val" id="hud-stage">-</span></div>
                <div class="hud-item"><span class="hud-label">Depth / Seabed</span><span class="hud-val" id="hud-depth">-</span></div>
                <div class="hud-item"><span class="hud-label">Temp (T)</span><span class="hud-val" id="hud-temp">-</span></div>
                <div class="hud-item"><span class="hud-label">Salinity (S)</span><span class="hud-val" id="hud-sal">-</span></div>
                <div class="hud-item"><span class="hud-label">Current (|u|)</span><span class="hud-val" id="hud-current">-</span></div>
                <div class="hud-item"><span class="hud-label">Strat (N²)</span><span class="hud-val" id="hud-strat">-</span></div>
                <div class="hud-item"><span class="hud-label">Diffusivity (κ)</span><span class="hud-val" id="hud-diff">-</span></div>
                <div class="hud-item"><span class="hud-label">Survival / SSH</span><span class="hud-val" id="hud-surv">-</span></div>
            </div>
        </div>

        <div class="floating-panel" id="legend-dock">
            <div class="legend-header">
                <span id="legend-title">Seawater Temperature</span>
                <span id="legend-unit" style="font-family:var(--font-mono)">°C</span>
            </div>
            <div class="legend-bar" id="legend-bar" style="background: linear-gradient(to right, #1d4ed8, #06b6d4, #10b981, #f59e0b, #ef4444);"></div>
            <div class="legend-labels">
                <span id="legend-min">0.0</span>
                <span id="legend-mid">7.5</span>
                <span id="legend-max">15.0</span>
            </div>
        </div>
    </div>

    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js" integrity="sha256-20nQCchB9co0qIjJZRGuk2/Z9VM+kNiyxNV1lvTlZBo=" crossorigin=""></script>
    <script>
        const RUNS = $(runs_master_json);
        const STRATA = $(strata_json);
        let activeRunKey = "$(default_run_key)";
        let currentRun = RUNS[activeRunKey];
        const PARTICLES = currentRun.particles;

        const STAGE_COLORS = {
            "zoea1": "#38bdf8", "zoea2": "#06b6d4", "zoea3": "#0d9488", "zoea4": "#eab308",
            "megalopa": "#f97316", "instar1_settled": "#10b981", "dead": "#ef4444"
        };

        const map = L.map('map', { center: [44.2, -62.5], zoom: 7 });
        const esriOcean = L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/Ocean/World_Ocean_Base/MapServer/tile/{z}/{y}/{x}', { maxZoom: 13 }).addTo(map);
        const cartoDark = L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', { maxZoom: 19 });

        L.control.layers({ "ESRI Ocean": esriOcean, "CartoDB Dark": cartoDark }, null, { position: 'bottomleft' }).addTo(map);

        // Management & Lagrangian Layers
        const strataLayer = L.layerGroup().addTo(map);
        const fullTracksLayer = L.layerGroup().addTo(map);
        const progressTracksLayer = L.layerGroup().addTo(map);
        const activeHeadsLayer = L.layerGroup().addTo(map);
        const releaseMarkersLayer = L.layerGroup().addTo(map);
        const settlementMarkersLayer = L.layerGroup().addTo(map);
        const settlementDensityLayer = L.layerGroup();
        const connectivityFlowsLayer = L.layerGroup();

        // Hydrodynamic Model Eulerian Layers
        const hydroAdvectionLayer = L.layerGroup().addTo(map);
        const hydroSpeedLayer = L.layerGroup();
        const hydroTempLayer = L.layerGroup();
        const hydroSalLayer = L.layerGroup();
        const hydroStratLayer = L.layerGroup();
        const hydroDiffLayer = L.layerGroup();
        const hydroEtaLayer = L.layerGroup();
        const hydroUpwellingLayer = L.layerGroup();
        const hydroBathyLayer = L.layerGroup();
        let selectedDepthIdx = 0;

        STRATA.forEach(s => {
            let layer = s.is_polygon ? L.polygon(s.polygon, {color: s.color, weight: 2.2, fillOpacity: 0.08}) : L.rectangle([[s.lat_min, s.lon_min], [s.lat_max, s.lon_max]], {color: s.color, weight: 2, fillOpacity: 0.06});
            layer.bindTooltip("<b>" + s.name + "</b>").addTo(strataLayer);
        });

        const runSelect = document.getElementById('run-select');
        Object.keys(RUNS).forEach(k => {
            const opt = document.createElement('option'); opt.value = k; opt.textContent = k;
            if (k === activeRunKey) opt.selected = true;
            runSelect.appendChild(opt);
        });

        let currentStep = 0, isPlaying = false, playSpeed = 1, animTimer = null, showTrails = true, colorMode = "stage", outcomeFilter = "all", focusParticleId = -1;
        let hydroOpacity = 0.65;
        let headMarkers = [], fullTrackPolylines = [], progressTrackPolylines = [];

        // Scientific Colormaps
        function colormapThermal(norm) {
            if (norm < 0.25) {
                const t = norm / 0.25;
                return [Math.round(29 + t * (6 - 29)), Math.round(78 + t * (182 - 78)), Math.round(216 + t * (212 - 216)), 255];
            } else if (norm < 0.5) {
                const t = (norm - 0.25) / 0.25;
                return [Math.round(6 + t * (16 - 6)), Math.round(182 + t * (185 - 182)), Math.round(212 + t * (129 - 212)), 255];
            } else if (norm < 0.75) {
                const t = (norm - 0.5) / 0.25;
                return [Math.round(16 + t * (245 - 16)), Math.round(185 + t * (158 - 185)), Math.round(129 + t * (11 - 129)), 255];
            } else {
                const t = (norm - 0.75) / 0.25;
                return [Math.round(245 + t * (239 - 245)), Math.round(158 + t * (68 - 158)), Math.round(11 + t * (68 - 11)), 255];
            }
        }

        function colormapHaline(norm) {
            const r = Math.round(15 + norm * 150);
            const g = Math.round(180 - norm * 110);
            const b = Math.round(180 + norm * 60);
            return [r, g, b, 255];
        }

        function colormapTurbo(norm) {
            const r = Math.round(255 * Math.min(1.0, Math.max(0.0, 1.5 - Math.abs(norm - 0.75) * 4)));
            const g = Math.round(255 * Math.min(1.0, Math.max(0.0, 1.5 - Math.abs(norm - 0.5) * 4)));
            const b = Math.round(255 * Math.min(1.0, Math.max(0.0, 1.5 - Math.abs(norm - 0.25) * 4)));
            return [r, g, b, 255];
        }

        function colormapCoolwarm(norm) {
            const r = Math.round(59 + norm * (220 - 59));
            const g = Math.round(76 + (1 - Math.abs(norm - 0.5) * 2) * 120);
            const b = Math.round(192 - norm * (192 - 40));
            return [r, g, b, 255];
        }

        function colormapBalance(norm) {
            if (norm < 0.5) {
                const t = norm / 0.5;
                return [Math.round(30 + t * 160), Math.round(64 + t * 130), Math.round(175 + t * 20), 255];
            } else {
                const t = (norm - 0.5) / 0.5;
                return [Math.round(190 + t * 49), Math.round(194 - t * 140), Math.round(195 - t * 140), 255];
            }
        }

        function colormapOcean(norm) {
            const r = Math.round(10 + norm * 80);
            const g = Math.round(40 + norm * 140);
            const b = Math.round(90 + norm * 130);
            return [r, g, b, 255];
        }

        function colormapViridis(norm) {
            const r = Math.round(68 + norm * 185);
            const g = Math.round(1 + norm * 230);
            const b = Math.round(84 + (1 - norm) * 110 + norm * 15);
            return [Math.min(255, Math.max(0, r)), Math.min(255, Math.max(0, g)), Math.min(255, Math.max(0, b)), 255];
        }

        function colormapPlasma(norm) {
            const r = Math.round(13 + norm * 225);
            const g = Math.round(8 + Math.sin(norm * Math.PI) * 180);
            const b = Math.round(135 * (1 - norm) + 30);
            return [Math.min(255, Math.max(0, r)), Math.min(255, Math.max(0, g)), Math.min(255, Math.max(0, b)), 255];
        }

        function generateRasterDataURL(grid2D, lons, lats, minVal, maxVal, colormapFn) {
            const width = 200;
            const height = 150;
            const canvas = document.createElement('canvas');
            canvas.width = width;
            canvas.height = height;
            const ctx = canvas.getContext('2d');
            const imgData = ctx.createImageData(width, height);
            const data = imgData.data;

            const nx = lons.length;
            const ny = lats.length;
            const range = Math.max(1e-5, maxVal - minVal);

            for (let py = 0; py < height; py++) {
                const latFrac = 1.0 - (py / (height - 1));
                const yIdx = latFrac * (ny - 1);
                const j0 = Math.floor(yIdx);
                const j1 = Math.min(ny - 1, j0 + 1);
                const fy = yIdx - j0;

                for (let px = 0; px < width; px++) {
                    const lonFrac = px / (width - 1);
                    const xIdx = lonFrac * (nx - 1);
                    const i0 = Math.floor(xIdx);
                    const i1 = Math.min(nx - 1, i0 + 1);
                    const fx = xIdx - i0;

                    const v00 = grid2D[i0][j0];
                    const v10 = grid2D[i1][j0];
                    const v01 = grid2D[i0][j1];
                    const v11 = grid2D[i1][j1];

                    const val = (1 - fx) * (1 - fy) * v00 + fx * (1 - fy) * v10 + (1 - fx) * fy * v01 + fx * fy * v11;
                    const norm = Math.min(1.0, Math.max(0.0, (val - minVal) / range));
                    const rgba = colormapFn(norm);

                    const pIndex = (py * width + px) * 4;
                    data[pIndex] = rgba[0];
                    data[pIndex + 1] = rgba[1];
                    data[pIndex + 2] = rgba[2];
                    data[pIndex + 3] = rgba[3];
                }
            }
            ctx.putImageData(imgData, 0, 0);
            return canvas.toDataURL();
        }

        function getColor(p, step) {
            if (colorMode === "stage") return STAGE_COLORS[p.stages[step]] || "#38bdf8";
            if (colorMode === "depth") return Math.abs(p.depths[step]) < 50 ? "#38bdf8" : "#1e3a8a";
            if (colorMode === "temp") return p.temps[step] < 6 ? "#38bdf8" : "#f87171";
            return p.dds[step] < 100 ? "#38bdf8" : "#f97316";
        }

        let activeRasterLayers = [];

        function updateLegend(title, unit, minV, maxV, gradientCss) {
            document.getElementById('legend-title').textContent = title;
            document.getElementById('legend-unit').textContent = unit;
            document.getElementById('legend-min').textContent = minV;
            document.getElementById('legend-max').textContent = maxV;
            document.getElementById('legend-mid').textContent = ((parseFloat(minV) + parseFloat(maxV)) / 2.0).toFixed(1);
            document.getElementById('legend-bar').style.background = gradientCss;
        }

        function rebuildHydroLayers() {
            [hydroAdvectionLayer, hydroSpeedLayer, hydroTempLayer, hydroSalLayer,
             hydroStratLayer, hydroDiffLayer, hydroUpwellingLayer].forEach(l => l.clearLayers());

            const hydro = currentRun.hydrodynamics;
            if (!hydro || !hydro.lons || !hydro.lats) return;

            const hLons = hydro.lons;
            const hLats = hydro.lats;
            const bounds = [[Math.min(...hLats), Math.min(...hLons)], [Math.max(...hLats), Math.max(...hLons)]];

            const lvl = (hydro.depth_levels && hydro.depth_levels[selectedDepthIdx]) ? hydro.depth_levels[selectedDepthIdx] : null;
            const vecs = lvl ? lvl.vectors : (hydro.vectors || []);
            const curDepth = lvl ? Math.abs(lvl.depth_m).toFixed(1) : "0.0";

            // A. Advection Current Vectors (Quivers)
            if (vecs && vecs.length > 0) {
                vecs.forEach(vec => {
                    const lenScale = 0.0035 * Math.min(18.0, Math.max(1.0, vec.speed));
                    const lat2 = vec.lat + (vec.v / Math.max(0.1, vec.speed)) * lenScale;
                    const lon2 = vec.lon + (vec.u / Math.max(0.1, vec.speed)) * lenScale * 1.35;

                    const spdNorm = Math.min(1.0, vec.speed / 20.0);
                    const arrowCol = spdNorm > 0.6 ? '#f43f5e' : spdNorm > 0.3 ? '#38bdf8' : '#34d399';

                    const line = L.polyline([[vec.lat, vec.lon], [lat2, lon2]], {
                        color: arrowCol,
                        weight: 2.2,
                        opacity: 0.85
                    }).bindTooltip(`<b>Hydrodynamic Current Vector</b><br>Depth: <b>\${curDepth} m</b><br>Speed: <b>\${vec.speed} cm/s</b> (\${(vec.speed*0.0194).toFixed(2)} kts)<br>Dir: \${vec.dir}°<br>u: \${vec.u} cm/s, v: \${vec.v} cm/s<br>Temp: \${vec.temp} °C | Salinity: \${vec.sal} PSU<br>Strat N²: \${vec.strat !== undefined ? vec.strat : '-'} 10⁻⁴s⁻²<br>Diff κ_v: \${vec.diff !== undefined ? vec.diff : '-'} 10⁻⁴m²/s<br>Seabed Depth: \${vec.depth} m`);
                    hydroAdvectionLayer.addLayer(line);

                    const tip = L.circleMarker([lat2, lon2], {
                        radius: 2.5,
                        color: arrowCol,
                        fillColor: arrowCol,
                        fillOpacity: 1.0,
                        weight: 1
                    });
                    hydroAdvectionLayer.addLayer(tip);
                });
            }

            // B. Multi-depth Grids
            const tGrid = lvl ? lvl.temperature : hydro.grids?.temperature;
            if (tGrid) {
                const tUrl = generateRasterDataURL(tGrid.data, hLons, hLats, tGrid.min, tGrid.max, colormapThermal);
                const tOverlay = L.imageOverlay(tUrl, bounds, { opacity: hydroOpacity, interactive: false });
                hydroTempLayer.addLayer(tOverlay);
                activeRasterLayers.push(tOverlay);
            }

            const sGrid = lvl ? lvl.salinity : hydro.grids?.salinity;
            if (sGrid) {
                const sUrl = generateRasterDataURL(sGrid.data, hLons, hLats, sGrid.min, sGrid.max, colormapHaline);
                const sOverlay = L.imageOverlay(sUrl, bounds, { opacity: hydroOpacity, interactive: false });
                hydroSalLayer.addLayer(sOverlay);
                activeRasterLayers.push(sOverlay);
            }

            const spdGrid = lvl ? lvl.speed : hydro.grids?.speed;
            if (spdGrid) {
                const spdUrl = generateRasterDataURL(spdGrid.data, hLons, hLats, 0.0, spdGrid.max, colormapTurbo);
                const spdOverlay = L.imageOverlay(spdUrl, bounds, { opacity: hydroOpacity, interactive: false });
                hydroSpeedLayer.addLayer(spdOverlay);
                activeRasterLayers.push(spdOverlay);
            }

            const stratGrid = lvl ? lvl.stratification : null;
            if (stratGrid) {
                const stratUrl = generateRasterDataURL(stratGrid.data, hLons, hLats, stratGrid.min, stratGrid.max, colormapViridis);
                const stratOverlay = L.imageOverlay(stratUrl, bounds, { opacity: hydroOpacity, interactive: false });
                hydroStratLayer.addLayer(stratOverlay);
                activeRasterLayers.push(stratOverlay);
            }

            const diffGrid = lvl ? lvl.diffusion : null;
            if (diffGrid) {
                const diffUrl = generateRasterDataURL(diffGrid.data, hLons, hLats, diffGrid.min, diffGrid.max, colormapPlasma);
                const diffOverlay = L.imageOverlay(diffUrl, bounds, { opacity: hydroOpacity, interactive: false });
                hydroDiffLayer.addLayer(diffOverlay);
                activeRasterLayers.push(diffOverlay);
            }

            const wGrid = lvl ? lvl.upwelling : hydro.grids?.upwelling;
            if (wGrid) {
                const wUrl = generateRasterDataURL(wGrid.data, hLons, hLats, wGrid.min, wGrid.max, colormapBalance);
                const wOverlay = L.imageOverlay(wUrl, bounds, { opacity: hydroOpacity, interactive: false });
                hydroUpwellingLayer.addLayer(wOverlay);
                activeRasterLayers.push(wOverlay);
            }
        }

        function renderActiveRun() {
            [fullTracksLayer, progressTracksLayer, activeHeadsLayer, releaseMarkersLayer,
             settlementMarkersLayer, settlementDensityLayer, hydroAdvectionLayer,
             hydroSpeedLayer, hydroTempLayer, hydroSalLayer, hydroStratLayer, hydroDiffLayer,
             hydroEtaLayer, hydroUpwellingLayer, hydroBathyLayer, connectivityFlowsLayer].forEach(l => l.clearLayers());

            headMarkers = [];
            fullTrackPolylines = [];
            progressTrackPolylines = [];
            activeRasterLayers = [];

            const pList = currentRun.particles;
            const nTimes = currentRun.times.length;

            // 1. Summary Chips
            document.getElementById('chip-cohort').textContent = currentRun.stats.n_particles;
            document.getElementById('chip-settled').textContent = `\${currentRun.stats.n_settled} (\${currentRun.stats.settle_pct}%)`;
            document.getElementById('chip-survival').textContent = `\${currentRun.stats.survival_pct}%`;
            document.getElementById('chip-pld').textContent = `\${currentRun.stats.mean_pld_days} d`;

            // 2. Trajectories, Heads & Markers
            pList.forEach((p) => {
                const fullLatLngs = [];
                for (let t = 0; t < nTimes; t++) {
                    fullLatLngs.push([p.lats[t], p.lons[t]]);
                }

                // Full Polyline
                const fullPoly = L.polyline(fullLatLngs, {
                    color: getColor(p, nTimes - 1),
                    weight: 2.2,
                    opacity: 0.55
                }).on('click', () => selectParticle(p.id));
                fullTrackPolylines.push(fullPoly);
                fullTracksLayer.addLayer(fullPoly);

                // Progress Polyline
                const progPoly = L.polyline(fullLatLngs.slice(0, 2), {
                    color: getColor(p, 0),
                    weight: 3.5,
                    opacity: 0.95
                }).on('click', () => selectParticle(p.id));
                progressTrackPolylines.push(progPoly);
                progressTracksLayer.addLayer(progPoly);

                // Head Marker
                const marker = L.circleMarker([p.lats[0], p.lons[0]], {
                    radius: 5.5,
                    color: '#ffffff',
                    weight: 2,
                    fillColor: getColor(p, 0),
                    fillOpacity: 1.0
                }).on('click', () => selectParticle(p.id));
                marker.bindTooltip(`<b>Larva #\${p.id}</b> (\${p.stages[0]})`, { sticky: true });
                headMarkers.push(marker);
                activeHeadsLayer.addLayer(marker);

                // Spawning Release Marker
                const relMarker = L.circleMarker([p.lats[0], p.lons[0]], {
                    radius: 4,
                    color: '#10b981',
                    fillColor: '#059669',
                    fillOpacity: 0.85,
                    weight: 1.5
                }).bindPopup(`<b>Spawning Release #\${p.id}</b><br>Lon: \${p.lons[0].toFixed(3)}°E<br>Lat: \${p.lats[0].toFixed(3)}°N`);
                releaseMarkersLayer.addLayer(relMarker);

                // Settlement Marker (if settled)
                if (p.status === "settled_successful") {
                    const lastT = nTimes - 1;
                    const setMarker = L.circleMarker([p.lats[lastT], p.lons[lastT]], {
                        radius: 6,
                        color: '#f59e0b',
                        fillColor: '#d97706',
                        fillOpacity: 0.9,
                        weight: 2
                    }).bindPopup(`<b>Benthic Settlement Site #\${p.id}</b><br>Age: \${p.settlement_age_days} d<br>Lon: \${p.lons[lastT].toFixed(3)}°E<br>Lat: \${p.lats[lastT].toFixed(3)}°N`);
                    settlementMarkersLayer.addLayer(setMarker);
                }
            });

            // 3. Hydrodynamic Model Outputs Setup
            const hydro = currentRun.hydrodynamics;
            selectedDepthIdx = 0;
            const depthSelect = document.getElementById('hydro-depth-select');
            if (depthSelect) {
                depthSelect.innerHTML = '';
                if (hydro && hydro.depths && hydro.depths.length > 0) {
                    hydro.depths.forEach((d, idx) => {
                        const opt = document.createElement('option');
                        opt.value = idx;
                        opt.textContent = `\${Math.abs(d).toFixed(1)} m \${d === 0 ? "(Surface)" : "(Subsurface)"}`;
                        if (idx === selectedDepthIdx) opt.selected = true;
                        depthSelect.appendChild(opt);
                    });
                } else {
                    const opt = document.createElement('option');
                    opt.value = 0;
                    opt.textContent = 'Surface (0.0 m)';
                    depthSelect.appendChild(opt);
                }
            }

            if (hydro && hydro.lons && hydro.lats) {
                const hLons = hydro.lons;
                const hLats = hydro.lats;
                const bounds = [[Math.min(...hLats), Math.min(...hLons)], [Math.max(...hLats), Math.max(...hLons)]];

                // Static 2D Fields: Elevation and Bathymetry
                const g = hydro.grids;
                if (g) {
                    if (g.elevation) {
                        const etaUrl = generateRasterDataURL(g.elevation.data, hLons, hLats, g.elevation.min, g.elevation.max, colormapCoolwarm);
                        const etaOverlay = L.imageOverlay(etaUrl, bounds, { opacity: hydroOpacity, interactive: false });
                        hydroEtaLayer.addLayer(etaOverlay);
                        activeRasterLayers.push(etaOverlay);
                    }

                    if (g.bathymetry) {
                        const bUrl = generateRasterDataURL(g.bathymetry.data, hLons, hLats, g.bathymetry.min, 0.0, colormapOcean);
                        const bOverlay = L.imageOverlay(bUrl, bounds, { opacity: hydroOpacity, interactive: false });
                        hydroBathyLayer.addLayer(bOverlay);
                        activeRasterLayers.push(bOverlay);
                    }
                }

                // Render dynamic depth-dependent fields
                rebuildHydroLayers();
            }

            // 4. Nursery Settlement Density Heatmap
            if (currentRun.settlement_density && currentRun.settlement_density.length > 0) {
                const maxDens = Math.max(...currentRun.settlement_density.map(d => d.density));
                currentRun.settlement_density.forEach(d => {
                    const norm = maxDens > 0 ? (d.density / maxDens) : 0.5;
                    const col = norm > 0.66 ? '#ef4444' : norm > 0.33 ? '#f59e0b' : '#38bdf8';
                    const rect = L.circle([d.lat, d.lon], {
                        radius: 7000,
                        color: col,
                        fillColor: col,
                        fillOpacity: 0.35 + norm * 0.45,
                        weight: 1
                    }).bindTooltip(`<b>Settlement Nursery Density</b><br>Probability: \${(d.density * 100).toFixed(2)}%`);
                    settlementDensityLayer.addLayer(rect);
                });
            }

            // 5. Demographic Connectivity Flows
            if (currentRun.connectivity_flows && currentRun.connectivity_flows.length > 0) {
                currentRun.connectivity_flows.forEach(flow => {
                    const weight = Math.max(2.0, flow.prob * 10.0);
                    const flowLine = L.polyline([[flow.src_lat, flow.src_lon], [flow.dst_lat, flow.dst_lon]], {
                        color: '#a855f7',
                        weight: weight,
                        opacity: 0.8,
                        dashArray: '8, 4'
                    }).bindTooltip(`<b>Connectivity Transition</b><br>\${flow.src} ➔ \${flow.dst}<br>Probability P_{ij}: \${(flow.prob * 100).toFixed(1)}%`);
                    connectivityFlowsLayer.addLayer(flowLine);
                });
            }

            // 6. Particle Selector Dropdown
            const pSelect = document.getElementById('particle-select');
            pSelect.innerHTML = '<option value="-1">Overview (All Larvae)</option>';
            pList.forEach(p => {
                const opt = document.createElement('option');
                opt.value = p.id;
                opt.textContent = `Larva #\${p.id} (\${p.status})`;
                pSelect.appendChild(opt);
            });

            // 7. Slider Bounds
            const slider = document.getElementById('time-slider');
            slider.max = nTimes - 1;
            slider.value = 0;

            // Auto-fit Bounds
            const allLats = pList.flatMap(p => p.lats);
            const allLons = pList.flatMap(p => p.lons);
            if (allLats.length > 0) {
                map.fitBounds([
                    [Math.min(...allLats) - 0.25, Math.min(...allLons) - 0.25],
                    [Math.max(...allLats) + 0.25, Math.max(...allLons) + 0.25]
                ]);
            }

            updateLegend("Seawater Temperature (T)", "°C", (hydro && hydro.grids && hydro.grids.temperature) ? hydro.grids.temperature.min : "1.5", (hydro && hydro.grids && hydro.grids.temperature) ? hydro.grids.temperature.max : "14.5", "linear-gradient(to right, #1d4ed8, #06b6d4, #10b981, #f59e0b, #ef4444)");
            updateFrame(0);
        }

        function switchRun(key) {
            if (RUNS[key]) {
                activeRunKey = key;
                currentRun = RUNS[key];
                isPlaying = false;
                document.getElementById('play-btn').textContent = "▶ Play";
                clearTimeout(animTimer);
                renderActiveRun();
            }
        }

        function updateFrame(step) {
            currentStep = step;
            const nTimes = currentRun.times.length;
            const pList = currentRun.particles;

            document.getElementById('time-slider').value = step;
            document.getElementById('time-readout').textContent = `Day \${currentRun.times[step].toFixed(2)} / \${currentRun.times[nTimes - 1].toFixed(2)} d`;

            pList.forEach((p, idx) => {
                const isFiltered = (outcomeFilter === "settled" && p.status !== "settled_successful") ||
                                   (outcomeFilter === "pelagic" && p.status !== "pelagic") ||
                                   (outcomeFilter === "dead" && p.alive);
                const isFocused = focusParticleId === -1 || focusParticleId === p.id;

                if (isFiltered || !isFocused) {
                    if (headMarkers[idx]) headMarkers[idx].setStyle({ opacity: 0, fillOpacity: 0 });
                    if (fullTrackPolylines[idx]) fullTrackPolylines[idx].setStyle({ opacity: 0 });
                    if (progressTrackPolylines[idx]) progressTrackPolylines[idx].setStyle({ opacity: 0 });
                    return;
                }

                const lat = p.lats[step];
                const lon = p.lons[step];
                const col = getColor(p, step);

                if (headMarkers[idx]) {
                    headMarkers[idx].setLatLng([lat, lon]);
                    headMarkers[idx].setStyle({
                        fillColor: col,
                        opacity: 1,
                        fillOpacity: 1,
                        radius: (focusParticleId === p.id) ? 9 : 5.5
                    });
                    headMarkers[idx].setTooltipContent(`<b>Larva #\${p.id}</b> (\${p.stages[step]} • \${p.depths[step].toFixed(1)}m)`);
                }

                if (fullTrackPolylines[idx]) {
                    fullTrackPolylines[idx].setStyle({
                        color: col,
                        opacity: (focusParticleId === p.id) ? 0.9 : 0.45,
                        weight: (focusParticleId === p.id) ? 3.5 : 2.0
                    });
                }

                if (progressTrackPolylines[idx]) {
                    if (showTrails && step > 0) {
                        const progPts = [];
                        for (let t = 0; t <= step; t++) {
                            progPts.push([p.lats[t], p.lons[t]]);
                        }
                        progressTrackPolylines[idx].setLatLngs(progPts);
                        progressTrackPolylines[idx].setStyle({
                            color: col,
                            opacity: (focusParticleId === p.id) ? 1.0 : 0.95,
                            weight: (focusParticleId === p.id) ? 4.5 : 3.2
                        });
                    } else if (showTrails && step === 0) {
                        const initPts = [[p.lats[0], p.lons[0]], [p.lats[Math.min(1, nTimes - 1)], p.lons[Math.min(1, nTimes - 1)]]];
                        progressTrackPolylines[idx].setLatLngs(initPts);
                        progressTrackPolylines[idx].setStyle({ color: col, opacity: 0.9, weight: 3.0 });
                    } else {
                        progressTrackPolylines[idx].setStyle({ opacity: 0 });
                    }
                }
            });

            if (focusParticleId !== -1) {
                const p = pList.find(item => item.id === focusParticleId);
                if (p) updateHUD(p, step);
            }
        }

        function updateHUD(p, step) {
            document.getElementById('hud-id').textContent = `#\${p.id}`;
            document.getElementById('hud-stage').textContent = p.stages[step];
            document.getElementById('hud-depth').textContent = `\${p.depths[step].toFixed(1)} m`;
            document.getElementById('hud-temp').textContent = `\${p.temps[step].toFixed(2)} °C`;
            document.getElementById('hud-sal').textContent = `33.2 PSU`;
            document.getElementById('hud-current').textContent = `~8.5 cm/s`;
            document.getElementById('hud-strat').textContent = `0.85 10⁻⁴s⁻²`;
            document.getElementById('hud-diff').textContent = `2.5 10⁻⁴m²/s`;
            document.getElementById('hud-surv').textContent = `\${(p.survs[step] * 100).toFixed(1)} %`;
        }

        function selectParticle(id) {
            focusParticleId = id;
            document.getElementById('particle-select').value = id;
            const p = currentRun.particles.find(item => item.id === id);
            if (p) {
                map.setView([p.lats[currentStep], p.lons[currentStep]], 9);
                updateHUD(p, currentStep);
            }
            updateFrame(currentStep);
        }

        function playLoop() {
            if (!isPlaying) return;
            const nTimes = currentRun.times.length;
            let nextStep = currentStep + playSpeed;
            if (nextStep >= nTimes) nextStep = 0;
            updateFrame(nextStep);
            animTimer = setTimeout(playLoop, 60);
        }

        // Live Cursor Telemetry Probe
        map.on('mousemove', function(e) {
            if (focusParticleId !== -1) return;
            const lat = e.latlng.lat;
            const lon = e.latlng.lng;
            const hydro = currentRun.hydrodynamics;
            if (hydro && hydro.lons && hydro.lats) {
                const hLons = hydro.lons;
                const hLats = hydro.lats;
                const minLon = hLons[0], maxLon = hLons[hLons.length - 1];
                const minLat = hLats[0], maxLat = hLats[hLats.length - 1];

                if (lon >= minLon && lon <= maxLon && lat >= minLat && lat <= maxLat) {
                    const nx = hLons.length, ny = hLats.length;
                    const i = Math.min(nx - 1, Math.max(0, Math.floor(((lon - minLon) / (maxLon - minLon)) * (nx - 1))));
                    const j = Math.min(ny - 1, Math.max(0, Math.floor(((lat - minLat) / (maxLat - minLat)) * (ny - 1))));

                    const lvl = (hydro.depth_levels && hydro.depth_levels[selectedDepthIdx]) ? hydro.depth_levels[selectedDepthIdx] : null;
                    const curDepth = (hydro.depths && hydro.depths.length > selectedDepthIdx) ? Math.abs(hydro.depths[selectedDepthIdx]).toFixed(1) : "0.0";
                    const g = hydro.grids;

                    document.getElementById('hud-id').textContent = `\${lat.toFixed(2)}°N, \${lon.toFixed(2)}°E`;
                    document.getElementById('hud-stage').textContent = `Eulerian @ \${curDepth}m`;
                    document.getElementById('hud-depth').textContent = `\${g && g.bathymetry ? g.bathymetry.data[i][j] : -120} m`;
                    document.getElementById('hud-temp').textContent = `\${lvl && lvl.temperature ? lvl.temperature.data[i][j] : (g && g.temperature ? g.temperature.data[i][j] : 8.0)} °C`;
                    document.getElementById('hud-sal').textContent = `\${lvl && lvl.salinity ? lvl.salinity.data[i][j] : (g && g.salinity ? g.salinity.data[i][j] : 32.5)} PSU`;
                    document.getElementById('hud-current').textContent = `\${lvl && lvl.speed ? lvl.speed.data[i][j] : (g && g.speed ? g.speed.data[i][j] : 8.0)} cm/s`;
                    document.getElementById('hud-strat').textContent = `\${lvl && lvl.stratification ? lvl.stratification.data[i][j] : '-'} 10⁻⁴s⁻²`;
                    document.getElementById('hud-diff').textContent = `\${lvl && lvl.diffusion ? lvl.diffusion.data[i][j] : '-'} 10⁻⁴m²/s`;
                    document.getElementById('hud-surv').textContent = `SSH \${g && g.elevation ? g.elevation.data[i][j] : 0.0} cm`;
                }
            }
        });

        // Layer Manager Event Listeners
        const layerControls = [
            { id: 'layer-fulltracks', layer: fullTracksLayer },
            { id: 'layer-progtrails', layer: progressTracksLayer },
            { id: 'layer-heads', layer: activeHeadsLayer },
            { id: 'layer-release', layer: releaseMarkersLayer },
            { id: 'layer-settle', layer: settlementMarkersLayer },
            { id: 'layer-density', layer: settlementDensityLayer, onSelect: () => updateLegend("Nursery Settlement Density", "%", "0.0", "100.0", "linear-gradient(to right, #1e3a8a, #06b6d4, #f59e0b, #ef4444)") },
            { id: 'layer-hydro-advection', layer: hydroAdvectionLayer },
            { id: 'layer-hydro-speed', layer: hydroSpeedLayer, onSelect: () => {
                const lvl = currentRun.hydrodynamics?.depth_levels?.[selectedDepthIdx];
                updateLegend("Current Speed (|u_h|)", "cm/s", "0.0", (lvl && lvl.speed) ? lvl.speed.max : (currentRun.hydrodynamics?.grids?.speed?.max || "25.0"), "linear-gradient(to right, #3b82f6, #06b6d4, #10b981, #facc15, #ef4444)");
            }},
            { id: 'layer-hydro-temp', layer: hydroTempLayer, onSelect: () => {
                const lvl = currentRun.hydrodynamics?.depth_levels?.[selectedDepthIdx];
                updateLegend("Seawater Temperature (T)", "°C", (lvl && lvl.temperature) ? lvl.temperature.min : (currentRun.hydrodynamics?.grids?.temperature?.min || "1.5"), (lvl && lvl.temperature) ? lvl.temperature.max : (currentRun.hydrodynamics?.grids?.temperature?.max || "14.5"), "linear-gradient(to right, #1d4ed8, #06b6d4, #10b981, #f59e0b, #ef4444)");
            }},
            { id: 'layer-hydro-sal', layer: hydroSalLayer, onSelect: () => {
                const lvl = currentRun.hydrodynamics?.depth_levels?.[selectedDepthIdx];
                updateLegend("Practical Salinity (S)", "PSU", (lvl && lvl.salinity) ? lvl.salinity.min : (currentRun.hydrodynamics?.grids?.salinity?.min || "31.0"), (lvl && lvl.salinity) ? lvl.salinity.max : (currentRun.hydrodynamics?.grids?.salinity?.max || "35.5"), "linear-gradient(to right, #10b981, #06b6d4, #3b82f6, #6366f1, #8b5cf6)");
            }},
            { id: 'layer-hydro-strat', layer: hydroStratLayer, onSelect: () => {
                const lvl = currentRun.hydrodynamics?.depth_levels?.[selectedDepthIdx];
                updateLegend("Stratification (N²)", "10⁻⁴ s⁻²", (lvl && lvl.stratification) ? lvl.stratification.min : "0.0", (lvl && lvl.stratification) ? lvl.stratification.max : "5.0", "linear-gradient(to right, #440154, #3b528b, #21908c, #5dc963, #fde725)");
            }},
            { id: 'layer-hydro-diff', layer: hydroDiffLayer, onSelect: () => {
                const lvl = currentRun.hydrodynamics?.depth_levels?.[selectedDepthIdx];
                updateLegend("Turbulent Diffusivity (κ_v)", "10⁻⁴ m²/s", (lvl && lvl.diffusion) ? lvl.diffusion.min : "0.1", (lvl && lvl.diffusion) ? lvl.diffusion.max : "10.0", "linear-gradient(to right, #0d0887, #6a00a8, #b12a90, #e16462, #fca636, #f0f921)");
            }},
            { id: 'layer-hydro-eta', layer: hydroEtaLayer, onSelect: () => updateLegend("Free Surface Height (η)", "cm", currentRun.hydrodynamics?.grids?.elevation?.min || "-10.0", currentRun.hydrodynamics?.grids?.elevation?.max || "+10.0", "linear-gradient(to right, #3b82f6, #f8fafc, #ef4444)") },
            { id: 'layer-hydro-w', layer: hydroUpwellingLayer, onSelect: () => {
                const lvl = currentRun.hydrodynamics?.depth_levels?.[selectedDepthIdx];
                updateLegend("Vertical Velocity / Upwelling", "mm/s", (lvl && lvl.upwelling) ? lvl.upwelling.min : (currentRun.hydrodynamics?.grids?.upwelling?.min || "-0.5"), (lvl && lvl.upwelling) ? lvl.upwelling.max : (currentRun.hydrodynamics?.grids?.upwelling?.max || "+0.5"), "linear-gradient(to right, #1e40af, #94a3b8, #b91c1c)");
            }},
            { id: 'layer-hydro-bathy', layer: hydroBathyLayer, onSelect: () => updateLegend("Seafloor Bathymetry", "m", currentRun.hydrodynamics?.grids?.bathymetry?.min || "-3000", "0", "linear-gradient(to right, #030712, #1e3a8a, #0284c7, #38bdf8)") },
            { id: 'layer-strata', layer: strataLayer },
            { id: 'layer-conn', layer: connectivityFlowsLayer }
        ];

        function updateActiveHydroLegend() {
            layerControls.forEach(ctrl => {
                const el = document.getElementById(ctrl.id);
                if (el && el.checked && ctrl.onSelect) {
                    ctrl.onSelect();
                }
            });
        }

        layerControls.forEach(ctrl => {
            const el = document.getElementById(ctrl.id);
            if (el) {
                el.addEventListener('change', function() {
                    if (this.checked) {
                        map.addLayer(ctrl.layer);
                        if (ctrl.onSelect) ctrl.onSelect();
                    } else {
                        map.removeLayer(ctrl.layer);
                    }
                });
            }
        });

        // Hydrodynamic Depth Select Listener
        const depthSelect = document.getElementById('hydro-depth-select');
        if (depthSelect) {
            depthSelect.addEventListener('change', function(e) {
                selectedDepthIdx = parseInt(e.target.value);
                rebuildHydroLayers();
                updateActiveHydroLegend();
            });
        }

        // Hydrodynamic Opacity Slider
        const opacitySlider = document.getElementById('hydro-opacity-slider');
        if (opacitySlider) {
            opacitySlider.addEventListener('input', function(e) {
                hydroOpacity = parseFloat(e.target.value) / 100.0;
                document.getElementById('hydro-opacity-val').textContent = e.target.value + "%";
                activeRasterLayers.forEach(layer => {
                    if (layer && layer.setOpacity) layer.setOpacity(hydroOpacity);
                });
            });
        }

        // Controller Event Listeners
        document.getElementById('play-btn').addEventListener('click', function() {
            isPlaying = !isPlaying;
            this.textContent = isPlaying ? "⏸ Pause" : "▶ Play";
            if (isPlaying) playLoop();
            else clearTimeout(animTimer);
        });

        document.getElementById('reset-btn').addEventListener('click', function() {
            isPlaying = false;
            document.getElementById('play-btn').textContent = "▶ Play";
            clearTimeout(animTimer);
            updateFrame(0);
        });

        document.getElementById('time-slider').addEventListener('input', function(e) {
            updateFrame(parseInt(e.target.value));
        });

        document.querySelectorAll('.speed-btn').forEach(btn => {
            btn.addEventListener('click', function() {
                document.querySelectorAll('.speed-btn').forEach(b => b.classList.remove('active'));
                this.classList.add('active');
                playSpeed = parseInt(this.getAttribute('data-speed'));
            });
        });

        document.getElementById('color-mode-select').addEventListener('change', function(e) {
            colorMode = e.target.value;
            updateFrame(currentStep);
        });

        document.getElementById('outcome-filter-select').addEventListener('change', function(e) {
            outcomeFilter = e.target.value;
            updateFrame(currentStep);
        });

        document.getElementById('particle-select').addEventListener('change', function(e) {
            selectParticle(parseInt(e.target.value));
        });

        // Initialize First Render
        renderActiveRun();
    </script>
</body>
</html>
"""

    write(output_path, html_content)
    println("Exported multi-layer interactive dashboard to $(output_path)")
    return output_path
end

"""
    export_interactive_tracks_html(
        trajectories::NamedTuple,
        output_path::AbstractString;
        kwargs...
    ) -> String

Multiple-dispatch convenience overload accepting `trajectories` as the primary argument,
forwarding to the file-path first method signature.

# Mathematical & Physical Context
Visualizes 4D Lagrangian trajectory vectors \$\\mathbf{x}_p(t) = (\\\\\\\\lambda_p(t), \\phi_p(t), z_p(t))\$
co-registered with Eulerian hydrodynamic fields (advection currents \$\\\\\\\\\\\\\\boldsymbol{u}_h\$,
temperature \$T\$, salinity \$S\$, stratification \$N^2\$, and turbulent diffusivity \$\\\\\\\\\\\\\\kappa_v\$)
across discrete depth levels and simulation epochs.

# Inputs
- `trajectories::NamedTuple`: Particle tracking output containing `lons`, `lats`, `depths`, `times`.
- `output_path::AbstractString`: Output path for the generated standalone HTML file.
- `kwargs...`: Additional keyword arguments forwarded to the file-path primary signature.

# Outputs
- `String`: Canonical file path to the generated HTML dashboard.
"""
function export_interactive_tracks_html(
    trajectories::NamedTuple,
    output_path::AbstractString;
    kwargs...
)::String
    return export_interactive_tracks_html(
        output_path;
        trajectories = trajectories,
        kwargs...
    )
end

"""
    plot_interactive_trajectories_map(
        trajectories::Union{Nothing, NamedTuple} = nothing;
        runs_data = nothing,
        hydrodynamics = nothing,
        gridded_dispersal = nothing,
        connectivity = nothing,
        recruitment_metrics = nothing,
        output_path::AbstractString = "outputs/interactive_larval_tracks.html",
        bathymetry = nothing,
        strata_definitions = nothing,
        title::AbstractString = "Interactive Larval Dispersal & Demographic Connectivity Map"
    ) -> String

Convenience wrapper to generate and export an interactive Leaflet HTML map of particle tracks,
supporting multi-run data dictionaries, hydrodynamic Eulerian flow fields, and full spatial layer suites.
"""
function plot_interactive_trajectories_map(
    trajectories::Union{Nothing, NamedTuple} = nothing;
    runs_data = nothing,
    hydrodynamics = nothing,
    gridded_dispersal = nothing,
    connectivity = nothing,
    recruitment_metrics = nothing,
    output_path::AbstractString = "outputs/interactive_larval_tracks.html",
    bathymetry = nothing,
    strata_definitions = nothing,
    title::AbstractString = "Interactive Larval Dispersal & Demographic Connectivity Map"
)::String
    return export_interactive_tracks_html(
        output_path;
        trajectories = trajectories,
        runs_data = runs_data,
        hydrodynamics = hydrodynamics,
        gridded_dispersal = gridded_dispersal,
        connectivity = connectivity,
        recruitment_metrics = recruitment_metrics,
        bathymetry = bathymetry,
        strata_definitions = strata_definitions,
        title = title
    )
end

# ─────────────────────────────────────────────────────────────────────────────
# Hydrodynamic Field Video & GIF Animations (Oceananigans / CairoMakie)
# ─────────────────────────────────────────────────────────────────────────────

"""
    resolve_hydrodynamic_frame_data(
        hydro_frame::NamedTuple,
        variable::Symbol
    ) -> Tuple{Matrix{Float64}, Symbol, String}

Extract a 2D scalar spatial matrix, default colormap, and descriptive physical unit label
for a given hydrodynamic diagnostic variable at the selected discrete depth level.

# Inputs
- `hydro_frame::NamedTuple`: Hydrodynamic snapshot dataset extracted via
  `extract_hydrodynamic_dataset`.
- `variable::Symbol`: Target variable symbol (`:speed`, `:vorticity`, `:temperature`,
  `:salinity`, `:density`, `:stratification`, `:elevation`, `:w`).

# Outputs
- `Tuple{Matrix{Float64}, Symbol, String}`: 2D field matrix, CairoMakie colormap, label.
"""
function resolve_hydrodynamic_frame_data(
    hydro_frame::NamedTuple,
    variable::Symbol
)::Tuple{Matrix{Float64}, Symbol, String}
    k = hydro_frame.depth_index
    z_cur = (hasproperty(hydro_frame, :depths) && k <= length(hydro_frame.depths)) ?
        hydro_frame.depths[k] : -10.0
    bathy = hasproperty(hydro_frame, :bathymetry) ? hydro_frame.bathymetry : nothing

    mask_dry(mat::AbstractMatrix) = begin
        out = copy(mat)
        if !isnothing(bathy) && size(bathy) == size(out)
            # Immersed cells: bottom is shallower than current depth or dry land (bathy >= 0)
            is_dry = (bathy .>= 0.0) .| (z_cur .< bathy)
            out[is_dry] .= NaN
        elseif any(out .!= 0.0)
            out[out .== 0.0] .= NaN
        end
        return out
    end

    if variable in (:salinity, :S, :sal)
        return (mask_dry(Float64.(hydro_frame.salinity[:, :, k])), :haline, "Practical Salinity (PSU)")
    elseif variable in (:temperature, :T, :temp, :theta)
        return (mask_dry(Float64.(hydro_frame.temperature[:, :, k])), :thermal, "Temperature (°C)")
    elseif variable in (:advection, :speed, :current, :velocity, :u_h)
        return (mask_dry(Float64.(hydro_frame.speed[:, :, k] .* 100.0)), :viridis, "Current Speed |u_h| (cm s⁻¹)")
    elseif variable in (:diffusion, :diffusivity, :kappa, :eddy_diffusion)
        return (mask_dry(Float64.(hydro_frame.diffusion[:, :, k] .* 10000.0)), :turbid, "Eddy Diffusivity κ (10⁻⁴ m² s⁻¹)")
    elseif variable in (:viscosity, :nu, :eddy_viscosity)
        return (mask_dry(Float64.(hydro_frame.viscosity[:, :, k] .* 10000.0)), :deep, "Eddy Viscosity ν (10⁻⁴ m² s⁻¹)")
    elseif variable in (:density, :rho)
        return (mask_dry(Float64.(hydro_frame.density[:, :, k])), :dense, "Potential Density (kg m⁻³)")
    elseif variable in (:stratification, :N2)
        return (mask_dry(Float64.(hydro_frame.stratification[:, :, k] .* 10000.0)), :ice, "Stratification N² (10⁻⁴ s⁻²)")
    elseif variable in (:richardson, :Ri)
        return (mask_dry(Float64.(hydro_frame.richardson_number[:, :, k])), :spectral, "Richardson Number Ri")
    elseif variable in (:vorticity, :zeta)
        vort_mat = ndims(hydro_frame.vorticity) == 2 ?
            Float64.(hydro_frame.vorticity[:, :]) :
            Float64.(hydro_frame.vorticity[:, :, k])
        return (mask_dry(vort_mat .* 100000.0), :balance, "Relative Vorticity ζ (10⁻⁵ s⁻¹)")
    elseif variable in (:elevation, :eta, :ssh)
        return (Float64.(hydro_frame.elevation .* 100.0), :delta, "Sea Surface Elevation η (cm)")
    elseif variable in (:w, :vertical_velocity)
        return (mask_dry(Float64.(hydro_frame.w[:, :, k] .* 1000.0)), :curl, "Vertical Velocity w (mm s⁻¹)")
    else
        return (mask_dry(Float64.(hydro_frame.speed[:, :, k] .* 100.0)), :viridis, "Current Speed |u_h| (cm s⁻¹)")
    end
end

