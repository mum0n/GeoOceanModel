using Oceananigans
using Oceananigans.Forcings: ContinuousForcing
#
#  Lightweight data-assimilation utilities for nudging hydrodynamic fields toward
#  observations (e.g. bottom temperature). Provides quality control, an
#  observation operator, Newtonian nudging tendencies, and a simple optimal-
#  interpolation / 3D-Var-style analysis update.
# ==============================================================================

"""
    qc_observations(values; vmin = -5.0, vmax = 40.0) -> Vector{Bool}

Basic quality control for scalar observations: finite and within `[vmin, vmax]`.
"""
function qc_observations(values::AbstractVector{<:Real}; vmin::Real = -5.0, vmax::Real = 40.0)
    return [isfinite(v) && vmin <= v <= vmax for v in values]
end

"""
    obs_indices(hydro, lons, lats, depths; bottom = false) -> (Vector{Int}, Vector{Int}, Vector{Int})

Map observations to (i, j, k) grid indices of `hydro`. When `bottom = true` the
deepest wet level at each (i, j) is used instead of `depths`.
"""
function obs_indices(hydro, lons::AbstractVector{<:Real}, lats::AbstractVector{<:Real},
                     depths::AbstractVector{<:Real}; bottom::Bool = false)
    is = [argmin(abs.(hydro.lons .- lon)) for lon in lons]
    js = [argmin(abs.(hydro.lats .- lat)) for lat in lats]
    ks = Int[]
    for (n, d) in enumerate(depths)
        i, j = is[n], js[n]
        if bottom
            bval = (hydro.bathymetry isa AbstractMatrix &&
                    size(hydro.bathymetry, 1) >= i && size(hydro.bathymetry, 2) >= j) ?
                   hydro.bathymetry[i, j] : nothing
            kb = nothing
            for (kk, z) in enumerate(hydro.depths)
                if isfinite(z) && (bval === nothing || !isfinite(bval) || z >= bval)
                    kb = kk
                    break
                end
            end
            push!(ks, isnothing(kb) ? 1 : kb)
        else
            push!(ks, argmin(abs.(hydro.depths .- d)))
        end
    end
    return (is, js, ks)
end

"""
    nudging_tendency(field, obs_value, tau; mask = nothing) -> Array

Newtonian nudging tendency `(obs_value - field) / tau` (per cell), optionally
masked. `tau` is in the same time units as the caller's step.
"""
function nudging_tendency(field::AbstractArray{<:Real}, obs_value::Real, tau::Real;
                          mask::Union{Nothing, AbstractArray{Bool}} = nothing)
    tendency = (obs_value .- field) ./ tau
    if !isnothing(mask)
        tendency = tendency .* mask
    end
    return tendency
end

"""
    optimal_interpolation(background, obs_i, obs_j, obs_val; B = 1.0, R = 1.0) -> Matrix{Float64}

Simple per-cell optimal interpolation / 3D-Var-style update:
`analysis = background + K (obs - background)` with `K = B / (B + R)`.
Multiple observations landing in the same cell are averaged.
"""
function optimal_interpolation(background::AbstractMatrix{<:Real},
                              obs_i::AbstractVector{Int}, obs_j::AbstractVector{Int},
                              obs_val::AbstractVector{<:Real};
                              B::Real = 1.0, R::Real = 1.0)
    A = Float64.(background)
    nx, ny = size(A)
    # Accumulate observation increments per cell.
    inc = zeros(Float64, nx, ny)
    cnt = zeros(Int, nx, ny)
    for n in eachindex(obs_val)
        i, j = obs_i[n], obs_j[n]
        (1 <= i <= nx && 1 <= j <= ny) || continue
        inc[i, j] += obs_val[n] - A[i, j]
        cnt[i, j] += 1
    end
    K = B / (B + R)
    for i in 1:nx, j in 1:ny
        if cnt[i, j] > 0
            A[i, j] += K * (inc[i, j] / cnt[i, j])
        end
    end
    return A
end

"""
    load_observation_csv(path::AbstractString) -> NamedTuple

Load a CSV with columns `lon,lat,time,depth,value`. Returns empty vectors when
`path` is empty or missing, so simulations still run without observations.
"""
function load_observation_csv(path::AbstractString)
    empty = (lon = Float64[], lat = Float64[], time = Float64[],
             depth = Float64[], value = Float64[])
    if isempty(strip(path)) || !isfile(path)
        return empty
    end
    lines = readlines(path)
    length(lines) < 2 && return empty
    header = [strip(x) for x in split(lines[1], ",")]
    idx = Dict(name => i for (i, name) in enumerate(header))
    for required in ("lon", "lat", "time", "depth", "value")
        haskey(idx, required) || return empty
    end
    lons = Float64[]; lats = Float64[]; times = Float64[]
    depths = Float64[]; vals = Float64[]
    for line in lines[2:end]
        isempty(strip(line)) && continue
        cols = split(line, ",")
        try
            push!(lons, parse(Float64, strip(cols[idx["lon"]])))
            push!(lats, parse(Float64, strip(cols[idx["lat"]])))
            push!(times, parse(Float64, strip(cols[idx["time"]])))
            push!(depths, parse(Float64, strip(cols[idx["depth"]])))
            push!(vals, parse(Float64, strip(cols[idx["value"]])))
        catch
            continue
        end
    end
end

"""
    build_observation_nudging(obs; tau = 3600.0, vmin = -5.0, vmax = 40.0) -> Union{ContinuousForcing, Nothing}

Build a temperature nudging forcing from an observation NamedTuple. Returns
`nothing` when no observations are available.
"""
function build_observation_nudging(obs; tau::Real = 3600.0, vmin::Real = -5.0,
                                  vmax::Real = 40.0)
    isempty(obs.lon) && return nothing
    mask = qc_observations(obs.value; vmin = vmin, vmax = vmax)
    lons = obs.lon[mask]
    lats = obs.lat[mask]
    times = obs.time[mask]
    vals = obs.value[mask]
    isempty(lons) && return nothing

    function nudge(x, y, z, t, T)
        # nearest time
        ti = argmin(abs.(times .- t))
        candidates = findall(times .== times[ti])
        if isempty(candidates)
            candidates = collect(1:length(times))
        end
        dists = [(lons[i] - x)^2 + (lats[i] - y)^2 for i in candidates]
        ni = candidates[argmin(dists)]
        return (vals[ni] - T) / tau
    end

    return ContinuousForcing(nudge; field_dependencies = :T)
end

export qc_observations, obs_indices, nudging_tendency, optimal_interpolation,
       load_observation_csv, build_observation_nudging