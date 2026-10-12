"""
    two_segment_z_faces(; z_break, z_min, nz_above, nz_below, scaling_above, scaling_below)

Build a two-segment vertically stretched grid resolving a shallow, biologically active layer in
detail while still reaching a much deeper domain floor.

The single-segment tanh used previously cannot do both jobs. Over 0 to -5000 m it puts its
*coarsest* cell at the surface, which left the shallowest cell centre at -258 m: with `nz = 20`
there were **no cell centres above -200 m and none in the 0-50 m nursery band**, so the cold
intermediate layer, the thermocline, and the settlement depth were all unresolved. Shallowing
`z_min` instead would clip the 22 % of the domain that lies below 400 m (the Laurentian Fan
reaches ~4850 m) and put a false wall at the shelf break.

This grid instead splits the column at `z_break`:
- **above** `z_break` (e.g. 0 to -400 m): `nz_above` tanh-stretched levels, surface-refined, so the
  near-surface and nursery layer is resolved at metre scale;
- **below** `z_break` (e.g. -400 to -5000 m): `nz_below` levels, so the deep basin is retained
  without spending levels where nothing is resolved.

# Inputs
- `z_break::Real`: Depth of the segment boundary, negative (e.g. `-400.0`).
- `z_min::Real`: Domain floor, negative (e.g. `-5000.0`).
- `nz_above::Int`: Number of levels between 0 and `z_break`.
- `nz_below::Int`: Number of levels between `z_min` and `z_break`.
- `scaling_above::Real`: tanh stretching strength for the upper segment (larger = more
  surface refinement). Default `2.0`.
- `scaling_below::Real`: tanh stretching strength for the lower segment. Default `0.0`, i.e.
  near-uniform, which is appropriate for the deep basin where vertical structure is weak.

# Outputs
- `Vector{Float64}`: `nz_above + nz_below + 1` face depths, strictly ascending from `z_min` to 0.
"""
function two_segment_z_faces(;
    z_break::Real = -400.0,
    z_min::Real = -5000.0,
    nz_above::Int = 24,
    nz_below::Int = 16,
    scaling_above::Real = 2.0,
    scaling_below::Real = 0.0
)
    nz_above > 0 || throw(ArgumentError("nz_above must be positive, got $nz_above"))
    nz_below > 0 || throw(ArgumentError("nz_below must be positive, got $nz_below"))
    zb = Float64(z_break)
    zf = Float64(z_min)
    (zf < zb < 0.0) || throw(ArgumentError(
        "require z_min < z_break < 0, got z_min=$zf z_break=$zb"))

    # tanh surface-refined face positions, 0 at the surface and 1 at the segment floor.
    # `s` is floored away from zero: the formula divides by `exp(s) - 1`, which is 0 at s = 0 and
    # would yield NaN. A tiny positive s is the uniform limit, so flooring is the correct way to
    # ask for "no stretching" rather than special-casing it.
    function stretched_faces(n::Int, depth::Float64, s_in::Real)
        s = max(0.01, Float64(s_in))
        denom = exp(s) - 1.0
        f = Vector{Float64}(undef, n + 1)
        for k in 0:n
            y = 1.0 - k / n                       # 1 at surface, 0 at floor
            f[k + 1] = -depth * ((exp(s * y) - 1.0) / denom)
        end
        f[end] = 0.0
        f
    end

    upper = stretched_faces(nz_above, -zb, Float64(scaling_above))    # -z_break .. 0
    lower = stretched_faces(nz_below, zb - zf, Float64(scaling_below))  # z_min-z_break .. 0

    # `lower` is already ascending (floor .. 0 of a column of thickness |z_break - z_min|);
    # shift it onto the real axis so it runs z_min .. z_break, then join.
    lower = lower .+ zb

    # Both segments end/start at z_break, so drop the duplicated join face; otherwise the grid has
    # a zero-thickness cell and one more face than levels.
    faces = vcat(lower, upper[2:end])
    faces[1] = zf
    faces[end] = 0.0
    issorted(faces) && all(diff(faces) .> 0) || error(
        "two_segment_z_faces produced non-monotonic or duplicate faces")
    return faces
end

"""
    stretched_tanh_z_faces(nz::Int = 20, Lz::Real = 5000.0;
                           scaling::Real = 2.0, csv_path = nothing) -> Vector{Float64}

Single-segment surface-refined grid: `nz` layers from `-Lz` to 0, finest at the surface.

This is a module-level, working replacement for the identically named helper that used to be
defined inside [`src/model/numerical_earth.jl`](@ref) and was therefore **not visible at module
scope** — the driver called it (`ParticleTrackingRun.jl`) and raised
`UndefVarError: stretched_tanh_z_faces not defined`, so a fresh hydrodynamics run could not build a
grid at all. The 2-year archive in `work/` predates that breakage.

Note that a single segment cannot resolve a shallow active layer while also reaching 5000 m: over
0 to -5000 m the top cell is still hundreds of metres thick. Prefer
[`two_segment_z_faces`](@ref) / [`scotian_shelf_z_faces`](@ref) for shelf work.
"""
function stretched_tanh_z_faces(
    nz::Int = 20,
    Lz::Real = 5000.0;
    scaling::Real = 2.0,
    csv_path::Union{Nothing, AbstractString} = nothing,
    strict::Bool = true
)
    nz > 0 || error("nz must be positive, got $nz")
    if !isnothing(csv_path) && !isempty(csv_path)
        if isfile(csv_path)
            f = load_vertical_grid_csv(csv_path)
            if length(f) == nz + 1
                return f
            end
            msg = "Vertical grid file $(csv_path) has $(length(f)) faces but nz + 1 = $(nz + 1) is required."
            # Silently substituting a generated grid here would run the whole simulation on a
            # vertical grid the operator did not choose, and the result looks plausible enough
            # that nothing downstream would reveal the substitution. `strict` defaults to true so
            # that a malformed file stops the run.
            strict && error(msg)
            @warn msg " Generating a tanh grid instead."
        elseif strict
            error("Vertical grid file $(csv_path) does not exist.")
        end
    end
    s = max(0.01, Float64(scaling))
    denom = exp(s) - 1.0
    depth = abs(Float64(Lz))
    f = Vector{Float64}(undef, nz + 1)
    for k in 0:nz
        y = 1.0 - k / nz                       # 1 at surface, 0 at floor
        f[k + 1] = -depth * ((exp(s * y) - 1.0) / denom)
    end
    f[1] = -depth
    f[end] = 0.0
    return f
end

"""
    scotian_shelf_z_faces(; z_min = -5000.0, z_break = -400.0, nz_above = 24, nz_below = 16)

Convenience wrapper around [`two_segment_z_faces`](@ref) tuned for the Scotian Shelf, where the
biologically active layer is roughly 10-400 m and the Laurentian Fan extends to ~4850 m.

Defaults give 40 levels: 24 in the upper 400 m (surface-refined, ~2 m at the surface) and 16 below
(the deep basin, near-uniform ~290 m). This puts ~10 levels in the 0-50 m nursery band and ~24 in
the 0-400 m active layer, versus **0** in that band under the previous 20-level single-tanh grid.
"""
function scotian_shelf_z_faces(;
    z_min::Real = -5000.0,
    z_break::Real = -400.0,
    nz_above::Int = 24,
    nz_below::Int = 16
)
    return two_segment_z_faces(z_break = z_break, z_min = z_min,
                               nz_above = nz_above, nz_below = nz_below,
                               scaling_above = 2.0, scaling_below = 0.0)
end
