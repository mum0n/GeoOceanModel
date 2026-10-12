"""
    tidal_forcing.jl

Resolving how tide enters the Oceananigans flow model, with the provenance and the placement
both selectable and both documented.

Two things decide whether a tidal forcing is correct, and neither is a single boolean:

1. **Provenance** — does the flow already contain tide? GLORYS12V1 and HYCOM are forced with
   tide and carry tidal constituents; WOA23 is a monthly climatology with no tidal variability
   at all; `synthetic` and `analytic` have none. See `tidal_provenance`.
2. **Placement** — where does the tide enter? See the note under `placement`.

`resolve_tidal_forcing` takes both. Defaulting either is how a wrong tide gets into a run.

See `TIDES_ASSESSMENT.md` for the full analysis and `hydrodynamic_model.jl` for the failure
that shaped the accepted form (8-13 m/s on the Shelf, watchdog tripped within six simulated
hours).

# Placement: through the edges, or through the interior

Three placements, and they are not equivalent:

- **`:boundary`** — relax toward the tidal velocity at the **open edges only**. Correct when
  the flow model is meant to be driven from the edges and have the tide propagate inward
  physically. This is the only placement that is right for a run whose interior is analytic:
  forcing the interior directly puts a tide everywhere at t = 0, which the real ocean does
  not do, and on a shallow shelf it is the configuration that diverged.
- **`:interior`** — a tidal body forcing across the domain. Only correct when the flow has no
  boundary constraint at all and the tide must be present from the start. Rarely what is
  wanted.
- **`:both`** — edges and interior together. Only correct when they are different tides:
  the edges driven by a reanalysis that already has tide, the interior by this forcing. Two
  tides of the same constituent is a doubling.

**`:boundary` is the default.** It is the placement that matches how tide actually enters,
and it is the placement that avoids the resonance that divergent run had.
"""

"""
    tidal_provenance(ocean_boundary_source::Symbol, hydrography_source::Symbol,
                     tides_source::Symbol) -> Symbol

Decide whether the flow itself already contains tide, so a tidal forcing is not added twice.

Returns `:none` (no tide anywhere; adding one is the only way to get it), `:in_flow` (the
flow carries tide; adding doubles it), or `:abstain` (the source is not in the table).

This is a lookup rather than a parameter because whether a product has tide is a property of
the product, not of the run: GLORYS12V1 and HYCOM are forced with tide and carry tidal
constituents in their velocity fields; WOA23 is a monthly climatological mean with no tidal
variability; `synthetic` and `analytic` have none. Keeping it in one place stops every caller
re-deriving it, or defaulting to the wrong branch.

# The failure this prevents

Superposing M2 onto a flow that already has it doubles it. M2 runs 0.009-0.159 m/s on the
Scotian Shelf's eastern seaboard and up to 0.713 m/s in the Bay of Fundy, so doubling gives a
signal comparable to the entire residual shelf flow. Omitting a tide is a visible error;
doubling one is not.
"""
function tidal_provenance(ocean_boundary_source::Symbol, hydrography_source::Symbol,
                          tides_source::Symbol)
    boundary_tidal = if ocean_boundary_source in (:glorys12v1, :hycom)
        :yes
    elseif ocean_boundary_source in (:woa23, :synthetic, :none)
        :no
    else
        :unknown
    end
    initial_tidal = if hydrography_source in (:glorys12v1, :hycom)
        :yes
    elseif hydrography_source in (:woa23, :synthetic, :none)
        :no
    else
        :unknown
    end

    if boundary_tidal === :yes || initial_tidal === :yes
        return :in_flow
    elseif boundary_tidal === :unknown || initial_tidal === :unknown
        return :abstain
    end
    return :none
end

"""
    resolve_tidal_forcing(;
        method::Symbol = :auto,
        placement::Symbol = :boundary,
        grid = nothing,
        ocean_boundary_source::Symbol = :glorys12v1,
        hydrography_source::Symbol = :woa23,
        tides_source::Symbol = :tpxo9,
        constituents::Vector{Symbol} = [:M2, :S2],
        active_edges::Union{Nothing, Vector{Symbol}} = nothing,
        time_interval = nothing,
        verbose::Bool = true,
        tide_already_in_flow::Union{Nothing, Bool} = nothing
    ) -> Union{Nothing, NamedTuple}

Resolve `(u = f, v = g)` for `build_hydrodynamic_model(; tidal_forcing = ...)`, or
`nothing` when tide should not be added.

# Why the placement default is `:boundary`

A body forcing across the domain is what produced 8-13 m/s and the divergence watchdog trip.
Relaxing toward the tidal velocity *at the open edges* is the placement that matches how the
tide actually enters, and the one the real field has.

# Why `tide_already_in_flow` defaults to nothing, not a boolean

The answer is a property of the source, so it is derived by `tidal_provenance` rather than
asked. Pass `true` or `false` only to override that derivation — which you should do only if
you know the source carries tide contrary to the table.

# Recipe selection

`:auto` tries `:harmonics` → `:coefficients` → `:analytic`, logging each step, because a
silent fallback is how a wrong tide gets into a run. Each recipe's benefit and risk is
commented in its own section below.
"""
function resolve_tidal_forcing(;
    method::Symbol = :auto,
    placement::Symbol = :boundary,
    grid = nothing,
    ocean_boundary_source::Symbol = :glorys12v1,
    hydrography_source::Symbol = :woa23,
    tides_source::Symbol = :tpxo9,
    constituents::Vector{Symbol} = [:M2, :S2],
    active_edges::Union{Nothing, Vector{Symbol}} = nothing,
    time_interval = nothing,
    verbose::Bool = true,
    tide_already_in_flow::Union{Nothing, Bool} = nothing
)
    placement in (:boundary, :interior, :both) || error(
        "Unknown tidal placement $(repr(placement)). Choose :boundary (edges only, the " *
        "physical default), :interior (a body forcing across the domain), or :both.")
    method in (:auto, :harmonics, :coefficients, :analytic) || error(
        "Unknown tidal recipe $(repr(method)). Choose :auto, :harmonics, :coefficients, " *
        "or :analytic.")

    derived = tidal_provenance(ocean_boundary_source, hydrography_source, tides_source)
    already_in_flow = something(tide_already_in_flow,
        derived !== :none && derived !== :abstain)

    if already_in_flow
        verbose && println("Tides: the flow already carries tide " *
            "(ocean_boundary_source = $(repr(ocean_boundary_source)), " *
            "hydrography_source = $(repr(hydrography_source))), so none is added here. " *
            "Adding it would double M2 and invalidate the transport. Returning nothing.")
        return nothing
    end
    if derived === :abstain
        verbose && println("Tides: NOT DECIDABLE -- $(repr(ocean_boundary_source)) and " *
            "$(repr(hydrography_source)) are not in the provenance table, so whether the " *
            "flow already has tide is unknown. Add them to `tidal_provenance` or pass " *
            "tide_already_in_flow explicitly. Proceeding as if there is no tide.")
    end

    if placement === :interior || placement === :both
        verbose && println("Tides: placement = $(repr(placement)) applies a *body* forcing " *
            "across the domain. That is the placement that reached 8-13 m/s on the " *
            "Scotian Shelf and tripped the divergence watchdog within six simulated hours " *
            "(0.4 m/s with tides off). Use :boundary unless the run has no constrained " *
            "edges at all.")
    end
    verbose && placement === :boundary &&
        println("Tides: placement = :boundary relaxes toward the tidal velocity at the " *
            "open edges only, so the tide propagates inward as it does in the real field.")

    order = if method == :auto
        [:harmonics, :coefficients, :analytic]
    else
        [method]
    end

    result = nothing
    tried = Symbol[]
    for recipe in order
        push!(tried, recipe)
        outcome = try
            _build_tidal_forcing(recipe; grid = grid, tides_source = tides_source,
                                 constituents = constituents,
                                 active_edges = active_edges,
                                 time_interval = time_interval, verbose = verbose)
        catch e
            verbose && println("Tides: recipe $(repr(recipe)) failed: " *
                               "$(first(split(sprint(showerror, e), "\n"))).")
            :failed
        end
        if outcome isa NamedTuple
            result = outcome
            verbose && method == :auto && println("Tides: using recipe $(repr(recipe))" *
                " (tried $(join(repr.(tried), ", "))).")
            break
        elseif outcome !== :unavailable && outcome !== :failed
            verbose && println("Tides: recipe $(repr(recipe)) returned nothing.")
        end
    end

    return result
end

"""
    _build_tidal_forcing(recipe::Symbol; kwargs...) -> Union{Nothing, NamedTuple, Symbol}

Dispatch to one recipe. Returns `:unavailable` or `:failed` so the selector can distinguish
"this route cannot be used here" from a hard error and move on without noise.
"""
function _build_tidal_forcing(recipe::Symbol; kwargs...)
    if recipe == :harmonics
        return _tidal_forcing_harmonics(; kwargs...)
    elseif recipe == :coefficients
        return _tidal_forcing_coefficients(; kwargs...)
    elseif recipe == :analytic
        return _tidal_forcing_analytic(; kwargs...)
    end
    return :unavailable
end

# ---------------------------------------------------------------------------------------
# Recipe 1: harmonic reconstruction from the atlas.
#
# BENEFIT: the most physical of the three -- real per-edge amplitudes and phases read from
#   TPXO, so the tide matches the domain instead of a constant. This is the form the model's
#   relaxation was designed around, and the one whose amplitudes were measured for this
#   shelf: 0.009-0.159 m/s on the eastern seaboard, 0.006-0.031 on the southern, up to
#   0.713 m/s in the Bay of Fundy.
# RISK: never executed. The path `read_tidal_velocity_harmonics` ->
#   `tidal_forcing_from_harmonics` has no callers anywhere in this package. Treat the first
#   run as unvalidated physics.
# ---------------------------------------------------------------------------------------
function _tidal_forcing_harmonics(;
    grid = nothing,
    tides_source::Symbol = :tpxo9,
    constituents::Vector{Symbol} = [:M2, :S2],
    active_edges::Union{Nothing, Vector{Symbol}} = nothing,
    time_interval = nothing,
    verbose::Bool = true
)
    grid === nothing && return :unavailable
    harmony_file = tpxo_constituent_file(tides_source)
    isfile(harmony_file) || (verbose &&
        println("Tides: no atlas at $(harmony_file); skipping the harmonics recipe.");
        return :unavailable)

    h = read_tidal_velocity_harmonics(harmony_file; constituents = constituents)
    return tidal_forcing_from_harmonics(h)
end

# ---------------------------------------------------------------------------------------
# Recipe 2: coefficient-grid reconstruction.
#
# BENEFIT: the most developed route in this package -- `TidalCoefficientGrid` carries
#   per-boundary amplitudes via `edge_tidal_velocity`, so each open edge is driven at its own
#   strength rather than one value for the domain. Physically the most defensible for a shelf
#   with a 0.713 m/s Fundy beside a 0.009 m/s eastern seaboard.
# RISK: also never executed, and there are two readers (`TidalCoefficientGrid` and
#   `TidalVelocityHarmonics`) with no statement of which is intended. If this recipe is
#   preferred, that ambiguity needs settling first -- it is a physics decision, not a code one.
# ---------------------------------------------------------------------------------------
function _tidal_forcing_coefficients(;
    grid = nothing,
    tides_source::Symbol = :tpxo9,
    constituents::Vector{Symbol} = [:M2, :S2],
    active_edges::Union{Nothing, Vector{Symbol}} = nothing,
    time_interval = nothing,
    verbose::Bool = true
)
    grid === nothing && return :unavailable
    harmony_file = tpxo_constituent_file(tides_source)
    isfile(harmony_file) || (verbose &&
        println("Tides: no atlas at $(harmony_file); skipping the coefficients recipe.");
        return :unavailable)

    harmonics = read_tidal_velocity_harmonics(harmony_file; constituents = constituents)
    g = TidalCoefficientGrid(harmonics)
    return tidal_forcing_coefficients(g)
end

# ---------------------------------------------------------------------------------------
# Recipe 3: analytic sines, the form `OpenBoundaryConditions` already implements inline.
#
# BENEFIT: needs no data file and no grid, so it always runs. Useful as a smoke test that the
#   relaxation plumbing is wired, and as a fallback that does not silently do nothing.
# RISK: **this is the form that diverged.** `numerical_earth.jl` applies `u_tide =
#   0.12 sin(...)` -- a spatially uniform amplitude at every point and every depth. On the
#   Scotian Shelf that reached 8-13 m/s and tripped the divergence watchdog within six
#   simulated hours; the same setup with tides off stayed at 0.4 m/s. Uniform forcing is
#   resonant on a shallow shelf. Use this to test the plumbing, not to run physics.
#
# It is kept available because the point of the selector is that a failed recipe is a
# one-word change. It is deliberately last, and deliberately labelled.
# ---------------------------------------------------------------------------------------
function _tidal_forcing_analytic(;
    grid = nothing,
    tides_source::Symbol = :tpxo9,
    constituents::Vector{Symbol} = [:M2, :S2],
    active_edges::Union{Nothing, Vector{Symbol}} = nothing,
    time_interval = nothing,
    verbose::Bool = true
)
    verbose && println("Tides: WARNING -- the analytic recipe is a spatially uniform " *
        "amplitude. It diverged on the Scotian Shelf (8-13 m/s, watchdog tripped within " *
        "6 simulated hours). Use it to test the plumbing, not to run physics.")

    omega_M2 = 2π / 44712.0
    omega_S2 = 2π / 43200.0
    u(x, y, z, t) = 0.12 * sin(omega_M2 * Float64(t)) + 0.05 * sin(omega_S2 * Float64(t))
    v(x, y, z, t) = 0.08 * cos(omega_M2 * Float64(t)) + 0.03 * cos(omega_S2 * Float64(t))
    return (u = u, v = v)
end
