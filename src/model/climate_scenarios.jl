"""
    climate_scenarios.jl

Climate-scenario deltas and their application to a hydrostatic model's initial state.

This file is deliberately limited to *forcing perturbations*. The temperature-dependent
pelagic larval duration and the thermal mortality rate that consume them live in
`larval_behavior.jl`, because they are stage biology rather than climate forcing.

# Data-driven scenarios
The deltas here are fixed per-emissions-scenario coefficients. They are not reanalysis, and
they carry no horizontal structure: the resulting anomaly is uniform in `(x, y)` and decays
only with depth. For spatially resolved, time-varying scenarios the perturbation should come
from reanalysis instead (GLORYS / ERA5 via `NumericalEarth`), applied with `Metadatum` and
`DataWrangling.FieldRegridding` — the same machinery `regrid_bathymetry` uses. Note that
`NumericalEarth.Column` is a struct describing a point/column location and is **not** a
dataset reader.
"""

using Oceananigans
using Dates
using NumericalEarth
using Interpolations: interpolate, extrapolate, Gridded, Linear
using Oceananigans.OutputReaders: FieldTimeSeries

"""
    apply_prescribed_surface_state!(model, sst, sss) -> model

Force the model's surface layer to a prescribed SST/SSS field.

# Why this uses `set!` and not boundary conditions

An earlier version of this function tried to "rebuild" `model.boundary_conditions.T.top`
from the prescribed fields. That could not work, for two independent reasons:

1. `HydrostaticFreeSurfaceModel` has no `boundary_conditions` field. Its fields are
   `architecture, grid, clock, advection, buoyancy, coriolis, free_surface, forcing,
   closure, particles, biogeochemistry, velocities, transport_velocities, tracers,
   pressure, closure_fields, timestepper, auxiliary_fields, vertical_coordinate,
   boundary_transport`. Boundary conditions live on the individual `Field`s and are baked
   in when the model is constructed, so the original code would have raised `FieldError`.
2. Even setting that aside, swapping a top `T`/`S` condition on an already-constructed
   standalone model is not a supported operation.

The supported way to drive a standalone model from a time-dependent surface state is to
build an `EarthSystemModel` with a `PrescribedOcean` component — see
[`prescribed_ocean_component`](@ref). For the common case where only the *initial* surface
state matters, this function writes the prescribed field into the model's top interior
layer with `set!`, which is well defined and is what the driver actually needs.

# Inputs
- `model`: `HydrostaticFreeSurfaceModel` to initialise.
- `sst`, `sss`: surface temperature (°C) and salinity (PSU). Either a `FieldTimeSeries`,
  a `Field`, or anything array-like broadcastable over the horizontal grid.

# Outputs
- The model, with its top interior layer set from `sst`/`sss`.

# Notes
Only the topmost interior layer is written. This sets an *initial condition*; it does not
impose a time-dependent Dirichlet condition, and it does not constrain the free surface.
"""
function apply_prescribed_surface_state!(model, sst, sss)
    grid = model.grid
    base = grid isa ImmersedBoundaryGrid ? grid.underlying_grid : grid

    # Exact shape validation: a silently broadcast or edge-clamped value here would
    # corrupt the initial state without any indication.
    check_shape(data, name) = begin
        values = surface_values(data)
        size(values, 1) == base.Nx && size(values, 2) == base.Ny || error(
            "$name must be (Nx=$(base.Nx), Ny=$(base.Ny)) to match the model grid, " *
            "got $(size(values)).")
        values
    end

    apply_to_surface_layer!(model.tracers.T, check_shape(sst, "sst"))
    apply_to_surface_layer!(model.tracers.S, check_shape(sss, "sss"))
    return model
end

"""
    apply_to_surface_layer!(field, values) -> field

Write `values` into the topmost interior layer of `field`, leaving all other layers
untouched, and push the result back into the model.

`set!(model, T = f)` would overwrite the *entire* tracer column with a depth-independent
callable, which is not what prescribing a *surface* state means. Editing the interior
array directly is exact: the top layer is replaced, everything below is preserved, and
there is no reliance on float equality against a surface depth.
"""
function apply_to_surface_layer!(field, values)
    data = Array(interior(field))
    data[:, :, end] .= Float64.(values)
    set!(field, data)
    return field
end

"""
    surface_values(data) -> AbstractMatrix{<:Real}

Extract the first time slice of a surface-state input as a 2D `(x, y)` array.

Accepts a `FieldTimeSeries` (first time slice), a `Field` (its parent array), or a plain
array. The first-slice behaviour is deliberate: these helpers initialise a model, they do
not impose a time-dependent boundary condition.
"""
function surface_values(data)
    if data isa AbstractArray && ndims(data) == 3
        # (Nx, Ny, n_times) — take the first time slice.
        return data[:, :, 1]
    end
    if hasproperty(data, :data) && data.data isa AbstractArray
        d = data.data
        return ndims(d) == 3 ? d[:, :, 1] : d
    end
    return data
end

"""
    prescribed_ocean_component(grid, sst, sss; kwargs...) -> PrescribedOcean

Build a `PrescribedOcean` component for use inside an `EarthSystemModel`.

This is the supported route for driving a model from time-dependent prescribed surface
state: `PrescribedOcean` holds SST/SSS/velocities as `FieldTimeSeries` and lets the ocean
component exchange surface fluxes with it. Use this when the surface state must evolve
during the run; use [`apply_prescribed_surface_state!`](@ref) when only the initial
surface state matters.

# Inputs
- `grid`: model grid.
- `sst`, `sss`: `FieldTimeSeries` of surface temperature and salinity.
- `times`: model time instances the series is sampled on.

# Outputs
- `PrescribedOcean` component.
"""
function prescribed_ocean_component(grid, sst, sss; times = nothing, density = 1025.0, heat_capacity = 4000.0)
    kwargs = Dict{Symbol, Any}(:density => density, :heat_capacity => heat_capacity)
    isnothing(times) || (kwargs[:times] = times)
    isnothing(sst) || (kwargs[:sea_surface_temperature] = sst)
    isnothing(sss) || (kwargs[:sea_surface_salinity] = sss)
    return PrescribedOcean(grid; kwargs...)
end

"""
    get_climate_scenario_deltas(
        scenario::Symbol = :ssp245;
        year::Int = 2050,
        baseline_year::Real = 2015.0,
        horizon_year::Real = 2050.0
    ) -> NamedTuple

Perturbation coefficients for a climate scenario, scaled linearly between `baseline_year`
and `horizon_year`.

# Returns
- `NamedTuple` with fields
  - `ΔT_surface` — surface temperature anomaly (°C)
  - `ΔT_deep` — deep temperature anomaly (°C)
  - `ΔS_surface` — surface salinity anomaly (PSU)
  - `Δwind_factor` — multiplier on the prescribed wind stress
  - `description` — human-readable scenario name

# Scenarios
| Symbol | Description |
|---|---|
| `:historical`, `:baseline`, `:climatology` | No perturbation (repeating annual cycle) |
| `:ssp126` | CMIP6 SSP1-2.6 (low emissions) |
| `:ssp245` | CMIP6 SSP2-4.5 (intermediate) |
| `:ssp370` | CMIP6 SSP3-7.0 (high) |
| `:ssp585` | CMIP6 SSP5-8.5 (very high) |
| `:marine_heatwave`, `:mhw` | Transient Category III/IV marine heatwave |

# Notes
These are fixed literature coefficients, not reanalysis, and carry no horizontal
structure. See the module docstring for the data-driven alternative.

# References
- IPCC (2021). *Climate Change 2021: The Physical Science Basis*, WG1.
- Brickman, D., Wang, Z., & DeTracey, B. (2018). *Progress in Oceanography*, 164, 49-64.
  DOI: 10.5194/gmd-9-3461-2016
"""
function get_climate_scenario_deltas(
    scenario::Symbol = :ssp245;
    year::Int = 2050,
    baseline_year::Real = 2015.0,
    horizon_year::Real = 2050.0
)
    span = horizon_year - baseline_year
    time_factor = span > 0.0 ? clamp((Float64(year) - Float64(baseline_year)) / span, 0.0, 5.0) : 1.0

    if scenario == :historical || scenario == :baseline || scenario == :climatology
        desc = scenario == :climatology ?
            "Climatological baseline (repeating annual cycle)" :
            "Historical / present-day climatological baseline"
        return (
            ΔT_surface = 0.0,
            ΔT_cil = 0.0,
            ΔT_deep = 0.0,
            ΔS_surface = 0.0,
            Δwind_factor = 1.0,
            description = desc
        )
    elseif scenario == :ssp126
        return (
            ΔT_surface = 1.1 * time_factor,
            ΔT_cil = 0.7 * time_factor,
            ΔT_deep = 0.5 * time_factor,
            ΔS_surface = -0.25 * time_factor,
            Δwind_factor = 1.0 + 0.05 * time_factor,
            description = "CMIP6 SSP1-2.6 (Low emissions / Paris target)"
        )
    elseif scenario == :ssp245
        return (
            ΔT_surface = 1.8 * time_factor,
            ΔT_cil = 1.2 * time_factor,
            ΔT_deep = 0.9 * time_factor,
            ΔS_surface = -0.45 * time_factor,
            Δwind_factor = 1.0 + 0.10 * time_factor,
            description = "CMIP6 SSP2-4.5 (Intermediate / Middle of the road)"
        )
    elseif scenario == :ssp370
        return (
            ΔT_surface = 2.6 * time_factor,
            ΔT_cil = 1.8 * time_factor,
            ΔT_deep = 1.4 * time_factor,
            ΔS_surface = -0.65 * time_factor,
            Δwind_factor = 1.0 + 0.15 * time_factor,
            description = "CMIP6 SSP3-7.0 (High emissions / Regional rivalry)"
        )
    elseif scenario == :ssp585
        return (
            ΔT_surface = 3.5 * time_factor,
            ΔT_cil = 2.4 * time_factor,
            ΔT_deep = 1.9 * time_factor,
            ΔS_surface = -0.85 * time_factor,
            Δwind_factor = 1.0 + 0.20 * time_factor,
            description = "CMIP6 SSP5-8.5 (Very high emissions / Fossil-fueled)"
        )
    elseif scenario == :marine_heatwave || scenario == :mhw
        return (
            ΔT_surface = 3.5,
            ΔT_cil = 1.0,
            ΔT_deep = 0.2,
            ΔS_surface = -0.1,
            Δwind_factor = 0.8,
            description = "Transient Category III/IV Marine Heatwave (MHW)"
        )
    else
        error(
            "Unknown climate scenario '$(scenario)'. " *
            "Available: :historical, :baseline, :climatology, :ssp126, :ssp245, " *
            ":ssp370, :ssp585, :marine_heatwave, :mhw"
        )
    end
end

"""
    apply_climate_scenario!(model; scenario, year, mixed_layer_depth) -> NamedTuple

Apply a climate-scenario anomaly to the model's existing `T`/`S`.

# Mathematical & Governing Formulation
The anomaly is added to whatever the model already holds, weighted towards the surface:
```math
\\Delta T(z) = \\Delta T_{deep} + (\\Delta T_{surf} - \\Delta T_{deep})\\, e^{z / h},
\\qquad
\\Delta S(z) = \\Delta S_{surf}\\, e^{z / h}
```
where \$h\$ is the e-folding depth. The surface-weighted form is the standard first-order
representation of a scenario anomaly: the signal is strongest at the surface and decays
into the deep ocean, with a floor at \$\\Delta T_{deep}\$.

# Why this perturbs rather than replaces

An earlier version *replaced* the tracer fields with a synthetic linear profile
\`\`T_0 + \\gamma z + \\Delta T(z)\`\` built from hardcoded \`baseline_*\` defaults. Three
problems followed from that:

1. It **discarded the baseline entirely**. The driver calls
   [`set_initial_stratification!`](@ref) — or loads real WOA23/GLORYS hydrography — and
   then this function, so a scenario run silently threw away the observed or
   three-layer-analytic state and substituted \`T_surf = 15\`, \`dT/dz = 0.01\`.
2. It **discarded horizontal structure**. The old \`climate_T(x, y, z)\` ignored \`x\` and
   \`y\` entirely, so any cross-shelf gradient, CIL structure, or data-driven spatial
   pattern in the baseline was erased.
3. It **was not a perturbation**, despite the name and docstring.

This version adds the anomaly to the current state, so it is composable with any baseline
and preserves spatial structure. The \`baseline_*\` keywords are gone; the baseline is now
whatever the model holds.

# Inputs
- `model`: Model to perturb; its `T`/`S` are read and written in place.
- `scenario::Symbol`: Emissions scenario.
- `year::Int`: Projection year.
- `mixed_layer_depth::Real`: e-folding depth of the anomaly (m).

# Outputs
- `NamedTuple`: The deltas that were applied, from
  [`get_climate_scenario_deltas`](@ref).

# Notes
The anomaly is horizontally uniform, so the result inherits the baseline's spatial
structure rather than adding any. A spatially resolved projection requires a CMIP
difference field; see the module docstring.

# References
- Brickman, D., Wang, Z., & DeTracey, B. (2018). *Progress in Oceanography*, 164, 49-64.
- Loder, J. W., van der Baaren, A., & Yashayaev, I. (2015). *Can. Tech. Rep. Hydrogr. Ocean Sci.*, 305, 142 pp.
"""
function apply_climate_scenario!(
    model;
    scenario::Symbol = :ssp245,
    year::Int = 2050,
    cil_depth::Real = 75.0,
    deep_depth::Real = 250.0
)
    deltas = get_climate_scenario_deltas(scenario, year = year)

    base = model.grid isa ImmersedBoundaryGrid ? model.grid.underlying_grid : model.grid

    Tdata = Array(interior(model.tracers.T))
    Sdata = Array(interior(model.tracers.S))

    @inbounds for k in 1:base.Nz
        z = Float64(Oceananigans.Grids.znode(k, base, Center()))
        
        # 3-Layer Vertical Stratification for the Scotian Shelf
        if z >= -cil_depth
            # Surface to Cold Intermediate Layer (CIL)
            f = z / -cil_depth
            ΔT = deltas.ΔT_surface * (1.0 - f) + deltas.ΔT_cil * f
            ΔS = deltas.ΔS_surface * (1.0 - f)
        elseif z >= -deep_depth
            # CIL to Deep slope water
            f = (z + cil_depth) / (-deep_depth + cil_depth)
            ΔT = deltas.ΔT_cil * (1.0 - f) + deltas.ΔT_deep * f
            ΔS = 0.0
        else
            # Below Deep boundary
            ΔT = deltas.ΔT_deep
            ΔS = 0.0
        end

        Tdata[:, :, k] .+= ΔT
        Sdata[:, :, k] .+= ΔS
    end

    set!(model.tracers.T, Tdata)
    set!(model.tracers.S, Sdata)
    return deltas
end
