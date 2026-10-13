"""
    hydrodynamic_model.jl

Configuration and initialization of hydrostatic free-surface ocean models
in Oceananigans.jl for coastal shelf domains.
"""

using Oceananigans
using Oceananigans.Units
# `ContinuousForcing` lives in `Oceananigans.Forcings`, which Oceananigans `using`s
# internally but does not re-export at top level. `ZeroForcing` is defined locally below,
# so it is deliberately not imported here.
using Oceananigans.BoundaryConditions: ContinuousBoundaryFunction
using Oceananigans.Forcings: ContinuousForcing, MultipleForcings, Relaxation
using Dates
using NCDatasets
using Interpolations
using Interpolations: interpolate, BSpline, Linear
import NumericalEarth: ocean_simulation, GLORYSDaily, GLORYSMonthly, WOAMonthly, WOAAnnual, BoundingBox, Metadatum
using SeawaterPolynomials.TEOS10: TEOS10EquationOfState

# Import domain constants

"""
    build_hydrodynamic_model(grid; kwargs...) -> HydrostaticFreeSurfaceModel

Build a hydrostatic ocean model through `NumericalEarth.ocean_simulation`, returning the
bare `HydrostaticFreeSurfaceModel` so that existing call sites (`set!`, `model.velocities`,
`model.tracers`, `model.closure`, `model.grid`) continue to work unchanged.

`ocean_simulation` returns an Oceananigans `Simulation`; this wrapper returns `sim.model`,
because this package supplies its own adaptive stepping, checkpointing and JLD2 output
machinery and therefore only needs the model object.

# Physics defaults chosen here, and why they differ from the library defaults
- `equation_of_state = TEOS10EquationOfState(reference_density = 1025)`. The library default
  is `reference_density = 1020`; this package works in the Boussinesq sense with ρ₀ = 1025.
  TEOS-10 is a full equation of state rather than the linearised form used previously, so
  buoyancy and N² differ from the old `LinearEquationOfState` results — this is an accuracy
  improvement, not a like-for-like swap.
- `coriolis` is left at the library default, `HydrostaticSphericalCoriolis`. The previous
  manual builder used `FPlane(latitude = 44.5)`, which freezes f and omits the β term; over a
  42–47.5 °N domain the β term is not negligible, so the spherical form is more appropriate.
- `closure` defaults to `VerticalScalarDiffusivity` with the caller's `ν`/`κ`, matching the
  previous behaviour. Pass `closure = :catke` for the library's prognostic NEMO-TKE closure.
- `free_surface = SplitExplicitFreeSurface(grid; cfl)`. The previous builder used
  `ImplicitFreeSurface`; the split-explicit barotropic solver is cheaper and is what the
  library considers the standard choice.
- `bottom_drag_coefficient` is left at the library default (`0.003`) because `Default` is not
  part of Oceananigans' public API in this build; `cd_drag` is therefore only honoured by the
  manual builder. Set it explicitly via the library kwarg if a different value is needed.

Surface heat/wind fluxes, tidal body forcing and the lateral sponge are *not* configured
here: `ocean_simulation` exposes them through different hooks (surface boundary conditions
and interior `forcing`) than this package's existing `Forcing`/BC construction, and they are
supplied by the caller through `forcing`/`boundary_conditions` below.
"""
function build_hydrodynamic_model(
    grid;
    Δt::Real = 120.0,
    coriolis_latitude::Real = 45.0,
    surface_wind_stress_x = 0.0001,
    surface_wind_stress_y = 0.0,
    surface_heat_flux = 0.0,
    tidal_forcing::Union{Nothing, NamedTuple} = nothing,
    sponge_forcing::Union{Nothing, NamedTuple} = nothing,
    bottom_drag::Real = 1e-4,
    cd_drag::Real = 2.5e-3,
    ν::Real = 1e-2,
    κ::Real = 1e-2,
    closure = nothing,
    closure_scheme = nothing,
    tracers::Tuple = (:T, :S),
    enable_o2::Bool = false,
    reference_density::Real = 1025.0,
    free_surface = nothing,
    momentum_advection = nothing,
    tracer_advection = nothing,
    boundary_conditions::NamedTuple = NamedTuple(),
    forcing::NamedTuple = NamedTuple(),
    open_boundary_conditions = nothing,
    lateral_boundary_relaxation::Bool = false,
    sponge_width::Float64 = 0.35,
    sponge_tau::Float64 = 3600.0,
    tidal_relaxation_timescale::Float64 = 3600.0,
    u_inflow::Float64 = -0.15,
    v_inflow::Float64 = 0.05,
    u_decay::Float64 = 200.0,
    v_decay::Float64 = 500.0,
    lon_range::Union{Nothing, Tuple{<:Real, <:Real}} = nothing,
    lat_range::Union{Nothing, Tuple{<:Real, <:Real}} = nothing,
    boundary_tracers::Union{Nothing, NamedTuple} = nothing,
    boundary_velocities::Union{Nothing, NamedTuple} = nothing,
    mask_bay_of_fundy::Bool = false,
    active_boundaries = (:east, :south, :west)
)
    active_tracers = (enable_o2 && !(:O2 in tracers)) ? (tracers..., :O2) : tracers

    # Lateral relaxation sponge (Price & Aumont 2011).
    #
    # The sponge itself is constructed where the forcing is assembled, because it must be
    # wrapped in a `ContinuousForcing` that declares its velocity dependency. Here we only
    # decide whether it is requested at all, and validate that the caller supplied the
    # study-domain bounds it will be anchored to.
    #
    # The TOML `[domain]` section is the single source of truth for geography; nothing about
    # the domain is compiled into this package.
    sponge_active = lateral_boundary_relaxation || !isnothing(open_boundary_conditions)
    if sponge_active && (isnothing(lon_range) || isnothing(lat_range))
        error(
            "Lateral boundary relaxation requested but no study-domain bounds were " *
            "supplied. Pass the ranges from the TOML `[domain]` section as lon_range " *
            "and lat_range to build_hydrodynamic_model."
        )
    end

    # Conversion factor from W/m^2 to kinematic heat flux (m*K/s): Q / (rho0 * cp)
    # Using reference density rho0 = 1025.0 kg/m^3 and specific heat cp = 3991.0 J/(kg*K)
    rho0_cp = 1025.0 * 3991.0

    # Surface heat flux boundary condition
    # When a temperature-dependent flux function is supplied, e.g. f(x, y, t, T),
    # construct a FluxBoundaryCondition with `field_dependencies = :T` which evaluates
    # dynamically on both CPU and GPU.
    T_flux_bc = if surface_heat_flux isa Function && applicable(surface_heat_flux, 0.0, 0.0, 0.0, 0.0)
        kinematic_flux_T = (x, y, t, T) -> -surface_heat_flux(x, y, t, T) / rho0_cp
        FluxBoundaryCondition(kinematic_flux_T, field_dependencies = :T)
    elseif surface_heat_flux isa Function
        kinematic_T_flux = if applicable(surface_heat_flux, 0.0, 0.0, 0.0, 0.0, 0.0)
            (x, y, z, t, S) -> -surface_heat_flux(x, y, z, t, S) / rho0_cp
        else
            (x, y, t) -> -surface_heat_flux(x, y, t) / rho0_cp
        end
        FluxBoundaryCondition(kinematic_T_flux)
    elseif surface_heat_flux isa AbstractMatrix
        FluxBoundaryCondition(-surface_heat_flux ./ rho0_cp)
    else
        FluxBoundaryCondition(-Float64(surface_heat_flux) / rho0_cp)
    end

    u_bc = surface_wind_stress_x isa BoundaryCondition ? surface_wind_stress_x :
           FluxBoundaryCondition(surface_wind_stress_x)
    v_bc = surface_wind_stress_y isa BoundaryCondition ? surface_wind_stress_y :
           FluxBoundaryCondition(surface_wind_stress_y)
    T_bc = T_flux_bc

    bcs = Dict{Symbol, Any}(
        :u => FieldBoundaryConditions(; top = u_bc),
        :v => FieldBoundaryConditions(; top = v_bc)
    )
    :T in active_tracers && (bcs[:T] = FieldBoundaryConditions(; top = T_bc))
    user_bcs = NamedTuple{Tuple(keys(bcs))}(Tuple(bcs[k] for k in keys(bcs)))

    # Extract grid bounds for sponge layer construction (needs to happen before sponge creation)
    base_g = grid isa ImmersedBoundaryGrid ? grid.underlying_grid : grid
    lon_min_grid = Float64(base_g.λᶠᵃᵃ[1])
    lon_max_grid = Float64(base_g.λᶠᵃᵃ[base_g.Nx + 1])
    lat_min_grid = Float64(base_g.φᵃᶠᵃ[1])
    lat_max_grid = Float64(base_g.φᵃᶠᵃ[base_g.Ny + 1])

    # Interior forcing: tides plus the lateral sponge, matching the manual builder's CPU path.
    # Construct proper bitstype forcing structs that Oceananigans can call in GPU kernels.

    # Interior forcing: tides plus the lateral sponge, matching the manual builder's CPU path.
    # Construct proper bitstype forcing structs that Oceananigans can call in GPU kernels.
    # Extract values from NamedTuples properly.
    u_tide_val = isnothing(tidal_forcing) ? nothing : tidal_forcing.u
    v_tide_val = isnothing(tidal_forcing) ? nothing : tidal_forcing.v

    # Relaxation rate and footprint for the tidal term. See the note where it is applied: the
    # tidal field is a velocity and must be relaxed toward, not forced as an acceleration.
    tidal_relax_rate = 1.0 / Float64(tidal_relaxation_timescale)
    # The tide is applied across the domain. When mask_bay_of_fundy is active, the hypertidal
    # shallows in the Bay of Fundy are masked to zero relaxation weight to prevent forcing
    # unphysical velocity shears near the immersed land boundary.
    tidal_mask = mask_bay_of_fundy ?
        ((x, y, z) -> (is_in_bay_of_fundy(x, y) ? 0.0 : 1.0)) :
        ((x, y, z) -> 1.0)

    # Sponge relaxation, expressed as an Oceananigans `Relaxation`.
    #
    # `Relaxation(; rate, mask, target)` contributes
    #     rate * mask(X) * (target(X, t) - field)
    # which, with `rate = 1/tau_relax` and `mask = gamma`, is exactly the sponge tendency
    # `-gamma * (psi - ref) / tau_relax`. This replaces an earlier
    # `ContinuousForcing(...; field_dependencies = (:u, :v))`, which had to be spelled that way
    # because the closure needed the instantaneous velocity.
    #
    # That declaration is what broke the GPU: a `ContinuousForcing` carrying *any*
    # `field_dependencies` produced invalid LLVM IR in
    # `gpu_compute_hydrostatic_free_surface_Gu!` (Oceananigans 0.111 + CUDA.jl 6), for velocity
    # and tracer dependencies alike. `Relaxation` needs no dependency declaration at all -- it
    # reads the field it relaxes through `r.relaxed`, which the materializer rebinds to the model
    # field and `Adapt`s per device -- so it lowers cleanly.
    #
    # The tidal body forcing is kept as a separate additive term via `MultipleForcings`, so the
    # total is numerically identical to the previous construction:
    #     tide(x, y, z, t) + rate * gamma * (ref - u)
    if sponge_active
        sponge_relaxation = LateralBoundaryRelaxation(
            (lon_min_grid, lon_max_grid), (lat_min_grid, lat_max_grid);
            sponge_width_deg = sponge_width,
            tau_relax = sponge_tau,
            u_inflow = u_inflow,
            v_inflow = v_inflow,
            u_decay = u_decay,
            v_decay = v_decay
        )

        relax_rate = 1.0 / Float64(sponge_tau)
        # Quadratic sponge weight, zero in the interior and 1 on the designated open boundaries.
        # Boundaries not in `active_boundaries` (e.g. northern continental margin) have zero weight.
        act_b_syms = Tuple(Symbol(lowercase(string(b))) for b in active_boundaries)
        act_e = :east in act_b_syms
        act_w = :west in act_b_syms
        act_n = :north in act_b_syms
        act_s = :south in act_b_syms
        sponge_mask = (x, y, z) -> compute_sponge_gamma(
            x, y, lon_min_grid, lon_max_grid, lat_min_grid, lat_max_grid, sponge_width,
            act_e, act_w, act_n, act_s
        )
        # If real boundary velocities (e.g. from GLORYS) are supplied, use them directly as targets.
        # Otherwise, fall back to the analytical ExponentialInflow reference profile.
        u_target = if !isnothing(boundary_velocities) && haskey(boundary_velocities, :u)
            u_val = boundary_velocities.u
            if u_val isa Function
                (x, y, z, t) -> Float64(u_val(x, y, z, t))
            elseif u_val isa AbstractField
                u_val
            else
                ufld = XFaceField(grid)
                set!(ufld, (x, y, z) -> Float64(u_val(x, y, -z)))
                ufld
            end
        else
            (x, y, z, t) -> Float64(sponge_relaxation.u_ref(x, y, z, t))
        end

        v_target = if !isnothing(boundary_velocities) && haskey(boundary_velocities, :v)
            v_val = boundary_velocities.v
            if v_val isa Function
                (x, y, z, t) -> Float64(v_val(x, y, z, t))
            elseif v_val isa AbstractField
                v_val
            else
                vfld = YFaceField(grid)
                set!(vfld, (x, y, z) -> Float64(v_val(x, y, -z)))
                vfld
            end
        else
            (x, y, z, t) -> Float64(sponge_relaxation.v_ref(x, y, z, t))
        end

        u_forcing = Relaxation(rate = relax_rate, mask = sponge_mask, target = u_target)
        v_forcing = Relaxation(rate = relax_rate, mask = sponge_mask, target = v_target)

        # Tidal forcing is a RELAXATION TOWARD the prescribed tidal velocity, not a body force.
        #
        # `tidal_velocity_coefficients` reconstructs the true velocity in m/s from the M2/S2
        # harmonic coefficients: u(t) = sum_k [Re_k cos(w_k t) + Im_k sin(w_k t)].
        # A `ContinuousForcing` on `u`/`v` is an ACCELERATION in m/s^2; passing a velocity there
        # injected `u_tide * dt` into the momentum equation every step, causing severe divergence.
        # Conversely, passing an acceleration tendency into `Relaxation` damped interior
        # velocities to zero.
        #
        # Relaxation toward the reconstructed velocity `(1/tau) * (u_tide - u)` provides the
        # physically consistent formulation: the target is in m/s and converges on timescale `tidal_tau`.
        #
        # A relaxation is the correct expression of a prescribed oscillating velocity: it adds
        # `(1/tau) * (u_tide(x, t) - u)`, so the target is a velocity and the model converges to
        # the tide on the timescale `tidal_relaxation_timescale` rather than accumulating it. That timescale is
        # a physics choice, not a numerical one -- small pins the flow to the tide and suppresses
        # wind and buoyancy; large lets the tide contribute while the rest of the model evolves.
        if !isnothing(u_tide_val)
            u_tide_target = (x, y, z, t) -> Float64(u_tide_val(x, y, z, t))
            u_forcing = MultipleForcings(
                u_forcing,
                Relaxation(rate = tidal_relax_rate, mask = tidal_mask, target = u_tide_target)
            )
        end
        if !isnothing(v_tide_val)
            v_tide_target = (x, y, z, t) -> Float64(v_tide_val(x, y, z, t))
            v_forcing = MultipleForcings(
                v_forcing,
                Relaxation(rate = tidal_relax_rate, mask = tidal_mask, target = v_tide_target)
            )
        end

        composed_forcing = (; u = u_forcing, v = v_forcing)
    else
        # No sponge. If tidal forcing is present, relax toward the prescribed tide;
        # otherwise composed_forcing is empty.
        composed_forcing = NamedTuple()
        if !isnothing(u_tide_val)
            composed_forcing = merge(composed_forcing,
                (; u = Relaxation(rate = tidal_relax_rate, mask = tidal_mask,
                                  target = (x, y, z, t) -> Float64(u_tide_val(x, y, z, t)))))
        end
        if !isnothing(v_tide_val)
            composed_forcing = merge(composed_forcing,
                (; v = Relaxation(rate = tidal_relax_rate, mask = tidal_mask,
                                  target = (x, y, z, t) -> Float64(v_tide_val(x, y, z, t)))))
        end
    end

    # Tracer sponge -- INCOMPLETE, and deliberately so documented here rather than left for
    # the reader to discover from a result.
    #
    # What it does: relaxes T and S inside the sponge band toward an observed climatology.
    #
    # What it does NOT do, which is the important part:
    #
    #   1. It does not constrain momentum. The u/v sponge still relaxes toward the analytic
    #      `u_inflow`/`v_inflow` constants, because WOA23 carries no currents. So the sponge
    #      fixes what the incoming water *is* (temperature, salt) and not how fast it is
    #      *going*. For a domain whose open edges are the main water source, that is half an
    #      open boundary, not a whole one, and a result that depends on inflow is still
    #      resting on two hand-entered numbers.
    #   2. It is time-independent. A climatology is one repeating state, so the boundary has
    #      no tidal, synoptic or seasonal cycle of its own. Eddies and storms arriving from
    #      outside are not represented.
    #   3. It covers T and S only. A model carrying oxygen (`enable_o2`) relaxes the other
    #      tracers not at all.
    #   4. Boundary targets are materialized onto `CenterField(grid)` objects so they
    #      execute natively on both CPU and GPU without run-time interpolation overhead
    #      or non-isbits closure restrictions.
    #
    # Closing any of these needs a source that supplies the missing quantity -- a reanalysis
    # with currents (GLORYS) for (1), a dated series rather than a climatology for (2). It
    # does not need more code here; the code is not the missing part.
    if !isnothing(boundary_tracers) && sponge_active
        # The tracer sponge relaxes the boundary band toward observed temperature and
        # salinity (WOA23 or GLORYS12V1). Rather than passing a host-bound closure capturing
        # an interpolation object (which is not isbits and cannot be lowered into a GPU kernel),
        # the target is materialized onto a `CenterField(grid)`.
        #
        # In Oceananigans, a `Field` target in `Relaxation` is directly adapted to a device
        # array (`CuDeviceArray` on GPU, `Array` on CPU) and indexed as `target[i, j, k]`.
        # This executes natively on both CPU and GPU architectures without run-time
        # interpolation overhead in the tendency kernel.
        for (nm, itp) in pairs(boundary_tracers)
            target_field = CenterField(grid)
            set!(target_field, (x, y, z) -> Float64(itp(x, y, -z)))
            composed_forcing = merge(
                composed_forcing,
                (nm => Relaxation(rate = relax_rate, mask = sponge_mask,
                                  target = target_field),)
            )
        end
    end

    # `composed_forcing` supplies the sponge/tide terms this function is responsible for, but a
    # caller-supplied `u`/`v` forcing must not be silently discarded. Merge so that `composed_forcing`
    # wins only for the components it actually defines and that the caller did not already set.
    # (The previous unconditional `merge(forcing, composed_forcing)` overwrote caller forcing with
    # `ZeroForcing` whenever the sponge was inactive, with no warning.)
    additions = filter(p -> !haskey(forcing, p.first), pairs(composed_forcing))
    merged_forcing = isempty(forcing) ? composed_forcing : merge(forcing, NamedTuple(additions))

    eff_closure = closure !== nothing ? closure : closure_scheme
    vertical_closure = if eff_closure === nothing
        VerticalScalarDiffusivity(VerticallyImplicitTimeDiscretization(); ν = Float64(ν), κ = Float64(κ))
    elseif eff_closure isa Symbol
        eff_closure in (:nemotke, :catke, :tke) ? CATKEVerticalDiffusivity() :
            VerticalScalarDiffusivity(VerticallyImplicitTimeDiscretization(); ν = Float64(ν), κ = Float64(κ))
    else
        eff_closure
    end
    horizontal_closure = HorizontalScalarDiffusivity(ν = 100.0, κ = 100.0)
    active_closure = (vertical_closure, horizontal_closure)

    kwargs = Dict{Symbol, Any}(
        :Δt => Float64(Δt),
        :closure => active_closure,
        :tracers => active_tracers,
        :reference_density => Float64(reference_density),
        :equation_of_state => TEOS10EquationOfState(; reference_density = Float64(reference_density)),
        :bottom_drag_background_velocity => Float64(bottom_drag),
        :boundary_conditions => merge(user_bcs, boundary_conditions),
        :forcing => merged_forcing
    )
    isnothing(free_surface) || (kwargs[:free_surface] = free_surface)
    isnothing(momentum_advection) || (kwargs[:momentum_advection] = momentum_advection)
    isnothing(tracer_advection) || (kwargs[:tracer_advection] = tracer_advection)

    sim = ocean_simulation(grid; model = :hydrostatic, kwargs...)
    return sim.model
end

"""
    apply_reanalysis_initial_conditions!(model; dataset, date, variables) -> model

Initialise prognostic tracers from a reanalysis dataset using `ClimaOcean.Metadatum`.

`Metadatum` is the supported mechanism for injecting observed/reanalysis state into an
Oceananigans model: it carries both the value and its provenance (dataset, date, variable
name) so that archived output remains traceable to the source it came from. This replaces
the previous hand-rolled `interpolate_ocean_state`, which fabricated constant fields when the
dataset could not be read.

# Inputs
- `model`: Model whose tracers are to be initialised.
- `dataset`: A NumericalEarth dataset, e.g. `GLORYSDaily()`, `GLORYSMonthly()`, `WOAMonthly()`.
- `date`: `DateTime` for the initial state.
- `variables`: Tracer symbols to initialise, e.g. `(:T, :S)`.

# Outputs
- The model, with tracers set.

# Notes
If the dataset cannot be fetched (no credentials, no network), the failure is reported
loudly rather than silently substituting constant values, so a run never quietly claims a
reanalysis-forced state it did not have.
"""
function apply_reanalysis_initial_conditions!(
    model;
    dataset,
    date::DateTime,
    variables::Tuple = (:T, :S)
)
    tracer_names = Dict(:temperature => :T, :salinity => :S)
    assignments = Pair{Symbol, Any}[]
    for v in variables
        haskey(tracer_names, v) || continue
        field = tracer_names[v]
        hasproperty(model.tracers, field) || continue
        push!(assignments, field => Metadatum(
            v; date = date, dataset = dataset))
    end
    isempty(assignments) && return model
    set!(model, (assignments..., u = 0.0, v = 0.0))
    return model
end

"""
    woa23_regridded_tracers(grid, temperature_file, salinity_file; oxygen_file=nothing)
        -> (T = (x, y, z) -> …, S = (x, y, z) -> …, O_2 = (x, y, z) -> …)
Read a downloaded WOA23 (or WOA-series) climatology from NetCDF and return **callable
interpolants** of conservative temperature `T` (°C), practical salinity `S` (PSU) and,
optionally, dissolved oxygen `O_2` (µmol kg⁻¹) on the model grid.

The returned callables take grid coordinates `(x, y, z)` in degrees East, degrees North
and metres (negative down) and can be handed straight to `set!(model, T = …, S = …)`, which
is how Oceananigans accepts a spatially varying initial condition.

# Why this exists
`NumericalEarth` can construct `WOAMonthly`/`WOAAnnual` dataset objects, but its
`Column(grid, dataset; …)` constructor does **not** read a dataset — `Column` describes a
point/column *location*. The supported injection path is `ClimaOcean.Metadatum`, which needs
`Downloads.download` to accept a `Metadatum` and so does not give a keyless end-to-end route
from a plain NOAA NCEI file. Since WOA23 is served from an open THREDDS/ERDDAP endpoint that
needs no credentials, reading the file directly and regridding it here keeps the keyless path
real rather than aspirational.

# Interpolation
Horizontal regridding is bilinear in `(lon, lat)` via `Interpolations.jl`; the vertical uses
the dataset's own standard depth levels with linear interpolation in depth, clamped to the
shallowest available level above and the deepest below. Clamping matters: WOA23's deepest
levels are far deeper than the model domain, so an unclamped query would extrapolate.

Values that WOA23 masks to its fill value (WOA uses `-1.0e30`) are replaced by `NaN` before
interpolation and then filled with the horizontally interpolated mean of the valid column, so
land or ice-covered cells do not poison neighbouring water cells with a huge negative bias.

# Inputs
- `grid`: Model grid (an `ImmersedBoundaryGrid` is unwrapped to its underlying grid).
- `temperature_file`, `salinity_file`: Paths to NetCDF files with `t_an` / `s_an`.
- `oxygen_file`: Optional path to a NetCDF file with `o_an`.

# Outputs
- NamedTuple of callables, omitting `O_2` when `oxygen_file` is `nothing` or the model has no
  `O_2` tracer.
"""
function woa23_regridded_tracers(
    grid,
    temperature_file::AbstractString,
    salinity_file::AbstractString;
    oxygen_file::Union{Nothing, AbstractString} = nothing
)
    base_g = grid isa ImmersedBoundaryGrid ? grid.underlying_grid : grid

    T_vals, lon_src, lat_src, dep_src = read_woa_variable(temperature_file, "t_an")
    S_vals, _, _, _ = read_woa_variable(salinity_file, "s_an")

    size(T_vals) == size(S_vals) || error(
        "WOA23 temperature $(size(T_vals)) and salinity $(size(S_vals)) grids disagree; " *
        "the two files must come from the same subset request.")

    T_interp = build_woa_interpolator(T_vals, lon_src, lat_src, dep_src)
    S_interp = build_woa_interpolator(S_vals, lon_src, lat_src, dep_src)

    # The model grid routinely extends outside the downloaded subset (a WOA request may
    # cover only the study box while the grid is larger) and far below the deepest WOA
    # level, so every query is clamped into the dataset's own bounds. Without this the
    # interpolator would throw on out-of-bounds indices; with it, boundary cells take the
    # nearest available value instead of an extrapolation. `read_woa_variable` has already
    # wrapped longitudes into signed degrees East, so longitude needs only clamping.
    #
    # The clamping, the coordinate-to-index conversion and the sign of `z` all live in
    # `_grid_tracer_query`, because they are the three ways a per-source copy of this went
    # wrong; see its docstring.
    out = (T = _grid_tracer_query(T_interp, lon_src, lat_src, dep_src),
           S = _grid_tracer_query(S_interp, lon_src, lat_src, dep_src))

    if !isnothing(oxygen_file) && isfile(oxygen_file)
        O_vals, _, _, _ = read_woa_variable(oxygen_file, "o_an")
        if size(O_vals) == size(T_vals)
            O_interp = build_woa_interpolator(O_vals, lon_src, lat_src, dep_src)
            out = merge(out, (O_2 = _grid_tracer_query(O_interp, lon_src, lat_src, dep_src),))
        else
            @warn "Skipping dissolved oxygen: WOA23 oxygen grid $(size(O_vals)) does not match " *
                  "the temperature/salinity grid $(size(T_vals))."
        end
    end

    return out
end

"""
    _grid_tracer_query(itp, lon, lat, depth) -> (x, y, z) -> value

Interpolate one source field onto the model grid, given an interpolant `itp` over source axes
`lon`, `lat` (degrees East/North) and `depth` (positive down).

Three things happen here, and each was wrong in its own copy when this was written per source:

* the query is **clamped** to the source bounds, because the model grid is routinely larger
  than the subset that was fetched — the alternative is an out-of-bounds throw, or worse, an
  extrapolation nobody asked for;
* physical coordinates are converted to the **interpolator's index coordinates** by a table
  lookup, not a linear rescale — a rescale maps an n-point axis onto [1, 2] rather than [1, n],
  which samples the first column of the file at every cell and looks like a field;
* the model grid's `z` is **negated**, since it is negative-down (0 at the surface) while every
  source axis here is positive-down — without the sign flip every query lands in the top level
  and the stratification collapses.
"""
function _grid_tracer_query(itp, lon, lat, depth)
    lo_z, hi_z = Float64(first(depth)), Float64(last(depth))
    lo_y, hi_y = Float64(first(lat)), Float64(last(lat))
    lo_x, hi_x = Float64(first(lon)), Float64(last(lon))

    to_index(v) = t -> clamp(searchsortedlast(v, Float64(t)), 1, length(v))
    to_ix, to_iy, to_iz = to_index(lon), to_index(lat), to_index(depth)

    return (x, y, z) -> itp(
        to_iz(clamp(-Float64(z), lo_z, hi_z)),
        to_iy(clamp(Float64(y), lo_y, hi_y)),
        to_ix(clamp(Float64(x), lo_x, hi_x))
    )
end

"""
    copernicus_regridded_tracers(grid, file; month = 1) -> (T = …, S = …)

Read a Copernicus Marine subset and return **callable interpolants** of temperature `T` (°C)
and salinity `S` (PSU) on the model grid.

The file is the one `fetch_copernicus_physics_subset` writes: `thetao`/`so` on
`(longitude, latitude, depth, time)` with twelve monthly steps. `month` picks the step.

This is deliberately not a second reader. The variability between hydrography *sources* is
which variable is called what and how the axes are spelled, and `read_woa_variable` already
resolves both — it normalises the depth convention, wraps longitudes, selects the time step
and maps land to NaN. What differs here is only the variable names, so the reader is shared
and this function supplies them. The alternative — a per-provider reader — is how this
package came to hold two WOA23 readers that had drifted apart.
"""
function copernicus_regridded_tracers(
    grid,
    file::AbstractString;
    month::Int = 1
)
    T_vals, lon, lat, dep = read_woa_variable(file, "thetao"; month = month)
    S_vals, _, _, _ = read_woa_variable(file, "so"; month = month)

    size(T_vals) == size(S_vals) || error(
        "Copernicus temperature $(size(T_vals)) and salinity $(size(S_vals)) grids disagree; " *
        "the two variables must come from the same file.")

    T_interp = build_woa_interpolator(T_vals, lon, lat, dep)
    S_interp = build_woa_interpolator(S_vals, lon, lat, dep)

    return (T = _grid_tracer_query(T_interp, lon, lat, dep),
            S = _grid_tracer_query(S_interp, lon, lat, dep))
end

"""
    read_woa_variable(filepath, varname) -> (values, lon, lat, depth)

Read one WOA23 variable plus its coordinate axes from NetCDF.

Returns `values` as a `(depth, lat, lon)` `Array` with all three axes in **ascending** order,
and WOA's missingness mapped to `NaN`.

Real WOA23 files are four-dimensional — `(lon, lat, depth, time)` with a singleton time axis —
so a length-1 trailing axis is dropped here. Land/ice appears as `Missing` in the current release
and as a large negative fill value in older ones; both become `NaN`.

Two convention differences from the model grid are handled here:

* WOA stores depth positive-downward (0, 5, 10 … m); the model grid is negative-downward, so
  depths are negated and re-sorted to ascending (i.e. surface first).
* Longitudes are wrapped into `[-180, 180)` and re-sorted alongside the data. This is a no-op for
  the global WOA23 grid (already `-179.88 … 179.88`) but matters for sector files that store the
  Atlantic as positive longitudes.
"""
function read_woa_variable(filepath::AbstractString, varname::AbstractString; month::Int = 1)
    NCDatasets.NCDataset(filepath, "r") do ds
        haskey(ds, varname) || error(
            "WOA file $(filepath) has no variable '$(varname)'.")

        # Coordinate variable names differ by provider: WOA23 uses lon/lat/depth, GLORYS uses
        # longitude/latitude/depth, and older netCDF uses z/lev. Probing a small alias list is
        # what lets one reader serve every source, which is the point of routing all boundary
        # hydrography through a single entry point.
        pick(names) = begin
            hit = findfirst(n -> haskey(ds, n), names)
            isnothing(hit) && error(
                "None of $(join(names, "/")) is present in $(basename(filepath)); it has " *
                "coordinate variables: $(join(sort(collect(filter(n -> !(n in ("time", "crs", "climatology_bounds")), keys(ds)))), ", ")).")
            names[hit]
        end
        lon_name = pick(("lon", "longitude", "x"))
        lat_name = pick(("lat", "latitude", "y"))
        dep_name = pick(("depth", "z", "lev", "depths"))

        lon_raw = collect(Float64, ds[lon_name][:])
        lat_raw = collect(Float64, ds[lat_name][:])
        dep_raw = collect(Float64, ds[dep_name][:])
        lon_wrapped = [mod(v + 180.0, 360.0) - 180.0 for v in lon_raw]

        # Normalise the depth axis to POSITIVE-down, whatever the file used. WOA23 ships both
        # conventions: the 1-degree files are negative-down (-300..0) and the 0.25-degree files
        # are positive-down (0..5500). Negating unconditionally, as this reader used to, turned
        # the second kind upside down and the returned axis no longer matched the documented
        # contract -- which made the boundary sampler map a query depth to the wrong level and
        # quietly flattened every profile. Deciding from the data rather than the filename means
        # a future re-release that flips the convention again cannot break it.
        dep_max = maximum(abs, dep_raw)
        dep_down = (minimum(dep_raw) >= -0.5 * dep_max) ? dep_raw : -dep_raw
        ip = sortperm(lon_wrapped)
        jp = sortperm(lat_raw)
        kp = sortperm(dep_down)

        lon = lon_wrapped[ip]
        lat = lat_raw[jp]
        depth = dep_down[kp]

        # Real WOA23 files carry a trailing singleton time axis, so the variable arrives as
        # (lon, lat, depth, time) rather than (lon, lat, depth). Accept either and drop a
        # length-1 trailing axis; anything else genuinely ambiguous is an error.
        # Dimension order is verified, not assumed. Permuting a (depth, lat, lon) file as if it
        # were (lon, lat, depth) still produces a correctly-sized array, so the shape check
        # below would pass while every value was scrambled -- which is precisely the class of
        # error that produces plausible-looking nonsense.
        raw = Array(ds[varname])
        dnames = String.(NCDatasets.dimnames(ds[varname]))
        want = (lon_name, lat_name, dep_name)
        if length(dnames) == 4
            time_dim = setdiff(dnames, want)
            length(time_dim) == 1 || error(
                "Variable '$(varname)' has dimensions $(join(dnames, ", ")); expected " *
                "$(join(want, ", ")) plus a single time axis.")
            taxis = findfirst(==(time_dim[1]), dnames)   # position of time, not its name
            ntime = size(raw, taxis)
            (ntime == 1 || month in 1:12) || error(
                "Variable '$(varname)' has $(ntime) time steps; pass month = 1:12 to choose " *
                "one. Refusing to average or to guess, because a boundary target that silently " *
                "averaged twelve months would not be the stated period.")
            tsel = ntime == 1 ? 1 : clamp(month, 1, ntime)
            raw = taxis == 4 ? raw[:, :, :, tsel] : permutedims(raw, (tsel, 4, 1, 2, 3))[1, :, :, :]
        elseif length(dnames) == 3
            sort(dnames) == sort(want) || error(
                "Variable '$(varname)' has dimensions $(join(dnames, ", ")); expected " *
                "$(join(want, ", ")) in any order.")
            # want is (lon, lat, depth); the array is in `dnames` order, so map each requested
            # axis onto the position it actually occupies.
            perm = ntuple(k -> findfirst(==(want[k]), dnames), 3)
            raw = permutedims(raw, perm)
        else
            error("Variable '$(varname)' has $(length(dnames)) dimensions; expected 3 " *
                  "($(join(want, ", "))) or 4 with a time axis.")
        end

        # From here `raw` is (lon, lat, depth). WOA masks land/ice either with a huge negative
        # fill value or, in the current release, with `Missing`; both must become NaN, since a
        # `Missing` left in place would poison every downstream interpolation.
        vals = permutedims(Float64.(coalesce.(raw, NaN)), (3, 2, 1))  # (lon,lat,depth)->(depth,lat,lon)
        vals = vals[kp, jp, ip]

        size(vals) == (length(depth), length(lat), length(lon)) || error(
            "Unexpected WOA variable '$(varname)' with size $(size(vals)); expected " *
            "(depth=$(length(depth)), lat=$(length(lat)), lon=$(length(lon))).")

        vals[vals .<= -1.0e29] .= NaN

        return (vals, lon, lat, depth)
    end
end

"""
    build_woa_interpolator(values, lon, lat, depth) -> (x, y, z) -> Float64

Wrap a `(depth, lat, lon)` array in a trilinear interpolator evaluable at model coordinates.

`values` must already have WOA fill values mapped to `NaN`. Masked cells are repaired first so
Replace masked cells (`NaN`, produced from WOA23's `Missing` land/seabed mask) so that
a query can never return `NaN` for a model cell sitting over land or below the seabed:

* below the deepest valid level in a column, the last valid value is **held**, not averaged;
* an isolated interior gap is filled by linear interpolation between its neighbours;
* a column with no valid depth at all takes the mean of the whole field, never zero.

Holding rather than averaging matters because on a shelf domain most of the grid is masked.
Averaging turned a column valid only in its top ~20 levels into one flat value all the way to
5500 m, and a fully-masked column into literal `0.0` -- that is 0 K and 0 PSU, which in a
TEOS-10 density is a large spurious freshening rather than a neutral filler.

The vertical axis is clamped to the dataset's depth range by the caller, so the interpolator is
only ever queried inside its bounds.
"""
function build_woa_interpolator(values::AbstractArray, lon, lat, depth)
    nz, ny, nx = size(values)

    # Mask repair, one column at a time, and deliberately NOT by averaging.
    #
    # WOA23 masks everything below the seabed and everything inland. On a shelf domain that is
    # most of the grid, so the old policy -- fill each column with its own mean, and fill
    # fully-masked columns with the mean of the whole field -- replaced the real water column
    # with a flat plateau. A shelf column valid only in its top ~20 levels came back
    # "measured" all the way to 5500 m at one averaged temperature, and a column with no valid
    # depths at all came back as literal 0.0, i.e. 0 K and 0 PSU.
    #
    # Neither is a neutral placeholder. A 0 PSU column feeding a TEOS-10 density is a large
    # spurious freshening, and a flat plateau destroys the stratification the grid exists to
    # resolve. So the fill carries the last valid value downward instead:
    #
    #   * below the deepest valid level, hold that value (nearest-valid, not an average);
    #   * a column with no valid depth at all takes the field mean, never zero;
    #   * an isolated interior gap is filled by linear interpolation between its neighbours.
    valid_all = filter(!isnan, values)
    isempty(valid_all) && error(
        "build_woa_interpolator: the field contains no valid values at all. It is empty or " *
        "entirely masked, and there is nothing to fill from.")
    field_mean = sum(valid_all) / length(valid_all)

    for j in 1:ny, i in 1:nx
        col = @view values[:, j, i]
        ok = findall(!isnan, col)
        if isempty(ok)
            col .= field_mean
            continue
        end
        first_ok, last_ok = first(ok), last(ok)
        # Below the seabed: hold the deepest observation.
        for k in (last_ok + 1):nz
            col[k] = col[last_ok]
        end
        # Interior gaps: linear between the surrounding valid levels.
        for k in (first_ok + 1):(last_ok - 1)
            isnan(col[k]) || continue
            lo = k - 1
            while lo >= first_ok && isnan(col[lo])
                lo -= 1
            end
            hi = k + 1
            while hi <= last_ok && isnan(col[hi])
                hi += 1
            end
            col[k] = col[lo] + (col[hi] - col[lo]) * (k - lo) / (hi - lo)
        end
    end

    # `values` is (depth, lat, lon) with each axis ascending, which is the order
    # `interpolate` expects for the returned array.
    return interpolate(values, (BSpline(Linear()), BSpline(Linear()), BSpline(Linear())))
end

"""
    _clamped_bracket(axis, x) -> (i0, i1, f)

Locate `x` on a strictly ascending `axis`, returning the two bracketing indices and the
fraction between them. Outside the axis the endpoints are held, so a query beyond the data
clamps to the edge rather than extrapolating or throwing. Clamping is the right choice at an
open boundary: the nearest value the dataset has is a far better answer than a linear
continuation off the end of it, and `searchsortedlast` on an out-of-range `x` would otherwise
index past the array.
"""
function _clamped_bracket(axis::AbstractVector{<:Real}, x::Real)
    n = length(axis)
    n == 1 && return (1, 1, 0.0)
    xf = Float64(x)
    xf <= axis[1] && return (1, 1, 0.0)
    xf >= axis[n] && return (n, n, 0.0)
    i = clamp(searchsortedlast(axis, xf), 1, n - 1)
    a, b = Float64(axis[i]), Float64(axis[i+1])
    return (i, i + 1, b == a ? 0.0 : (xf - a) / (b - a))
end

"""
    _trilinear(vals, ax, ay, az, x, y, z) -> Float64

Trilinear sample of `vals` (ordered depth, latitude, longitude) at physical coordinates.
Written out rather than delegating to `Interpolations` because that package indexes by array
position, not by physical coordinate, and mixing the two silently returns a plausible number
from the wrong place.
"""
function _trilinear(vals::AbstractArray{<:Real, 3}, ax, ay, az, x, y, z)
    i0, i1, fx = _clamped_bracket(ax, x)
    j0, j1, fy = _clamped_bracket(ay, y)
    k0, k1, fz = _clamped_bracket(az, z)
    c000 = @inbounds vals[k0, j0, i0]; c100 = @inbounds vals[k0, j0, i1]
    c010 = @inbounds vals[k0, j1, i0]; c110 = @inbounds vals[k0, j1, i1]
    c001 = @inbounds vals[k1, j0, i0]; c101 = @inbounds vals[k1, j0, i1]
    c011 = @inbounds vals[k1, j1, i0]; c111 = @inbounds vals[k1, j1, i1]
    c00 = c000 + fx * (c100 - c000); c10 = c010 + fx * (c110 - c010)
    c01 = c001 + fx * (c101 - c001); c11 = c011 + fx * (c111 - c011)
    c0 = c00 + fy * (c10 - c00); c1 = c01 + fy * (c11 - c01)
    return c0 + fz * (c1 - c0)
end

"""
    _check_boundary_field_is_real(path, vals, what)

Refuse a boundary field that is structurally incapable of being an ocean.

This exists because a boundary file once came out of the download path with 5 depth levels
and no horizontal variation whatsoever: T = 14.0 and S = 31.5 at every point of an 18 x 8.5
degree box. It parsed cleanly, interpolated cleanly, and would have relaxed the sponge band
toward a constant while every downstream number looked entirely reasonable. Nothing about the
*use* of that file reveals the problem, so it has to be caught at the point of loading.

Two independent checks, because either failure alone is enough to make the field useless and
they fail for different reasons:

* **Depth coverage.** A real WOA23 column has 102 standard levels to 5500 m. A handful of
  levels means the column was truncated, and since the sampler clamps, the model would then
  treat the deepest level's value as applying all the way to the seabed.
* **Horizontal variation.** Real temperature and salinity at a fixed depth differ from place
  to place. A field that is constant across the domain is a stub or a default, and a sponge
  built from it is not constraining anything.

The thresholds are deliberately loose. They are here to catch a broken file, not to
second-guess a legitimately uniform region.
"""
function _check_boundary_field_is_real(path::AbstractString, vals::AbstractArray, what::AbstractString)
    nz, ny, nx = size(vals)
    nz < 20 && error(
        "Boundary field $(what) in $(basename(path)) has only $(nz) depth level(s). A real " *
        "ocean climatology has tens of levels spanning to the seabed; this file is truncated. " *
        "Since the sampler clamps at the ends, a truncated column would have its deepest value " *
        "applied to the whole water column below it. Refusing rather than using it.")

    # Surface layer only: the whole-column spread would be dominated by the vertical
    # gradient and would not detect a horizontally constant field.
    surface = @view vals[1, :, :]
    spread = maximum(surface) - minimum(surface)
    spread < 1.0e-3 && error(
        "Boundary field $(what) in $(basename(path)) is identical at every point of the " *
        "domain at the surface (spread $(round(spread, digits=6))). A real climatology varies " *
        "horizontally, so this file is a stub or a default rather than data. Refusing rather " *
        "than relaxing the sponge toward a constant.")
    return nothing
end

"""
    _temperature_to_celsius_offset(units) -> Float64

Offset required to convert a NetCDF temperature `units` string to Celsius (°C):
0.0 for an in-situ or potential temperature scale already in Celsius (°C),
-273.15 for absolute temperature (Kelvin), and an error for anything else.

Oceananigans hydrostatic models with TEOS-10 equation of state store conservative
temperature in degrees Celsius (°C), matching WOA23 climatological hydrography (`t_an`).
Boundary hydrography sources in Celsius (such as GLORYS12V1 `thetao` and WOA23 `t_an`)
require no offset (0.0). Sources provided in Kelvin require subtracting 273.15.
"""
function _temperature_to_celsius_offset(units::AbstractString)
    u = lowercase(strip(units))
    isempty(u) && error("Boundary temperature has no `units` attribute.")
    (occursin("kelvin", u) || endswith(u, " k")) && return -273.15
    (occursin("celsius", u) || occursin(r"degree[s]?\s*_?\s*c\b", u)) && return 0.0
    error(
        "Boundary temperature has units \"$(units)\", which is neither Kelvin nor Celsius. " *
        "Refusing rather than guessing, because an unhandled unit would make the sponge " *
        "relax the open edges to an erroneous temperature."
    )
end

# Backward compatibility alias
const _temperature_offset_k = _temperature_to_celsius_offset


"""
    build_boundary_tracer_interpolators(T_path, S_path; T_name, S_name)

Build `(; T, S)` boundary targets from observed hydrography, for the lateral sponge.

Both file paths are explicit and both are required, because the sources disagree about how
they are packaged: WOA23 publishes temperature and salinity in separate files, GLORYS packs
both into one, and HYCOM puts them in one file under different names. Accepting a single
optional path and guessing the other is how this ended up with two different readers and a
truncated file nobody noticed. A caller now has to say where each field lives, and the grids
are checked to match before anything is returned.

Returns two callables, each `f(lon, lat, depth)` with **positive-down** depth. The WOA reader
negates its source axis, so this comes out positive-down; `build_hydrodynamic_model` negates
`z` before calling, so the sponge sees the model's negative-down coordinate.

This is what gives a keyless open edge real water in it. WOA23 is a monthly climatology
averaged over many years, so it describes an average boundary rather than a particular day;
it is a large improvement on unconstrained edges and a modest one on a dated reanalysis.

**It is an incomplete open boundary, not a complete one.** It constrains temperature and
salinity only. WOA23 carries no currents, so the inflow *velocity* still comes from the
configured `u_inflow`/`v_inflow` constants; and because a climatology is one repeating state,
the boundary has no tidal or synoptic cycle of its own. Any result that depends on how much
water crosses an open edge is still resting on those constants. A source with currents
(GLORYS) is what actually closes that, and it needs an account.
"""
function build_boundary_tracer_interpolators(T_path::AbstractString,
                                              S_path::AbstractString;
                                              T_name::AbstractString = "t_an",
                                              S_name::AbstractString = "s_an",
                                              month::Int = 1,
                                              months::Union{Nothing, Vector{Int}} = nothing)
    # `months` averages several monthly fields into one. This is what an "annual" boundary
    # means, and it matters: the model interior is initialised from the ANNUAL WOA23
    # climatology when `hydrography_month = 0`, so a boundary that quietly became JANUARY
    # (as this did, via `month == 0 ? 1 : month`) is a different physical state from the one
    # the interior starts in. Measured at the eastern open boundary the surface difference
    # was -7.9 K, with a band-wide mean of -1.8 K, injected as a persistent density front
    # across a 0.35-degree sponge: that is what drove max(|w|) to ~1.7 m/s. An annual mean of
    # the same product is consistent with the annual initial condition by construction.
    mlist = isnothing(months) ? [month] : months
    mean_over_months(path, name) = begin
        acc = nothing
        cnt = nothing
        lo = la = de = nothing
        for m in mlist
            V, lo_m, la_m, de_m = read_woa_variable(path, name; month = m)
            if acc === nothing
                (acc, cnt, lo, la, de) = (copy(V), Float64.(isfinite.(V)), lo_m, la_m, de_m)
            else
                (lo_m == lo && la_m == la && de_m == de) || error(
                    "Monthly fields for $(name) are on different grids; month $(m) does not " *
                    "match the others, so they cannot be averaged.")
                acc .+= ifelse.(isfinite.(V), V, 0.0)
                cnt .+= ifelse.(isfinite.(V), 1.0, 0.0)
            end
        end
        V = ifelse.(cnt .> 0, acc ./ max.(cnt, 1.0), NaN)
        (V, lo, la, de)
    end

    Tv, lon, lat, depth = mean_over_months(T_path, T_name)
    Sv, lon_s, lat_s, depth_s = mean_over_months(S_path, S_name)
    (lon_s == lon && lat_s == lat && depth_s == depth) || error(
        "Boundary temperature and salinity are on different grids: $(basename(T_path)) has " *
        "$(length(lon))x$(length(lat))x$(length(depth)) and $(basename(S_path)) has " *
        "$(length(lon_s))x$(length(lat_s))x$(length(depth_s)). They are combined by trilinear " *
        "sampling on one axis set, so they must agree.")

    _check_boundary_field_is_real(T_path, Tv, T_name)
    _check_boundary_field_is_real(S_path, Sv, S_name)

    t_offset = NCDatasets.NCDataset(T_path, "r") do ds
        haskey(ds, T_name) || error("$(basename(T_path)) has no variable $(T_name).")
        _temperature_to_celsius_offset(get(ds[T_name].attrib, "units", ""))
    end

    # Land mask repair.
    #
    # WOA23 marks land with `Missing`, which the reader turns into NaN, and a shelf domain
    # like the Scotian Shelf is bordered by masked cells. `_trilinear` returns NaN if *any*
    # of its eight corners is masked, so a single masked neighbour anywhere in the sponge band
    # would propagate NaN into the model -- and a NaN in a `Relaxation` target is not a small
    # error, it poisons the run.
    #
    # Two stages, in this order:
    #   1. each column is filled from its own valid depths, so a coastal column keeps the
    #      profile it does have rather than being flattened to a global average;
    #   2. columns masked over their entire depth range carry no information at all, so they
    #      take the mean of the whole field.
    for V in (Tv, Sv)
        global_valid = filter(!isnan, V)
        isempty(global_valid) && error(
            "Boundary hydrography contains no valid values at all; the source file is empty " *
            "or entirely masked.")
        global_mean = sum(global_valid) / length(global_valid)
        for j in axes(V, 2), i in axes(V, 3)
            col = @view V[:, j, i]
            bad = findall(isnan, col)
            isempty(bad) && continue
            valid = filter(!isnan, col)
            col[bad] .= isempty(valid) ? global_mean : sum(valid) / length(valid)
        end
end

    Tf = (x, y, z) -> _trilinear(Tv, lon, lat, depth, x, y, z) + t_offset
    Sf = (x, y, z) -> _trilinear(Sv, lon, lat, depth, x, y, z)
    return (; T = Tf, S = Sf)
end

"""
    seawater_freezing_temperature(S::Real) -> Float64

Calculate seawater freezing temperature in °C as a function of practical salinity \$S\$ (PSU)
using the UNESCO / Millero (1978) thermodynamic formulation:
```math
T_{\\text{freeze}}(S) = -0.0575 S + 0.00171 S^{1.5} - 0.000215 S^2
```
"""
@inline function seawater_freezing_temperature(S::Real)::Float64
    s_val = Float64(S)
    if s_val < 0.0
        return 0.0
    end
    return -0.0575 * s_val + 0.00171 * (s_val^1.5) - 0.000215 * (s_val^2)
end

"""
    LateralBoundaryRelaxation{TF}

Bitstype lateral boundary sponge relaxation forcing following Price & Aumont (2011)
and Arctic Ocean shelf modeling implementations. Rather than imposing artificial localized
surface wind stress to drive regional shelf circulation, lateral boundary relaxation nudges
prognostic velocities \$(u, v)\$ and active tracers \$(T, S, O_2)\$ towards upstream and open ocean
hydrographic boundary conditions within peripheral sponge zones.

# Mathematical & Governing Formulation
```math
F_{\\text{relax}}(x, y, z, t, \\psi) = -\\frac{\\gamma(x, y)}{\\tau_{\\text{relax}}} \\left(\\psi - \\psi_{\\text{ref}}(x, y, z, t)\\right)
```
where the sponge relaxation weight \$\\gamma(x, y) \\in [0, 1]\$ increases smoothly (quadratically)
from 0.0 at the interior edge of the sponge layer to 1.0 at the outer domain boundary.

# References
- Price, J. F., & Aumont, O. (2011). Lateral boundary conditions and regional shelf circulation.
  *Journal of Physical Oceanography*, 41(5), 903-921.
"""
struct LateralBoundaryRelaxation{TU, TV}
    lon_domain       :: Tuple{Float64, Float64}
    lat_domain       :: Tuple{Float64, Float64}
    sponge_width_deg :: Float64
    tau_relax        :: Float64
    u_inflow         :: Float64
    v_inflow         :: Float64
    u_ref            :: TU
    v_ref            :: TV
end

"""
    ExponentialInflow{T}

Bitstype reference profile for open-boundary inflow, \$\\psi_{ref}(z) = A \\exp(z / d)\$.

A concrete struct is used rather than a closure so that the sponge forcing built from it
stays `isbits`. A closure capturing `A` and `d` is *not* `isbits`, which makes the enclosing
`ContinuousForcing` unlowerable into a GPU kernel and triggers
`KernelError: passing non-bitstype argument`.
"""
struct ExponentialInflow{T <: AbstractFloat}
    amplitude::T
    decay    :: T
end

@inline (f::ExponentialInflow)(x, y, z, t) = f.amplitude * exp(z / f.decay)

"""
    LateralBoundaryRelaxation(lon_domain, lat_domain; kwargs...)

Construct a regional lateral boundary relaxation sponge zone following Price & Aumont (2011).

# Arguments
- `lon_domain::Tuple{Real, Real}`: Zonal domain boundaries (lon_min, lon_max) in °E.
- `lat_domain::Tuple{Real, Real}`: Meridional domain boundaries (lat_min, lat_max) in °N.

# Keyword Arguments
- `sponge_width_deg::Real=0.5`: Width of peripheral buffer sponge layer in degrees.
- `tau_relax::Real=86400.0`: Relaxation timescale \$\\tau\$ in seconds (default 1 day).
- `u_inflow::Real=-0.15`: Reference zonal upstream inflow velocity in m/s.
- `v_inflow::Real=0.05`: Reference meridional upstream inflow velocity in m/s.
- `u_decay::Real=100.0`: e-folding depth of the zonal reference profile in metres.
- `v_decay::Real=100.0`: e-folding depth of the meridional reference profile in metres.

# Notes
`u_ref`/`v_ref` default to [`ExponentialInflow`](@ref) values rather than closures so the
result is `isbits` and can be compiled for GPU.
"""
function LateralBoundaryRelaxation(
    lon_domain::Tuple{Real, Real},
    lat_domain::Tuple{Real, Real};
    sponge_width_deg::Real = 0.5,
    tau_relax::Real = 86400.0,
    u_inflow::Real = -0.15,
    v_inflow::Real = 0.05,
    u_decay::Real = 100.0,
    v_decay::Real = 100.0
)
    return LateralBoundaryRelaxation(
        (Float64(lon_domain[1]), Float64(lon_domain[2])),
        (Float64(lat_domain[1]), Float64(lat_domain[2])),
        Float64(sponge_width_deg),
        Float64(tau_relax),
        Float64(u_inflow),
        Float64(v_inflow),
        ExponentialInflow(Float64(u_inflow), Float64(u_decay)),
        ExponentialInflow(Float64(v_inflow), Float64(v_decay))
    )
end

@inline function _sponge_relaxation(r::LateralBoundaryRelaxation, ref, x, y, z, t, psi::Real)
    gamma = compute_sponge_gamma(
        x, y,
        r.lon_domain[1], r.lon_domain[2],
        r.lat_domain[1], r.lat_domain[2],
        r.sponge_width_deg
    )
    if gamma <= 0.0
        return 0.0
    end
    return -gamma * (Float64(psi) - Float64(ref(x, y, z, t))) / r.tau_relax
end

"""
    sponge_relaxation_u(r, x, y, z, t, psi) -> Float64

Zonal (\$u\$) sponge relaxation tendency, with no runtime selector argument.

This is the entry point the GPU forcing path uses. It deliberately takes only
`isbitstype` arguments: a `Symbol` cannot be passed through a CUDA kernel ABI, so a
`Symbol`-dispatched method makes the enclosing `ContinuousForcing` unlowerable and the
free-surface kernel fails to compile with `InvalidIRError`. Keeping the two components as
separate methods keeps the kernel body free of non-bits values.
"""
@inline function sponge_relaxation_u(r::LateralBoundaryRelaxation, x, y, z, t, psi::Real)
    return _sponge_relaxation(r, r.u_ref, x, y, z, t, psi)
end

"""
    sponge_relaxation_v(r, x, y, z, t, psi) -> Float64

Meridional (\$v\$) sponge relaxation tendency; see [`sponge_relaxation_u`](@ref) for why this
is separate rather than `Symbol`-dispatched.
"""
@inline function sponge_relaxation_v(r::LateralBoundaryRelaxation, x, y, z, t, psi::Real)
    return _sponge_relaxation(r, r.v_ref, x, y, z, t, psi)
end

# Runtime-selected forms, retained for host-side and test callers. These are deliberately
# not used to build GPU forcing: the `Symbol` argument is what makes them unlowerable.
@inline function (r::LateralBoundaryRelaxation)(x, y, z, t, var::Symbol, psi::Real)
    return if var === :u
        sponge_relaxation_u(r, x, y, z, t, psi)
    elseif var === :v
        sponge_relaxation_v(r, x, y, z, t, psi)
    else
        0.0
    end
end

@inline (r::LateralBoundaryRelaxation)(x, y, z, t, psi) = sponge_relaxation_u(r, x, y, z, t, psi)
@inline (r::LateralBoundaryRelaxation)(x, y, z, t) = 0.0

"""
    compute_sponge_gamma(
        x, y, lon_min, lon_max, lat_min, lat_max, sponge_width;
        active_east = true, active_west = true,
        active_north = true, active_south = true
    ) -> Float64

Compute the quadratic sponge relaxation weight γ ∈ [0, 1] for multi-boundary
relaxation following Price & Aumont (2011). Evaluates only open boundaries designated
by active flags, avoiding unphysical inward momentum relaxation against closed
or land-bounded margins.
"""
@inline function compute_sponge_gamma(
    x::Real, y::Real,
    lon_min::Real, lon_max::Real,
    lat_min::Real, lat_max::Real,
    sponge_width::Real,
    active_east::Bool,
    active_west::Bool,
    active_north::Bool,
    active_south::Bool
)::Float64
    # Eastern boundary (x → lon_max)
    gamma_e = if !active_east || x <= (lon_max - sponge_width)
        0.0
    elseif x >= lon_max
        1.0
    else
        ((x - (lon_max - sponge_width)) / sponge_width)^2
    end

    # Western boundary (x → lon_min)
    gamma_w = if !active_west || x >= (lon_min + sponge_width)
        0.0
    elseif x <= lon_min
        1.0
    else
        (((lon_min + sponge_width) - x) / sponge_width)^2
    end

    # Northern boundary (y → lat_max)
    gamma_n = if !active_north || y <= (lat_max - sponge_width)
        0.0
    elseif y >= lat_max
        1.0
    else
        ((y - (lat_max - sponge_width)) / sponge_width)^2
    end

    # Southern boundary (y → lat_min)
    gamma_s = if !active_south || y >= (lat_min + sponge_width)
        0.0
    elseif y <= lat_min
        1.0
    else
        (((lat_min + sponge_width) - y) / sponge_width)^2
    end

    return max(gamma_e, gamma_w, gamma_n, gamma_s)
end

@inline function compute_sponge_gamma(
    x::Real, y::Real,
    lon_min::Real, lon_max::Real,
    lat_min::Real, lat_max::Real,
    sponge_width::Real
)::Float64
    return compute_sponge_gamma(
        x, y, lon_min, lon_max, lat_min, lat_max, sponge_width,
        true, true, true, true
    )
end

"""
    compute_sponge_velocity_target(u_inflow_mean, v_inflow_mean, u_decay, v_decay, z) -> Tuple{Float64, Float64}

Compute the target velocity for sponge relaxation with exponential depth decay.
"""
@inline function compute_sponge_velocity_target(
    u_inflow_mean::Real, v_inflow_mean::Real,
    u_decay::Real, v_decay::Real,
    z::Real
)::Tuple{Float64, Float64}
    u_target = Float64(u_inflow_mean) * exp(Float64(z) / Float64(u_decay))
    v_target = Float64(v_inflow_mean) * exp(Float64(z) / Float64(v_decay))
    return (u_target, v_target)
end

"""
    HorizontalMomentumForcingU{TF}

Bitstype callable struct for zonal momentum forcing on GPU/CPU without field dependencies.
Combines depth-tapered harmonic tidal body forcing, lateral sponge relaxation,
and Patankar-type quasi-implicit quadratic bottom drag.
"""
struct HorizontalMomentumForcingU{TF}
    tidal          :: TF
    has_sponge     :: Bool
    lon_min        :: Float64   # western domain boundary (°E)
    lon_max        :: Float64   # eastern domain boundary (°E)
    lat_min        :: Float64   # southern domain boundary (°N)
    lat_max        :: Float64   # northern domain boundary (°N)
    sponge_width   :: Float64   # sponge buffer width (degrees)
    sponge_tau     :: Float64   # relaxation timescale (s)
    u_inflow_mean  :: Float64   # reference zonal inflow velocity (m/s)
    u_inflow_decay :: Float64   # depth e-folding scale (m)
    bottom_drag    :: Float64
    cd_drag        :: Float64
end

"""
    (m::HorizontalMomentumForcingU)(x, y, z, t, u, v)

Zonal momentum forcing: tidal body force + multi-boundary quadratic sponge relaxation
(Price & Aumont 2011) + Patankar-implicit quadratic bottom drag.

Sponge gamma is the maximum of all four boundary ramp functions, ramping quadratically
from 0 at the inner sponge edge to 1 at the domain wall. This is identical to the
formulation in `LateralBoundaryRelaxation` and avoids ad-hoc sine-taper momentum
discontinuities that cause CFL blow-up near immersed boundaries.
"""
@inline function (m::HorizontalMomentumForcingU)(x, y, z, t, u, v)
    sponge_val = 0.0
    if m.has_sponge
        gamma = compute_sponge_gamma(
            x, y,
            m.lon_min, m.lon_max,
            m.lat_min, m.lat_max,
            m.sponge_width
        )
        if gamma > 0.0
            u_target = m.u_inflow_mean * exp(Float64(z) / m.u_inflow_decay)
            sponge_val = -gamma * (Float64(u) - u_target) / m.sponge_tau
        end
    end

    tide_val = m.tidal(x, y, z, t)

    speed = sqrt(u^2 + v^2)
    tau_ref = 120.0
    drag_coeff = (m.bottom_drag + m.cd_drag * speed) /
                 (1.0 + (m.bottom_drag + m.cd_drag * speed) * tau_ref)
    drag_val = -drag_coeff * Float64(u)
    return tide_val + sponge_val + drag_val
end

@inline (m::HorizontalMomentumForcingU)(x, y, z, t, u) = m(x, y, z, t, u, 0.0)
@inline (m::HorizontalMomentumForcingU)(x, y, z, t) = m(x, y, z, t, 0.0, 0.0)

"""
    HorizontalMomentumForcingV{TF}

Bitstype callable struct for meridional momentum forcing on GPU/CPU.
Combines harmonic tidal body forcing, multi-boundary quadratic sponge relaxation
(Price & Aumont 2011), and Patankar-type quasi-implicit quadratic bottom drag.
No sine wall tapering is applied to tidal forcing; relaxation covers all four
open boundaries symmetrically.
"""
struct HorizontalMomentumForcingV{TF}
    tidal          :: TF
    has_sponge     :: Bool
    lon_min        :: Float64   # western domain boundary (°E)
    lon_max        :: Float64   # eastern domain boundary (°E)
    lat_min        :: Float64   # southern domain boundary (°N)
    lat_max        :: Float64   # northern domain boundary (°N)
    sponge_width   :: Float64   # sponge buffer width (degrees)
    sponge_tau     :: Float64   # relaxation timescale (s)
    v_inflow_mean  :: Float64   # reference meridional inflow velocity (m/s)
    v_inflow_decay :: Float64   # depth e-folding scale (m)
    bottom_drag    :: Float64
    cd_drag        :: Float64
end

@inline function (m::HorizontalMomentumForcingV)(x, y, z, t, u, v)
    sponge_val = 0.0
    if m.has_sponge
        gamma = compute_sponge_gamma(
            x, y,
            m.lon_min, m.lon_max,
            m.lat_min, m.lat_max,
            m.sponge_width
        )
        if gamma > 0.0
            v_target = m.v_inflow_mean * exp(Float64(z) / m.v_inflow_decay)
            sponge_val = -gamma * (Float64(v) - v_target) / m.sponge_tau
        end
    end

    tide_val = m.tidal(x, y, z, t)

    speed = sqrt(u^2 + v^2)
    tau_ref = 120.0
    drag_coeff = (m.bottom_drag + m.cd_drag * speed) /
                 (1.0 + (m.bottom_drag + m.cd_drag * speed) * tau_ref)
    drag_val = -drag_coeff * Float64(v)
    return tide_val + sponge_val + drag_val
end

@inline (m::HorizontalMomentumForcingV)(x, y, z, t, v) = m(x, y, z, t, 0.0, v)
@inline (m::HorizontalMomentumForcingV)(x, y, z, t) = m(x, y, z, t, 0.0, 0.0)

"""
    set_initial_stratification!(model;

        surface_temperature::Union{Nothing, Real} = nothing,
        surface_temp::Union{Nothing, Real} = nothing,
        bottom_temperature::Union{Nothing, Real} = nothing,
        bottom_temp::Union{Nothing, Real} = nothing,
        cil_temperature::Union{Nothing, Real} = nothing,
        cil_temp::Union{Nothing, Real} = nothing,
        slope_temperature::Union{Nothing, Real} = nothing,
        slope_temp::Union{Nothing, Real} = nothing,
        temperature_gradient::Real = 0.01,
        temp_stratification::Union{Nothing, Function} = nothing,
        salinity::Union{Nothing, Real, Function} = 33.0,
        lon_range::Union{Nothing, Tuple} = nothing,
        lat_range::Union{Nothing, Tuple} = nothing,
    )

Set physically plausible initial `T` and `S` profiles on `model`.

Temperature is built from up to four overlapping components — a surface value, a
bottom value, a cold-intermediate-layer (CIL) minimum, and a continental-slope
minimum — with `temperature_gradient` setting the thermocline sharpness. Salinity is
constant unless a function is supplied.

All components are optional: with none given, a linear profile from
`surface_temperature` (default 14 °C) to `bottom_temperature` (default 4 °C) at
`temperature_gradient` is used, giving a stably stratified column.

# Notes
- The `temperature_*`/`temp_*` and `*_temp` spellings are accepted as aliases.
- Domain bounds, when supplied, are used to taper the shelf-slope component so it
  weakens outside the configured `lon_range`/`lat_range`.
- Values are broadcast across the horizontal grid, so the resulting field is
  horizontally uniform apart from the slope taper.
"""
function set_initial_stratification!(
    model;
    surface_temperature::Union{Nothing, Real} = nothing,
    surface_temp::Union{Nothing, Real} = nothing,
    bottom_temperature::Union{Nothing, Real} = nothing,
    bottom_temp::Union{Nothing, Real} = nothing,
    cil_temperature::Union{Nothing, Real} = nothing,
    cil_temp::Union{Nothing, Real} = nothing,
    slope_temperature::Union{Nothing, Real} = nothing,
    slope_temp::Union{Nothing, Real} = nothing,
    temperature_gradient::Union{Nothing, Real} = nothing,
    temp_stratification::Union{Nothing, Real} = nothing,
    salinity::Union{Real, Function} = 33.0,
    lon_range::Union{Nothing, Tuple} = nothing,
    lat_range::Union{Nothing, Tuple} = nothing,
    stratification_type::Symbol = :three_layer,
    kwargs...
)
    # Resolve domain bounds
    lon_min, lon_max = isnothing(lon_range) ? (-71.0, -53.0) : Float64.(lon_range)
    lat_min, lat_max = isnothing(lat_range) ? (40.0, 48.5) : Float64.(lat_range)

    if lon_min >= lon_max || lat_min >= lat_max
        error("Invalid domain bounds: lon_range=$(lon_range), lat_range=$(lat_range)")
    end

    # Resolve surface temperature (default 14.0 °C)
    T0 = if !isnothing(surface_temperature)
        Float64(surface_temperature)
    elseif !isnothing(surface_temp)
        Float64(surface_temp)
    else
        14.0
    end

    # Resolve vertical temperature gradient (default 0.01 °C/m)
    dT_dz = if !isnothing(temperature_gradient)
        Float64(temperature_gradient)
    elseif !isnothing(temp_stratification)
        Float64(temp_stratification)
    else
        0.01
    end

    # Resolve bottom / CIL / slope water temperatures
    b_temp = if !isnothing(bottom_temperature)
        Float64(bottom_temperature)
    elseif !isnothing(bottom_temp)
        Float64(bottom_temp)
    else
        nothing
    end

    T_cil_min = if !isnothing(cil_temperature)
        Float64(cil_temperature)
    elseif !isnothing(cil_temp)
        Float64(cil_temp)
    elseif !isnothing(b_temp) && b_temp <= 5.0
        Float64(b_temp)
    else
        1.5
    end

    T_slope = if !isnothing(slope_temperature)
        Float64(slope_temperature)
    elseif !isnothing(slope_temp)
        Float64(slope_temp)
    elseif !isnothing(b_temp) && b_temp > 5.0
        Float64(b_temp)
    else
        8.5
    end

    # Horizontal gradient scales (°C) across the Scotian Shelf
    ΔT_cross = 8.0
    ΔT_along = 5.0
    ΔT_cil   = 2.5
    T_cil_edge = T_cil_min + ΔT_cil

    # Salinity gradient scales (PSU)
    S0       = Float64(salinity isa Real ? salinity : 33.0)
    ΔS_cross = 1.5
    ΔS_deep  = 2.5

    # Deep water relaxation scale (m)
    h_scale = (dT_dz > 0.0 && T_slope > T_cil_edge) ?
        max(50.0, min(500.0, (T_slope - T_cil_edge) / dT_dz)) : 100.0

    function temp_profile(lon, lat, z)
        dx = lon_max - lon_min
        dy = lat_max - lat_min
        x_norm = max(0.0, min(1.0, (lon - lon_min) / dx))
        y_norm = max(0.0, min(1.0, (lat - lat_min) / dy))
        δT_h = ΔT_cross * x_norm - ΔT_along * y_norm

        T_surf = T0 + δT_h

        if stratification_type == :linear
            return T_surf + dT_dz * z
        end

        # Three-layer Northwest Atlantic / Scotian Shelf structure (Petrie & Drinkwater 1993)
        # Cold Intermediate Layer core & edge with horizontal gradient modulation
        T_cil_min_local = T_cil_min + 0.40 * δT_h
        T_cil_edge_local = T_cil_min_local + ΔT_cil

        if z > -20.0
            # Surface mixed layer thermocline
            frac = (z + 20.0) / 20.0
            return T_cil_edge_local + frac * (T_surf - T_cil_edge_local)
        elseif z > -80.0
            # Cold Intermediate Layer (CIL): parabolic minimum centered at z = -50 m
            centre_frac = (z + 50.0) / 30.0
            return T_cil_min_local + ΔT_cil * centre_frac^2
        else
            # Deep Warm Slope Water: smooth exponential relaxation toward T_slope
            z_deep = abs(z) - 80.0
            T_slope_local = T_slope + 0.25 * δT_h
            return T_cil_edge_local +
                   (T_slope_local - T_cil_edge_local) * (1.0 - exp(-z_deep / h_scale))
        end
    end

    sal_profile = if salinity isa Function
        salinity
    else
        function(lon, lat, z)
            dx = lon_max - lon_min
            x_norm = max(0.0, min(1.0, (lon - lon_min) / dx))
            S_cross = ΔS_cross * x_norm
            # Salinity increases with depth to guarantee static gravitational stability: ∂ρ/∂z ≤ 0
            S_depth = ΔS_deep * (1.0 - exp(-abs(z) / 150.0))
            return S0 + S_cross + S_depth
        end
    end

    o2_arg = if :O2 in keys(model.tracers)
        o2_input = get(kwargs, :oxygen, nothing)
        if o2_input isa Function
            o2_input
        elseif o2_input isa Real
            (lon, lat, z) -> Float64(o2_input)
        else
            function(lon, lat, z)
                dx = lon_max - lon_min
                x_norm = max(0.0, min(1.0, (lon - lon_min) / dx))
                # Well-oxygenated surface & CIL (~300 umol/kg), deeper slope/basin depletion
                o2_surf = 305.0 - 15.0 * x_norm
                o2_dep = 100.0 * (1.0 - exp(-abs(Float64(z)) / 150.0))
                return o2_surf - o2_dep
            end
        end
    else
        nothing
    end

    if !isnothing(o2_arg)
        set!(model, T = temp_profile, S = sal_profile, O2 = o2_arg)
    else
        set!(model, T = temp_profile, S = sal_profile)
    end
    return nothing
end

"""
    set_initial_conditions!(model; kwargs...)

Flexible setter for initializing arbitrary model fields (velocities, tracers).

# Inputs
- `model`: Oceananigans model instance.
- `kwargs...`: Named fields and corresponding functions, arrays, or constants.
"""
function set_initial_conditions!(model; kwargs...)
    set!(model; kwargs...)
    return nothing
end
