# GeoOceanModel

A regional ocean hydrodynamic model that transforms geospatial environmental configurations into realized three-dimensional ocean circulation fields, persisting results directly into the GeoData layer as self-contained Zarr stores.

## Architectural Overview

`GeoOceanModelRun.jl` orchestrates modular execution segments implemented in [`src/driver.jl`](file:///c:/home/jae/projects/GeoOceanModel/src/driver.jl):

- **Data Ingestion (`--data`)**: Assimilates high-resolution bathymetric soundings, coastline polygons, surface winds, and open-boundary hydrography via [`GeoDataSources`](file:///c:/home/jae/projects/GeoDataSources/README.md).
- **Grid & Immersed Boundary (`--grid`)**: Constructs spherical latitude-longitude grids on CPU or GPU (CUDA), fitting real seafloor topography as immersed boundaries.
- **Hydrodynamic Model (`--model`)**: Assembles hydrostatic free-surface primitive equation models with TEOS-10 equation of state, Coriolis accelerations, tidal body forcing, atmospheric fluxes, and open boundary sponge relaxation.
- **Numerical Simulation (`--sim`)**: Advances baroclinic and barotropic modes adaptively, logging Courant numbers and streaming flow snapshots to Zarr.

## Running

```bash
julia --project=. GeoOceanModelRun.jl --all                       # configs/default.toml
julia --project=. GeoOceanModelRun.jl --all --config=scotian_shelf
julia --project=. GeoOceanModelRun.jl --grid --config=scotian_shelf
```

Segment flags (`--data`, `--grid`, `--model`, `--sim`, `--all`) execute a designated stage and its prerequisites.

## Grid Architecture & Coordinate Systems

Computational grids are constructed via [`build_shelf_grid`](file:///c:/home/jae/projects/GeoOceanModel/src/data/grid_bathymetry.jl#L22) as a single coordinate topology:

- **Standard Geographic Grid**:
  - *Conceptual Overview*: A conventional rectangular box aligned north-south and east-west along lines of latitude and longitude. Ideal for localized domains (e.g. Cape Breton demonstration) where the coast is relatively straight or aligned with cardinal directions.
  - *Technical Formulation*: Regular spherical `LatitudeLongitudeGrid` bounded by `[domain]` limits `lon_min`, `lon_max`, `lat_min`, `lat_max`.

### Advection Formulations

- *Conceptual Overview*: When grid lines follow curved paths rather than straight meridians, fluid moving across the grid experiences geometric coordinate curvature. The numerical equations must account for this curvature so that ocean currents do not artificially gain or lose energy.
- *Technical Formulation*: The Oceananigans hydrostatic free-surface solver uses fifth-order `WENO()` advection for momentum and for the active temperature and salinity tracers. On a spherical grid the metric terms that carry the curvature are part of the grid itself, so the conservative form of the momentum equation is advected directly.

## Lateral Boundary Relaxation & Sponge Architecture

### Conceptual Overview (Generalists)

Regional ocean models simulate a localized section of the ocean rather than the entire planet. At the open boundaries of the computational domain, two physical processes must occur simultaneously:
- Outgoing waves, tides, and eddies must leave the domain smoothly without reflecting off artificial edges back into the interior.
- Inflow currents and water masses (such as the cold, fresh Labrador Current entering from the east) must be introduced smoothly from broader regional observations or ocean reanalyses (e.g., Copernicus GLORYS12V1).

To achieve this, the model applies a peripheral buffer zone known as a sponge layer (Price & Aumont 2011). Inside this sponge band, prognostic variables are nudged toward external boundary states with a damping weight $\gamma$ that increases quadratically from $\gamma = 0$ in the interior to $\gamma = 1$ at the boundary.

On a geographic grid aligned with lines of latitude and longitude, the sponge band is specified with simple bounding coordinates (`lat_min`, `lat_max`, `lon_min`, `lon_max`). [`CartesianSpongeMask`](file:///c:/home/jae/projects/GeoOceanModel/src/model/hydrodynamic_model.jl) evaluates the quadratic damping function in those coordinates, which is exactly the grid the model runs on.

### Mathematical Formulation & Numerical Implementation (Specialists)

#### Quadratic Sponge Damping

The quadratic sponge weight $\gamma \in [0, 1]$ is computed in geographic coordinates $(\lambda, \phi)$. For an active southern boundary with lower coordinate limit $\phi_{\min}$ and sponge layer width $L_s$:

$$\gamma_s(\phi) = \begin{cases}
1, & \phi \le \phi_{\min} \\
\left(\frac{\phi_{\min} + L_s - \phi}{L_s}\right)^2, & \phi_{\min} < \phi < \phi_{\min} + L_s \\
0, & \phi \ge \phi_{\min} + L_s
\end{cases}$$

Similarly, for an active eastern boundary with upper coordinate limit $\lambda_{\max}$:

$$\gamma_e(\lambda) = \begin{cases}
1, & \lambda \ge \lambda_{\max} \\
\left(\frac{\lambda - (\lambda_{\max} - L_s)}{L_s}\right)^2, & \lambda_{\max} - L_s < \lambda < \lambda_{\max} \\
0, & \lambda \le \lambda_{\max} - L_s
\end{cases}$$

The composite lateral mask combines all active boundary contributions:

$$\gamma(\lambda, \phi) = \max\left(\gamma_e(\lambda), \gamma_w(\lambda), \gamma_n(\phi), \gamma_s(\phi)\right)$$

In Oceananigans, lateral relaxation is introduced as a tendency term:

$$\mathcal{F}_\psi = -\frac{\gamma(\lambda, \phi)}{\tau_{\text{relax}}} \left(\psi - \psi_{\text{target}}\right)$$

where $\psi \in \{u, v, T, S\}$ and $\tau_{\text{relax}}$ is the relaxation timescale (default 3600 seconds).

#### Boundary Velocity: Observed Current vs Prescribed Constant

The momentum sponge and the tracer sponge have different targets, and the difference matters.

The **tracer** sponge relaxes temperature and salinity toward the observed boundary state, and
has always done so. The **momentum** sponge relaxes toward a *prescribed velocity*: until
now, a single uniform constant (`u_inflow`, `v_inflow`) applied over the whole open face,
with a vertical decay. For the Scotian Shelf that constant stood in for the Nova Scotia
Current — a southwestward, slope-following jet confined to the 200–400 m layer — and a
uniform westward velocity over a 400 m face is not that.

When the boundary source carries currents, the momentum sponge now relaxes toward the
**observed** velocity instead. GLORYS12V1 supplies `uo` and `vo` alongside `thetao` and
`so`; they are fetched with the same subset, interpolated on the same axes, and handed to
the model as `boundary_velocities`. `u_inflow`/`v_inflow` remain the fallback for a
source with no currents (WOA23 has none) and are used only when there are none.

Two implementation details are not optional:

- The target must be a **callable**, not a field. A `Relaxation` whose target is a `Field`
  computes an *exactly zero* tendency for a face-located velocity field in this
  Oceananigans build, whatever the field contains. The same `Relaxation` with a field
  target works for a centre-located tracer, which is why the tracer sponge uses one and
  the momentum sponge cannot. A boundary relaxation that silently does nothing is worse
  than one that is absent, so a field is rejected with an error rather than accepted.
- The tidal velocity is added to the boundary velocity **inside one relaxation**, because
  the boundary carries the sum. Two separate relaxations would settle on the *average* of
  the two targets, halving the tidal amplitude at the boundary.

#### GPU Device Kernel Lowering

Passing closures that capture heap-allocated objects or variables of abstract type prevents CUDA LLVM IR generation. [`CartesianSpongeMask`](file:///c:/home/jae/projects/GeoOceanModel/src/model/hydrodynamic_model.jl) is declared as a concrete, immutable `isbits` struct. With `Adapt.adapt_structure(to, s) = s`, the functor lowers into device registers and executes inline without GPU scalar indexing or memory allocation.

#### Barotropic Wave Stability & Substepping

In shallow shelf and slope waters, barotropic external gravity waves propagate at phase speed:

$$c = \sqrt{g H}$$

At the shelf break floor depth $H = 350\,\text{m}$, $c \approx 58.6\,\text{m/s}$. Across a cross-shelf grid spacing $\Delta y \approx 3,380\,\text{m}$, the barotropic Courant number satisfies:

$$\text{CFL}_{\text{wave}} = \frac{c \, \Delta t}{\Delta y} \approx 0.0173 \, \Delta t$$

Oceananigans employs a split-explicit free-surface solver ([`SplitExplicitFreeSurface`](file:///c:/home/jae/projects/GeoOceanModel/src/model/hydrodynamic_model.jl)) where the 2D barotropic equations are integrated over $N_{\text{sub}}$ substeps per baroclinic time step $\Delta t$. When $\Delta t \ge 20.0\,\text{s}$ with $\text{cfl} = 0.7$, $N_{\text{sub}}$ drops to 1, disabling the barotropic substepping filter over steep topographic slopes. Restricting $\Delta t_{\max} \le 12.0\text{--}15.0\,\text{s}$ maintains active substepping averaging and keeps $\text{CFL}_{\text{wave}} \le 0.21\text{--}0.26$, preventing forward-backward acoustic-gravitational resonance.

## Vertical Discretization & Standardized Depth Conventions

### Intuitive Overview (Generalists)

In oceanography and coastal management, water depth is universally described as a positive distance below the surface (e.g. "a water depth of 20 meters", "a shelf break at 200 meters"). Conversely, physical ocean models use a vertical coordinate axis $z$ where $z = 0$ is the sea surface and underwater positions are negative elevations ($z = -20\,\text{m}$, $z = -200\,\text{m}$).

Past configurations mixed positive and negative numbers (e.g. `inshore_depth = -20.0`, but `shelf_slope = 500.0` and `max_save_depth = 400.0`), creating human error and confusion. To eliminate this ambiguity:
- All user-facing configuration options represent physical depths as positive quantities ($D \ge 0$).
- The model converts these depths to negative vertical elevations ($z = -D$) internally.
- Existing legacy configuration files with negative values are automatically and seamlessly converted via absolute value normalization, preventing crashes.

### Technical Parameter Glossary (Specialists)

- **`depths`**: Monotonically increasing vector of positive vertical face depths in meters starting at $0.0\,\text{m}$ (surface) to domain floor $z_{\min}$ (e.g. $[0.0, 15.0, 30.0, \dots, 350.0]\,\text{m}$). Explicit depths bypass analytical stretching functions. Face thicknesses $\Delta z$ are clustered in the photic mixed layer ($\Delta z = 15\,\text{m}$) and expand toward the seabed ($\Delta z = 40\,\text{m}$), satisfying both the vertical shear Courant condition $\Delta t \le \Delta z_{\min} / |w|$ and thermocline resolution.
- **`vertical_break_depth`**: Positive depth in meters ($200.0\,\text{m}$ for Scotian Shelf, $400.0\,\text{m}$ default) identifying the physical shelf-slope break where the continental terrace terminates and plunges into the continental slope. In `two_segment` stretching, this splits the column into a refined upper stratum and coarse deep stratum.
- **`inshore_depth`**: Positive water depth in meters ($20.0\,\text{m}$) pinning the landward boundary. Enforces a non-zero water column floor ($H \ge 20\,\text{m}$) to prevent dry cliff singularities and unphysical shoreline velocity shears.
- **`shelf_slope`**: Total seabed relief scale across the continental shelf in meters ($D_{\text{break}} - D_{\text{inshore}} = 200.0 - 20.0 = 180.0\,\text{m}$).
- **`max_save_depth`**: Maximum positive depth in meters ($400.0\,\text{m}$) subsampled for Zarr storage. Eliminates deep inactive abyssal layers from disk output while retaining complete epipelagic and mesopelagic larval habitat fields.

### Defensive Backward Compatibility

In [`src/config/conversion.jl`](file:///c:/home/jae/projects/GeoOceanModel/src/config/conversion.jl) and [`HydrodynamicOptions`](file:///c:/home/jae/projects/GeoOceanModel/src/config/options.jl#L110), depth inputs are normalized defensively via `abs(Float64(...))`. In [`src/driver.jl`](file:///c:/home/jae/projects/GeoOceanModel/src/driver.jl#L408), the vertical break elevation evaluates $z_{\text{break}} = -\operatorname{abs}(D_{\text{break}})$, ensuring consistent execution regardless of input sign convention.

## Topography & Immersed Boundaries

Seafloor topography is ingested from high-resolution bathymetric soundings and mapped to the computational grid via [`build_immersed_grid_from_real_data`](file:///c:/home/jae/projects/GeoOceanModel/src/data/grid_bathymetry.jl#L484):

- **Bilinear Regridding**: Cell centres lie on a longitude-latitude lattice, so topography is interpolated with a bilinear (`regrid_2d_field`) remap onto the grid's interior centres, evaluated on CPU before GPU array transfer.
- **Topographic Smoothing & Floors**: Bathymetry is clamped by minimum water depth $H_{\min} = \text{inshore\_depth}$ ($20.0\,\text{m}$) and maximum depth $H_{\max}$ ($1,500\,\text{m}$ or domain depth). Bounding maximum depth constrains the maximum barotropic gravity wave speed $c = \sqrt{g H_{\max}} \le 121\,\text{m/s}$, ensuring stable split-explicit free-surface subcycling.
- **Hypertidal Masking**: Optional masking of the upper Bay of Fundy (`mask_bay_of_fundy = true`) sets hypertidal shallows to solid land, dampening non-linear tidal resonance and spurious vertical velocity spikes ($> 1\,\text{m/s}$) outside snow crab habitat.

### Tide Handling: GeoOceanModel vs GeoLarvalTracking

Tides are applied in GeoOceanModel via TPXO9 boundary forcing (placement = `:boundary`), which relaxes
toward TPXO9 tidal velocity at the open boundaries only. The barotropic tidal wave then propagates
inward dynamically through the shallow-water equations. This avoids the resonance that a uniform
analytic body forcing caused (8–13 m/s divergence within 6 hours).

Because GeoOceanModel's stored flow field now carries the real tidal cycle, **GeoLarvalTracking must NOT
superpose analytic tidal velocity on top** — that would double-count tide. The `[tides]` section in
GeoLarvalTracking TOML configs should have `enable_tides = false` when the ocean model already carries
tide from GeoOceanModel.

The `[tides]` section in GeoOceanModel TOML configs controls:
- `enable_tides`: Whether TPXO9 tidal boundary forcing is active
- `tides_source`: "tpxo9" for TPXO9 global tidal atlas
- `tidal_placement`: ":boundary" (edge-only relaxation, the physical default)
- `tidal_relaxation_timescale`: Relaxation timescale at open edges (seconds)
- `tidal_constituents`: Active constituents (e.g. ["M2", "S2"])

## Configuration Profiles

Configuration files are consolidated in [`configs/`](file:///c:/home/jae/projects/GeoOceanModel/configs):

| Configuration | Grid Type | Resolution / Size | Primary Data Sources | Purpose |
|---|---|---|---|---|
| [`scotian_shelf.toml`](file:///c:/home/jae/projects/GeoOceanModel/configs/scotian_shelf.toml) | Standard | $400 \times 120 \times 16$ ($400\,\text{m}$ floor) | NWA soundings, GLORYS12V1, ERA5 | Snow Crab production run along the Scotian Shelf margin |
| [`default.toml`](file:///c:/home/jae/projects/GeoOceanModel/configs/default.toml) | Standard | $32 \times 38 \times 8$ ($1500\,\text{m}$ floor) | ETOPO2022, WOA23, Open-Meteo | Lightweight keyless Cape Breton demonstration |

### Shared Lakehouse Configuration

`GeoOceanModel` reads central data products from `lakehouse/GeoData/` and archives simulation stores to `lakehouse/GeoOceanModel/`. The central root is resolved via `GeoData.lakehouse_dir()` in order of precedence:
- Environment variables: `GEOCEAN_STORE_ROOT` or `GEODATA_LAKEHOUSE`.
- Local project configuration: `LocalPreferences.toml` (`[GeoData] lakehouse_dir`).
- User configuration: `~/.geodata/storage.toml` (`[storage] store_root`).
- Proximity fallback: Relative path `../lakehouse/GeoOceanModel`.

Persistent lakehouse path configuration:
```julia
using GeoData
GeoData.set_lakehouse_dir!("/path/to/shared/lakehouse")
```

## Output & Data Handover

Simulations produce a directory-based Zarr archive `hydrodynamics.zarr`:
- **Variables**: Three-dimensional prognostic fields $u, v, w, T, S$ at each output cadence interval (`output_schedule_seconds`).
- **Coordinate Attributes**: Stored in root metadata (`.zattrs`) and `time.json`, capturing the 1-D longitude and latitude cell centres alongside vertical layer centres and timestamps.
- **State Checkpoints**: Oceananigans JLD2 state checkpoints enable seamless auto-restart from intermediate timesteps.

## Interactive Notebooks

Pluto interactive documentation and inspection notebooks are located in [`notebooks/`](file:///c:/home/jae/projects/GeoOceanModel/notebooks):
- [`notebooks/scotian_shelf.jl`](file:///c:/home/jae/projects/GeoOceanModel/notebooks/scotian_shelf.jl): Production domain, bathymetric filtering, and full hydrodynamics.
- [`notebooks/default.jl`](file:///c:/home/jae/projects/GeoOceanModel/notebooks/default.jl): Keyless demonstration domain.

Run notebooks via:
```bash
julia --project=. -e "using Pluto; Pluto.run()"
```
