# GeoOceanModel

A regional ocean hydrodynamic model that transforms geospatial environmental configurations into realized three-dimensional ocean circulation fields, persisting results directly into the GeoData layer as self-contained Zarr stores.

## Architectural Overview

`GeoOceanModelRun.jl` orchestrates modular execution segments implemented in [`src/driver.jl`](file:///c:/home/jae/projects/GeoOceanModel/src/driver.jl):

- **Data Ingestion (`--data`)**: Assimilates high-resolution bathymetric soundings, coastline polygons, surface winds, and open-boundary hydrography via [`GeoDataSources`](file:///c:/home/jae/projects/GeoDataSources/README.md).
- **Grid & Immersed Boundary (`--grid`)**: Constructs spherical latitude-longitude or rotated orthogonal curvilinear grids on CPU or GPU (CUDA), fitting real seafloor topography as immersed boundaries.
- **Hydrodynamic Model (`--model`)**: Assembles hydrostatic free-surface primitive equation models with TEOS-10 equation of state, Coriolis accelerations, tidal body forcing, atmospheric fluxes, and open boundary sponge relaxation.
- **Climate Perturbation (`--climate`)**: Optional thermal stratification adjustment representing projected marine heatwaves.
- **Numerical Simulation (`--sim`)**: Advances baroclinic and barotropic modes adaptively, logging Courant numbers and streaming flow snapshots to Zarr.

## Running

```bash
julia --project=. GeoOceanModelRun.jl --all                       # configs/default.toml
julia --project=. GeoOceanModelRun.jl --all --config=scotian_shelf
julia --project=. GeoOceanModelRun.jl --grid --config=scotian_shelf
```

Segment flags (`--data`, `--grid`, `--model`, `--climate`, `--sim`, `--all`) execute a designated stage and its prerequisites.

## Grid Architecture & Coordinate Systems

Computational grids are constructed via [`build_shelf_grid`](file:///c:/home/jae/projects/GeoOceanModel/src/data/grid_bathymetry.jl#L22), supporting two coordinate topologies:

- **Standard Geographic Grid (`orientation = "standard"`)**:
  - *Conceptual Overview*: A conventional rectangular box aligned north-south and east-west along lines of latitude and longitude. Ideal for localized domains (e.g. Cape Breton demonstration) where the coast is relatively straight or aligned with cardinal directions.
  - *Technical Formulation*: Regular spherical `LatitudeLongitudeGrid` bounded by `[domain]` limits `lon_min`, `lon_max`, `lat_min`, `lat_max`.

- **Rotated Orthogonal Curvilinear Grid (`orientation = "rotated"`)**:
  - *Conceptual Overview*: The Atlantic Canadian continental margin runs diagonally from southwest to northeast (roughly $50^\circ$ clockwise from North). A standard north-south box simulating this region must include vast areas of deep abyssal ocean to the south and dry mainland Canada to the north, wasting substantial computing power. By rotating the computational coordinate frame parallel to the coastline and shelf edge, the simulation aligns with the natural geography, covering the entire habitat corridor from Georges Bank to Cabot Strait with roughly 60% fewer horizontal grid cells.
  - *Technical Formulation*: Orthogonal curvilinear `RotatedLatitudeLongitudeGrid` constructed on an unrotated sphere whose North Pole has been repositioned so that the equator aligns with the target shelf strike azimuth $\theta = 50.0^\circ$. Centered at $(\lambda_0, \phi_0) = (-63.0^\circ\text{E}, 44.0^\circ\text{N})$ with along-shelf extent $L_\xi = 850.0\,\text{km}$ and cross-shelf extent $L_\eta = 220.0\,\text{km}$, discretized into $Nx = 220 \times Ny = 65 = 14,300$ horizontal cells (versus $240 \times 150 = 36,000$ cells for an unrotated bounding box).

### Euler Pole Derivation (Specialist Details)

For rotated grids, the geographic coordinates of the rotated North Pole $(\lambda_p, \phi_p)$ are derived in [`compute_rotated_pole`](file:///c:/home/jae/projects/GeoOceanModel/src/data/grid_bathymetry.jl#L142).

Let $(\lambda_0, \phi_0)$ be the domain center in spherical coordinates and $\theta$ be the shelf strike azimuth measured clockwise from true North. The orthonormal local geographic tangent unit vectors are:

$$\hat{e}_E = [-\sin\lambda_0, \cos\lambda_0, 0]^T, \quad \hat{e}_N = [-\sin\phi_0\cos\lambda_0, -\sin\phi_0\sin\lambda_0, \cos\phi_0]^T$$

The along-shelf tangent vector directed toward azimuth $\theta$ is:

$$\hat{e}_\xi = \sin\theta\,\hat{e}_E + \cos\theta\,\hat{e}_N$$

The orthogonal cross-shelf normal vector directed into the open ocean is:

$$\hat{e}_\eta = \cos\theta\,\hat{e}_E - \sin\theta\,\hat{e}_N$$

The Euler rotation pole vector $\mathbf{P} = [P_x, P_y, P_z]^T$ defining the axis of the rotated sphere is the right-handed unit normal $\mathbf{P} = \hat{e}_\xi \times \hat{e}_\eta$, yielding:

$$\lambda_p = \operatorname{atan2}(P_y, P_x), \quad \phi_p = \operatorname{asin}(P_z)$$

Given mean Earth radius $R = 6,371\,\text{km}$, the unrotated angular coordinate half-widths $(\Delta \lambda', \Delta \phi')$ centered on the rotated equator ($\phi' = 0$) are:

$$\Delta \lambda' = \frac{L_\xi}{2 R}, \quad \Delta \phi' = \frac{L_\eta}{2 R}$$

### Advection Formulations

- *Conceptual Overview*: When grid lines follow curved paths rather than straight meridians, fluid moving across the grid experiences geometric coordinate curvature. The numerical equations must account for this curvature so that ocean currents do not artificially gain or lose energy.
- *Technical Formulation*: The Oceananigans hydrostatic free-surface solver requires `VectorInvariant()` momentum advection on curvilinear grids. This formulation discretizes the advection term as $\boldsymbol{u} \cdot \nabla \boldsymbol{u} = \boldsymbol{\omega} \times \boldsymbol{u} + \frac{1}{2}\nabla|\boldsymbol{u}|^2$, computing relative vorticity $\boldsymbol{\omega} = \nabla \times \boldsymbol{u}$ directly on cell vertices without Christoffel symbol singularities. The model automatically switches momentum advection to `VectorInvariant()` on `OrthogonalSphericalShellGrid` instances, while preserving fifth-order `WENO()` advection for active temperature and salinity tracers.

## Lateral Boundary Relaxation & Curvilinear Sponge Architecture

### Conceptual Overview (Generalists)

Regional ocean models simulate a localized section of the ocean rather than the entire planet. At the open boundaries of the computational domain, two physical processes must occur simultaneously:
- Outgoing waves, tides, and eddies must leave the domain smoothly without reflecting off artificial edges back into the interior.
- Inflow currents and water masses (such as the cold, fresh Labrador Current entering from the east) must be introduced smoothly from broader regional observations or ocean reanalyses (e.g., Copernicus GLORYS12V1).

To achieve this, the model applies a peripheral buffer zone known as a sponge layer (Price & Aumont 2011). Inside this sponge band, prognostic variables are nudged toward external boundary states with a damping weight $\gamma$ that increases quadratically from $\gamma = 0$ in the interior to $\gamma = 1$ at the boundary.

On a standard geographic grid aligned with lines of latitude and longitude, a sponge layer is specified using simple bounding coordinates (`lat_min`, `lat_max`, `lon_min`, `lon_max`). However, when the computational domain is rotated by $50^\circ$ to follow the Scotian Shelf margin, lines of constant geographic latitude intersect the domain diagonally. Using geographic bounding boxes on a rotated domain truncates sponge damping across roughly 95% of the southern boundary, leaving the open shelf break exposed to wave reflection while concentrating damping into a narrow corner triangle. The resulting shear discontinuity causes numerical divergence within hours.

[`RotatedSpongeMask`](file:///c:/home/jae/projects/GeoOceanModel/src/model/hydrodynamic_model.jl) resolves this issue by rotating geographic coordinates $(\lambda, \phi)$ back into curvilinear coordinates $(\lambda', \phi')$ before evaluating the quadratic damping function. This aligns the sponge layer with the true curvilinear perimeter of the grid, providing uniform, seamless boundary damping across all designated open boundaries.

### Mathematical Formulation & Numerical Implementation (Specialists)

#### Inverse Spherical Euler Rotation

Let $(\lambda_p, \phi_p)$ denote the geographic coordinates of the rotated North Pole, and let $\phi_p^* = \frac{\pi}{2} - \phi_p$ be the co-latitude rotation angle. For an arbitrary grid face or center point with geographic longitude $\lambda$ and latitude $\phi$, the corresponding unit Cartesian vector in the Earth-centered inertial frame is:

$$\mathbf{x} = \begin{bmatrix} \cos\phi \cos\lambda \\ \cos\phi \sin\lambda \\ \sin\phi \end{bmatrix}$$

The transformation to rotated curvilinear coordinates $(\lambda', \phi')$ proceeds through two successive rotations:
- A rotation by angle $\lambda_p$ about the $Z$-axis:
  $$\mathbf{x}_1 = \begin{bmatrix} \cos\lambda_p & \sin\lambda_p & 0 \\ -\sin\lambda_p & \cos\lambda_p & 0 \\ 0 & 0 & 1 \end{bmatrix} \mathbf{x}$$
- A rotation by angle $\phi_p^*$ about the $Y$-axis:
  $$\mathbf{x}' = \begin{bmatrix} \cos\phi_p^* & 0 & -\sin\phi_p^* \\ 0 & 1 & 0 \\ \sin\phi_p^* & 0 & \cos\phi_p^* \end{bmatrix} \mathbf{x}_1$$

The curvilinear coordinates $(\lambda', \phi')$ are then recovered via:

$$\lambda' = \operatorname{atan2}(x'_y, x'_x), \quad \phi' = \arcsin(x'_z)$$

#### Quadratic Sponge Damping

The quadratic sponge weight $\gamma \in [0, 1]$ is computed natively in rotated coordinates $(\lambda', \phi')$. For an active southern boundary with lower coordinate limit $\phi'_{\min}$ and sponge layer width $L_s$:

$$\gamma_s(\phi') = \begin{cases}
1, & \phi' \le \phi'_{\min} \\
\left(\frac{\phi'_{\min} + L_s - \phi'}{L_s}\right)^2, & \phi'_{\min} < \phi' < \phi'_{\min} + L_s \\
0, & \phi' \ge \phi'_{\min} + L_s
\end{cases}$$

Similarly, for an active eastern boundary with upper coordinate limit $\lambda'_{\max}$:

$$\gamma_e(\lambda') = \begin{cases}
1, & \lambda' \ge \lambda'_{\max} \\
\left(\frac{\lambda' - (\lambda'_{\max} - L_s)}{L_s}\right)^2, & \lambda'_{\max} - L_s < \lambda' < \lambda'_{\max} \\
0, & \lambda' \le \lambda'_{\max} - L_s
\end{cases}$$

The composite lateral mask combines all active boundary contributions:

$$\gamma(\lambda', \phi') = \max\left(\gamma_e(\lambda'), \gamma_w(\lambda'), \gamma_n(\phi'), \gamma_s(\phi')\right)$$

In Oceananigans, lateral relaxation is introduced as a tendency term:

$$\mathcal{F}_\psi = -\frac{\gamma(\lambda', \phi')}{\tau_{\text{relax}}} \left(\psi - \psi_{\text{target}}\right)$$

where $\psi \in \{u, v, T, S\}$ and $\tau_{\text{relax}}$ is the relaxation timescale (default 3600 seconds).

#### GPU Device Kernel Lowering

Passing closures that capture heap-allocated objects or variables of abstract type prevents CUDA LLVM IR generation. Both [`CartesianSpongeMask`](file:///c:/home/jae/projects/GeoOceanModel/src/model/hydrodynamic_model.jl) and [`RotatedSpongeMask`](file:///c:/home/jae/projects/GeoOceanModel/src/model/hydrodynamic_model.jl) are declared as concrete, immutable `isbits` structs with precomputed trigonometric constants ($\cos\lambda_p, \sin\lambda_p, \cos\phi_p^*, \sin\phi_p^*$). With `Adapt.adapt_structure(to, s) = s`, the functors lower into device registers and execute inline without GPU scalar indexing or memory allocation.

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

- **Curvilinear 2D Bilinear Regridding**: For rotated grids, cell centers $(\lambda_{i,j}, \phi_{i,j})$ form an unstructured non-Cartesian coordinate matrix. Topography is interpolated using optimized 2D bilinear interpolation (`_regrid_curvilinear_2d`) evaluated on CPU before GPU array transfer, eliminating host-device scalar indexing stalls.
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
| [`scotian_shelf.toml`](file:///c:/home/jae/projects/GeoOceanModel/configs/scotian_shelf.toml) | Rotated ($50^\circ$) | $220 \times 65 \times 12$ ($350\,\text{m}$ floor) | NWA soundings, GLORYS12V1, ERA5 | Snow Crab production run along the Scotian Shelf margin |
| [`default.toml`](file:///c:/home/jae/projects/GeoOceanModel/configs/default.toml) | Standard | $50 \times 50 \times 8$ ($1500\,\text{m}$ floor) | ETOPO2022, WOA23, Open-Meteo | Lightweight keyless Cape Breton demonstration |
| [`eastern_scotian_shelf.toml`](file:///c:/home/jae/projects/GeoOceanModel/configs/eastern_scotian_shelf.toml) | Standard | $140 \times 100 \times 12$ | NWA soundings, GLORYS12V1, ERA5 | High-resolution regional Eastern Scotian Shelf domain |
| [`western_scotian_shelf.toml`](file:///c:/home/jae/projects/GeoOceanModel/configs/western_scotian_shelf.toml) | Standard | $100 \times 70 \times 9$ | NWA soundings, GLORYS12V1, ERA5 | Regional Western Scotian Shelf / Browns Bank domain |

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
- **Coordinate Attributes**: Stored in root metadata (`.zattrs`) and `time.json`, capturing 2D curvilinear cell centres $(\lambda_{i,j}, \phi_{i,j})$ or 1D coordinates, alongside vertical layer centres and timestamps.
- **State Checkpoints**: Oceananigans JLD2 state checkpoints enable seamless auto-restart from intermediate timesteps.

## Interactive Notebooks

Pluto interactive documentation and inspection notebooks are located in [`notebooks/`](file:///c:/home/jae/projects/GeoOceanModel/notebooks):
- [`notebooks/scotian_shelf.jl`](file:///c:/home/jae/projects/GeoOceanModel/notebooks/scotian_shelf.jl): Rotated curvilinear production domain, bathymetric filtering, and full hydrodynamics.
- [`notebooks/default.jl`](file:///c:/home/jae/projects/GeoOceanModel/notebooks/default.jl): Keyless demonstration domain.

Run notebooks via:
```bash
julia --project=. -e "using Pluto; Pluto.run()"
```
